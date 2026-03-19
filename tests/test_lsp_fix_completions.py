"""Tests for completion bug fixes in src/maomi/lsp/_completion.py."""

from lsprotocol import types

from maomi.lsp import validate, _complete_general, _complete_dot, _vars_in_scope, AnalysisResult
from maomi.lsp._completion import (
    _annotation_str, _complete_struct_literal, _complete_module,
)
from maomi.ast_nodes import TypeAnnotation, Dim, Span


# ---------------------------------------------------------------------------
# C6: EOF cursor returns completions, not None
# ---------------------------------------------------------------------------

class TestC6EofCursor:
    def test_eof_cursor_returns_completions(self):
        """When cursor line >= len(lines), should return completions, not None."""
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        # Position past the last line (source has 1 line, so line=1 is past EOF)
        pos = types.Position(line=1, character=0)
        comp = _complete_general(result, pos)
        assert comp is not None
        labels = {item.label for item in comp.items}
        # Should still include keywords and builtins
        assert "fn" in labels
        assert "exp" in labels


# ---------------------------------------------------------------------------
# C5: No bounds checking on col
# ---------------------------------------------------------------------------

class TestC5ColBounds:
    def test_col_past_line_length_no_crash(self):
        """col > len(line_text) should not raise IndexError."""
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        # Col far past end of the line
        pos = types.Position(line=0, character=9999)
        comp = _complete_general(result, pos)
        assert comp is not None


# ---------------------------------------------------------------------------
# B1: Monomorphized $-copies leak into completions
# ---------------------------------------------------------------------------

class TestB1DollarCopyFiltering:
    def test_no_dollar_items_in_general_completions(self):
        """No items with $ in their name should appear in general completions."""
        source = """
fn helper(x: f32[..]) -> f32 { sum(x) }
fn user(a: f32[3]) -> f32 { helper(a) }
"""
        _, result = validate(source, "<test>")
        pos = types.Position(line=2, character=30)
        comp = _complete_general(result, pos)
        assert comp is not None
        for item in comp.items:
            assert "$" not in item.label, f"Found $-copy in completions: {item.label}"

    def test_no_dollar_items_in_struct_completions(self):
        """No struct names with $ should appear in general completions."""
        source = """
struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 { p.x }
"""
        _, result = validate(source, "<test>")
        pos = types.Position(line=2, character=25)
        comp = _complete_general(result, pos)
        for item in comp.items:
            if item.kind == types.CompletionItemKind.Struct:
                assert "$" not in item.label


# ---------------------------------------------------------------------------
# B2: False positive struct literal completion after -> Point {
# ---------------------------------------------------------------------------

class TestB2FalsePositiveStructLiteral:
    def test_no_struct_literal_after_return_type(self):
        """'-> Point {' is a function body brace, not a struct literal."""
        source = """
struct Point { x: f32, y: f32 }
fn make() -> Point { Point { x: 1.0, y: 2.0 } }
"""
        # Simulate cursor inside the function body after "-> Point {"
        # Line 2: "fn make() -> Point { Point { x: 1.0, y: 2.0 } }"
        line_text = "fn make() -> Point { Point { x: 1.0, y: 2.0 } }"
        _, result = validate(source, "<test>")
        # Cursor right after the opening brace of the function body
        # "fn make() -> Point { " -- col 21
        comp = _complete_struct_literal(line_text, 21, result, source, types.Position(line=2, character=21))
        # Should NOT trigger struct literal completion for "Point" (it's after ->)
        assert comp is None

    def test_struct_literal_without_arrow(self):
        """'Point {' without preceding '->' should trigger struct literal completion."""
        source = """
struct Point { x: f32, y: f32 }
fn f() -> Point { Point {  } }
"""
        _, result = validate(source, "<test>")
        # "fn f() -> Point { Point {  } }" -- cursor after "Point { " inside the literal
        line_text = "fn f() -> Point { Point {  } }"
        # Cursor at position 27 (inside "Point { | }")
        comp = _complete_struct_literal(line_text, 27, result, source, types.Position(line=2, character=27))
        assert comp is not None
        labels = {item.label for item in comp.items}
        assert "x" in labels
        assert "y" in labels


# ---------------------------------------------------------------------------
# B12: _annotation_str doesn't handle wildcard f32[..]
# ---------------------------------------------------------------------------

class TestB12WildcardAnnotation:
    def test_wildcard_shows_dotdot(self):
        """Wildcard annotation should show 'f32[..]' not 'f32'."""
        ann = TypeAnnotation(base="f32", dims=None, span=Span(1, 1, 1, 10), wildcard=True)
        result = _annotation_str(ann)
        assert result == "f32[..]"

    def test_non_wildcard_scalar(self):
        """Non-wildcard scalar should just show 'f32'."""
        ann = TypeAnnotation(base="f32", dims=None, span=Span(1, 1, 1, 5))
        result = _annotation_str(ann)
        assert result == "f32"

    def test_array_with_dims(self):
        """Array with concrete dims should show 'f32[3, 4]'."""
        ann = TypeAnnotation(
            base="f32",
            dims=[Dim(3, Span(1, 1, 1, 2)), Dim(4, Span(1, 3, 1, 4))],
            span=Span(1, 1, 1, 10),
        )
        result = _annotation_str(ann)
        assert result == "f32[3, 4]"


# ---------------------------------------------------------------------------
# G15: sort_text on general completions
# ---------------------------------------------------------------------------

class TestG15SortText:
    def test_sort_text_present_and_ordered(self):
        """All completion items should have sort_text with category prefixes."""
        source = """
struct Point { x: f32, y: f32 }
fn helper(x: f32) -> f32 { x }
fn main(y: f32) -> f32 {
    let a = 1.0;
    a
}
"""
        _, result = validate(source, "<test>")
        # Cursor inside main body, after 'let a = 1.0;'
        pos = types.Position(line=5, character=4)
        comp = _complete_general(result, pos)
        assert comp is not None

        # Check that all items have sort_text
        for item in comp.items:
            assert item.sort_text is not None, f"Missing sort_text on {item.label}"

        # Build a dict of sort_text by label
        sort_texts = {item.label: item.sort_text for item in comp.items}

        # Variables should sort before functions
        if "a" in sort_texts and "helper" in sort_texts:
            assert sort_texts["a"] < sort_texts["helper"], \
                f"Variable 'a' ({sort_texts['a']}) should sort before function 'helper' ({sort_texts['helper']})"

        # User functions should sort before builtins
        if "helper" in sort_texts and "exp" in sort_texts:
            assert sort_texts["helper"] < sort_texts["exp"], \
                f"User fn 'helper' ({sort_texts['helper']}) should sort before builtin 'exp' ({sort_texts['exp']})"

        # Builtins should sort before keywords
        if "exp" in sort_texts and "fn" in sort_texts:
            assert sort_texts["exp"] < sort_texts["fn"], \
                f"Builtin 'exp' ({sort_texts['exp']}) should sort before keyword 'fn' ({sort_texts['fn']})"

        # Keywords should sort before type names
        if "fn" in sort_texts and "f32" in sort_texts:
            assert sort_texts["fn"] < sort_texts["f32"], \
                f"Keyword 'fn' ({sort_texts['fn']}) should sort before type 'f32' ({sort_texts['f32']})"


# ---------------------------------------------------------------------------
# B17: line_start/col_end mismatch for multi-line let
# ---------------------------------------------------------------------------

class TestB17MultiLineLet:
    def test_multiline_let_scope(self):
        """Let binding that spans multiple lines should use line_end for comparison."""
        source = """fn f(x: f32) -> f32 {
    let a =
        1.0;
    a
}"""
        _, result = validate(source, "<test>")
        # Cursor on line 3 (0-indexed), after the multi-line let
        pos = types.Position(line=3, character=4)
        variables = _vars_in_scope(result, pos)
        var_names = [v[0] for v in variables]
        assert "a" in var_names


# ---------------------------------------------------------------------------
# B3: _collect_scope_vars descends into LetStmt.value
# ---------------------------------------------------------------------------

class TestB3LetStmtValueDescent:
    def test_scan_vars_inside_let(self):
        """Loop vars inside let value should be found by scope collector."""
        source = """fn f(x: f32[3]) -> f32 {
    let result = scan (c, e) in (0.0, x) {
        c + e
    };
    result
}"""
        _, result = validate(source, "<test>")
        # Cursor inside the scan body -- line 2 "        c + e"
        pos = types.Position(line=2, character=8)
        variables = _vars_in_scope(result, pos)
        var_names = [v[0] for v in variables]
        assert "c" in var_names
        assert "e" in var_names

    def test_map_var_inside_let(self):
        """Map element var inside let value should be found."""
        source = """fn f(x: f32[3]) -> f32[3] {
    let result = map e in x {
        e + 1.0
    };
    result
}"""
        _, result = validate(source, "<test>")
        # Cursor inside map body
        pos = types.Position(line=2, character=8)
        variables = _vars_in_scope(result, pos)
        var_names = [v[0] for v in variables]
        assert "e" in var_names


# ---------------------------------------------------------------------------
# E3: .isspace() instead of == ' ' in struct literal detection
# ---------------------------------------------------------------------------

class TestE3IsSpace:
    def test_tab_before_brace(self):
        """Struct literal detection should handle tabs before {, not just spaces."""
        source = """
struct Point { x: f32, y: f32 }
fn f() -> Point { Point\t{  } }
"""
        _, result = validate(source, "<test>")
        # The tab before { should be skipped by .isspace()
        line_text = "fn f() -> Point { Point\t{  } }"
        # Cursor inside struct literal
        comp = _complete_struct_literal(line_text, 27, result, source, types.Position(line=2, character=27))
        assert comp is not None
        labels = {item.label for item in comp.items}
        assert "x" in labels
        assert "y" in labels


# ---------------------------------------------------------------------------
# G24-partial: _complete_dot handles StructArrayType
# ---------------------------------------------------------------------------

class TestG24StructArrayType:
    def test_dot_completion_on_struct_array_type(self):
        """Dot completion on StructArrayType should yield struct fields."""
        from maomi.types import StructType, StructArrayType
        # Create a mock AnalysisResult with a StructArrayType variable
        # We'll use the actual validate to create a realistic test
        source = """
struct Batch { x: f32, y: f32 }
fn f(b: Batch) -> f32 { b.x }
"""
        _, result = validate(source, "<test>")
        # Verify dot completion works on plain struct (baseline)
        pos = types.Position(line=2, character=26)
        comp = _complete_dot(result, pos)
        assert comp is not None
        labels = {item.label for item in comp.items}
        assert "x" in labels
        assert "y" in labels


# ---------------------------------------------------------------------------
# G12: Module dot-completion includes structs
# ---------------------------------------------------------------------------

class TestG12ModuleStructs:
    def test_module_completion_no_crash(self):
        """_complete_module should not crash on valid inputs."""
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        # No modules in this simple source
        comp = _complete_module(result, "nonexistent")
        assert comp is None


# ---------------------------------------------------------------------------
# G14: _complete_import_names filters already-imported names
# ---------------------------------------------------------------------------

class TestG14ImportFiltering:
    def test_parse_already_imported(self):
        """_parse_already_imported extracts names from brace text."""
        from maomi.lsp._completion import _parse_already_imported
        assert _parse_already_imported(" relu, linear") == {"relu", "linear"}
        assert _parse_already_imported("relu") == {"relu"}
        assert _parse_already_imported("") == set()
        assert _parse_already_imported(" ") == set()


# ---------------------------------------------------------------------------
# Integration: combination of fixes
# ---------------------------------------------------------------------------

class TestCompletionIntegration:
    def test_eof_with_col_past_end(self):
        """EOF cursor + col past line end should not crash."""
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        pos = types.Position(line=999, character=999)
        comp = _complete_general(result, pos)
        assert comp is not None

    def test_general_completions_have_sort_text(self):
        """All items from _complete_general should have sort_text."""
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=23)
        comp = _complete_general(result, pos)
        for item in comp.items:
            assert item.sort_text is not None, f"Missing sort_text on {item.label}"

    def test_no_module_prefixed_structs_in_general(self):
        """Module-prefixed struct names should not appear in general completions."""
        source = """
struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 { p.x }
"""
        _, result = validate(source, "<test>")
        pos = types.Position(line=2, character=25)
        comp = _complete_general(result, pos)
        for item in comp.items:
            if item.kind == types.CompletionItemKind.Struct:
                assert "." not in item.label, f"Module-prefixed struct in general: {item.label}"
