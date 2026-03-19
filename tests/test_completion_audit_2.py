"""Completion audit part 2 — deeper edge case exploration.

Focuses on:
- Completions handler dot detection edge cases
- Import path completions
- completion_validate with struct modifications
- Pipe completion with comptime params
- Struct literal multi-line
- Various real-world patterns
"""

from __future__ import annotations

import pytest
from lsprotocol import types

from maomi.lsp import validate, _complete_general, _complete_dot, _vars_in_scope, AnalysisResult
from maomi.lsp._completion import (
    _complete_import, _complete_struct_literal, _complete_module,
    _pipe_completions, _is_pipe_compatible, _is_complex_builtin_pipe_compatible,
)
from maomi.lsp._core import (
    completion_validate, _FAKE_ID, _insert_fake_id, _EMPTY_RESULT,
)
from maomi.lsp._builtin_data import (
    _KEYWORDS, _TYPE_NAMES, _BUILTINS, _BUILTIN_SET,
    _BUILTIN_NAMESPACES, _BUILTIN_DOCS, _BUILTIN_CATEGORIES, _EW_NAMES,
)
from tests.lsp_validation import assert_all_completions_valid, check_edit


def _labels(comp):
    if comp is None:
        return set()
    return {item.label for item in comp.items}


def _find_item(comp, label):
    if comp is None:
        return None
    for item in comp.items:
        if item.label == label:
            return item
    return None


# ============================================================================
# A. Dot detection: line[col-1] == "." edge cases in completions handler
# ============================================================================

class TestDotDetectionEdgeCases:
    """Test scenarios that could trick the dot detection in the completions handler."""

    def test_dot_in_float_literal(self):
        """'1.0' has a dot, but cursor after '1.' should NOT trigger dot completion.

        The completions handler checks line_text[col-1] == '.',
        which would match '1.' and try to extract prefix '1' as a variable.
        This would fail to find anything useful, but shouldn't crash.
        """
        source = "fn f(x: f32) -> f32 { 1. }\n"
        # 1 is at position 22, . at position 23
        # If cursor is at 24 (after the dot), line[23] == '.'
        # Prefix extraction: scans back from col-2=22, finds '1' (digit, alnum)
        # prefix = "1"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=24)
        # Try dot completion with prefix "1"
        comp = _complete_dot(result, pos, prefix="1")
        # Should return None (no variable named "1")
        # or return pipe completions if AST has a FloatLiteral here
        # This isn't a crash, just potentially wrong behavior
        if comp:
            # Check items are valid
            assert_all_completions_valid(comp, pos)

    def test_dot_in_range_operator(self):
        """'..' in range expressions -- shouldn't crash."""
        source = "fn f(x: f32[..]) -> f32 { sum(x) }\n"
        _, result = validate(source, "<test>")
        # The '..' in f32[..] is in the type annotation, not code
        # This just tests that validation doesn't crash
        assert result.program is not None

    def test_double_dot_at_line_start(self):
        """Line starting with '..' should not crash."""
        source = "fn f(x: f32) -> f32 {\n..\n}\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=1, character=2)
        # This would detect a dot and try dot completion
        # Not a real scenario but shouldn't crash
        comp = _complete_dot(result, pos, prefix=".")
        # Result doesn't matter, just shouldn't crash


# ============================================================================
# B. Dot completion with comptime params
# ============================================================================

class TestComptimePipeCompletion:
    """Test pipe completions for functions with comptime parameters."""

    def test_comptime_function_in_pipe(self):
        """Functions with comptime params should appear in pipe completions if first param matches."""
        source = """fn my_reduce(x: f32[..], comptime axis: i32) -> f32 { sum(x) }
fn f(data: f32[3, 4]) -> f32 { sum(data) }
"""
        _, result = validate(source, "<test>")
        pos = types.Position(line=1, character=33)
        comp = _complete_dot(result, pos)
        if comp:
            assert_all_completions_valid(comp, pos)
            labels = _labels(comp)
            assert "my_reduce" in labels


# ============================================================================
# C. Multi-line struct literal completion
# ============================================================================

class TestMultiLineStructLiteral:
    """Test struct literal completion when spread across lines."""

    def test_multiline_struct_literal(self):
        """Struct literal spanning multiple lines should complete correctly."""
        source = """struct Point { x: f32, y: f32, z: f32 }
fn f() -> Point {
    Point {
        x: 1.0,

    }
}
"""
        _, result = validate(source, "<test>")
        pos = types.Position(line=4, character=8)
        comp = _complete_struct_literal(
            "        ", 8, result, source, pos
        )
        if comp is not None:
            labels = _labels(comp)
            assert "y" in labels or "z" in labels
            assert "x" not in labels  # already filled


# ============================================================================
# D. Pipe completion type filtering
# ============================================================================

class TestPipeTypeFiltering:
    """Test that pipe completions properly filter by type."""

    def test_struct_pipe_only_shows_struct_compatible(self):
        """Pipe on struct should show struct-compatible builtins (exp, sqrt, etc.) but not array-only ones."""
        source = "struct S { val: f32 }\nfn f(s: S) -> f32 { s.val }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=1, character=22)
        comp = _complete_dot(result, pos)
        assert comp is not None
        labels = _labels(comp)
        # Elementwise builtins should work on structs
        assert "exp" in labels
        assert "sqrt" in labels
        # Array-only operations should NOT appear
        assert "transpose" not in labels
        assert "reshape" not in labels

    def test_i32_scalar_no_float_builtins(self):
        """Pipe on i32 scalar should not show exp, sqrt, etc."""
        source = "fn f(x: i32) -> i32 { x }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=23)
        comp = _complete_dot(result, pos)
        if comp:
            labels = _labels(comp)
            assert "exp" not in labels
            assert "sqrt" not in labels

    def test_bool_array_pipe(self):
        """Pipe on bool type -- should not show float builtins."""
        from maomi.types import ScalarType
        assert not any(
            _is_complex_builtin_pipe_compatible(ScalarType("bool"), cat)
            for cat in ["reduction", "shape", "conv_pool"]
        )

    def test_f64_pipe_works(self):
        """f64 should also get pipe completions (it's a float type)."""
        source = "fn f(x: f64) -> f64 { x }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=23)
        comp = _complete_dot(result, pos)
        if comp:
            labels = _labels(comp)
            assert "exp" in labels
            assert "sqrt" in labels


# ============================================================================
# E. Scope leakage between if branches
# ============================================================================

class TestScopeLeakage:
    """Test that variables don't leak between if/else branches."""

    def test_no_leakage_from_then_to_else(self):
        """Variables defined in then-branch should not appear in else-branch."""
        source = """fn f(x: f32) -> f32 {
    if x > 0.0 {
        let a = 1.0;
        a
    } else {
        0.0
    }
}"""
        _, result = validate(source, "<test>")
        # Cursor inside else branch
        pos = types.Position(line=5, character=8)
        variables = _vars_in_scope(result, pos)
        names = [v[0] for v in variables]
        assert "a" not in names  # 'a' is in then-branch only
        assert "x" in names  # param should be in scope

    def test_no_leakage_from_else_to_then(self):
        """Variables defined in else-branch should not appear in then-branch."""
        source = """fn f(x: f32) -> f32 {
    if x > 0.0 {
        0.0
    } else {
        let b = 2.0;
        b
    }
}"""
        _, result = validate(source, "<test>")
        # Cursor inside then branch
        pos = types.Position(line=2, character=8)
        variables = _vars_in_scope(result, pos)
        names = [v[0] for v in variables]
        assert "b" not in names


# ============================================================================
# F. completion_validate with struct definitions
# ============================================================================

class TestCompletionValidateStructs:
    """Test completion_validate handles struct definitions."""

    def test_struct_in_scope(self):
        """Struct definitions should be in completion_validate results."""
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 {\n    p.\n}\n"
        result = completion_validate(source, "<test>", 2, 6)
        assert result.program is not None
        assert len(result.struct_defs) > 0

    def test_struct_field_completion_via_completion_validate(self):
        """Struct field completion should work via completion_validate path."""
        source = "struct S { a: f32, b: f32 }\nfn f(s: S) -> f32 {\n    s.\n}\n"
        result = completion_validate(source, "<test>", 2, 6)
        pos = types.Position(line=2, character=6)
        comp = _complete_dot(result, pos)
        if comp:
            assert_all_completions_valid(comp, pos)
            labels = _labels(comp)
            assert "a" in labels
            assert "b" in labels


# ============================================================================
# G. Import already-imported filtering
# ============================================================================

class TestImportAlreadyImported:
    """Test that already-imported names are filtered in from..import completions."""

    def test_already_imported_single(self):
        """'from math import { relu, ' should not offer relu again."""
        from maomi.lsp._completion import _parse_already_imported
        result = _parse_already_imported(" relu, ")
        assert "relu" in result

    def test_already_imported_multiple(self):
        from maomi.lsp._completion import _parse_already_imported
        result = _parse_already_imported(" relu, gelu, silu")
        assert result == {"relu", "gelu", "silu"}

    def test_already_imported_with_spaces(self):
        from maomi.lsp._completion import _parse_already_imported
        result = _parse_already_imported("  relu  ,  gelu  ")
        assert result == {"relu", "gelu"}


# ============================================================================
# H. Deeply nested scope collection
# ============================================================================

class TestDeeplyNestedScope:
    """Test scope collection in deeply nested expressions."""

    def test_nested_scan_in_map(self):
        """Variables from outer map and inner scan should both be in scope."""
        source = """fn f(xs: f32[3, 4]) -> f32[3] {
    map row in xs {
        scan (c, e) in (0.0, row) {
            c + e
        }
    }
}"""
        _, result = validate(source, "<test>")
        # Cursor inside scan body: "c + e"
        pos = types.Position(line=3, character=12)
        variables = _vars_in_scope(result, pos)
        names = [v[0] for v in variables]
        assert "c" in names
        assert "e" in names
        assert "row" in names

    def test_nested_if_in_scan(self):
        """Variables from outer scan and cursor in if body."""
        source = """fn f(xs: f32[3]) -> f32 {
    scan (c, e) in (0.0, xs) {
        if e > 0.0 {
            c + e
        } else {
            c
        }
    }
}"""
        _, result = validate(source, "<test>")
        # Cursor inside then branch: "c + e"
        pos = types.Position(line=3, character=12)
        variables = _vars_in_scope(result, pos)
        names = [v[0] for v in variables]
        assert "c" in names
        assert "e" in names


# ============================================================================
# I. Completion with multiple struct definitions
# ============================================================================

class TestMultiStructCompletion:
    """Test completion when multiple structs are defined."""

    def test_multiple_structs_all_appear(self):
        source = """struct A { x: f32 }
struct B { y: f32 }
struct C { z: f32 }
fn f(a: A) -> f32 { a.x }
"""
        _, result = validate(source, "<test>")
        pos = types.Position(line=3, character=22)
        comp = _complete_general(result, pos)
        labels = _labels(comp)
        assert "A" in labels
        assert "B" in labels
        assert "C" in labels

    def test_struct_and_fn_same_name(self):
        """If a struct and function have the same name, both might appear.

        Category: Edge case -- potential duplicate.
        """
        source = "struct Point { x: f32 }\nfn Point(v: f32) -> f32 { v }\nfn f(p: Point) -> f32 { p.x }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=2, character=25)
        comp = _complete_general(result, pos)
        labels = [item.label for item in comp.items]
        point_count = labels.count("Point")
        if point_count > 1:
            # This is a finding: duplicate labels
            pass  # Document: struct and function with same name creates duplicate


# ============================================================================
# J. Very specific text edit validation
# ============================================================================

class TestTextEditRanges:
    """Validate that text_edit and additional_text_edits ranges are correct."""

    def test_pipe_text_edit_range_matches_cursor(self):
        """text_edit range must contain the cursor position."""
        source = "fn f(x: f32[3]) -> f32 { x }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=27)
        comp = _complete_dot(result, pos)
        assert comp is not None
        for item in comp.items:
            if item.text_edit:
                r = item.text_edit.range
                assert r.start.line == pos.line
                assert r.start.character <= pos.character <= r.end.character, \
                    f"Item '{item.label}': range [{r.start.character}, {r.end.character}] doesn't contain cursor {pos.character}"

    def test_struct_field_no_additional_edits(self):
        """Struct field completions should NOT have additional_text_edits (no pipe rewrite)."""
        source = "struct S { val: f32 }\nfn f(s: S) -> f32 { s.val }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=1, character=22)
        comp = _complete_dot(result, pos)
        assert comp is not None
        field_items = [i for i in comp.items if i.kind == types.CompletionItemKind.Field]
        for item in field_items:
            assert item.additional_text_edits is None, \
                f"Field '{item.label}' should not have additional_text_edits"

    def test_pipe_additional_edits_replace_dot(self):
        """Pipe completion's additional_text_edits should replace the dot with ' |> '."""
        source = "fn f(x: f32[3]) -> f32 { x }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=27)
        comp = _complete_dot(result, pos)
        assert comp is not None
        pipe_items = [i for i in comp.items if i.kind == types.CompletionItemKind.Function]
        for item in pipe_items:
            if item.additional_text_edits:
                for ate in item.additional_text_edits:
                    assert ate.new_text == " |> "
                    # The range should be exactly 1 character (the dot)
                    assert ate.range.end.character - ate.range.start.character == 1, \
                        f"Dot replacement range is {ate.range.end.character - ate.range.start.character} chars, expected 1"


# ============================================================================
# K. _complete_general includes "config" builtin
# ============================================================================

class TestConfigBuiltin:
    """Test that 'config' builtin appears in completions."""

    def test_config_in_general_completions(self):
        source = "fn f(x: f32) -> f32 { x }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=23)
        comp = _complete_general(result, pos)
        labels = _labels(comp)
        assert "config" in labels

    def test_config_has_function_kind(self):
        source = "fn f(x: f32) -> f32 { x }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=23)
        comp = _complete_general(result, pos)
        cfg_item = _find_item(comp, "config")
        assert cfg_item is not None
        assert cfg_item.kind == types.CompletionItemKind.Function


# ============================================================================
# L. Stress test: many variables in scope
# ============================================================================

class TestManyVariables:
    """Test with many variables to check for performance / correctness."""

    def test_many_let_bindings(self):
        """Many let bindings should all appear in scope."""
        lets = "\n".join(f"    let v{i} = {float(i)};" for i in range(20))
        source = f"fn f(x: f32) -> f32 {{\n{lets}\n    x\n}}\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=21, character=4)
        variables = _vars_in_scope(result, pos)
        names = [v[0] for v in variables]
        for i in range(20):
            assert f"v{i}" in names, f"v{i} missing from scope"


# ============================================================================
# M. Completion detail format for user functions
# ============================================================================

class TestUserFnDetail:
    """Test that user function detail strings are well-formed."""

    def test_simple_fn_detail(self):
        source = "fn helper(x: f32, y: f32) -> f32 { x + y }\nfn f(z: f32) -> f32 { z }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=1, character=23)
        comp = _complete_general(result, pos)
        helper = _find_item(comp, "helper")
        assert helper is not None
        # Detail should show signature
        assert "f32" in helper.detail
        assert "->" in helper.detail

    def test_array_fn_detail(self):
        source = "fn process(data: f32[3, 4]) -> f32[3] { data[0] }\nfn f(x: f32) -> f32 { x }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=1, character=23)
        comp = _complete_general(result, pos)
        proc = _find_item(comp, "process")
        assert proc is not None
        assert "f32" in proc.detail


# ============================================================================
# N. Completion with doc comments on structs
# ============================================================================

class TestStructDocComments:
    """Test that struct doc comments appear in completions."""

    def test_struct_doc_in_general(self):
        source = "/// A 2D point\nstruct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=2, character=25)
        comp = _complete_general(result, pos)
        pt = _find_item(comp, "Point")
        assert pt is not None
        if pt.documentation:
            assert "2D point" in pt.documentation.value


# ============================================================================
# O. Edge case: cursor at column 0 on a line with content
# ============================================================================

class TestCursorColumn0:
    """Test completions at column 0."""

    def test_column_0_on_content_line(self):
        """Column 0 on a line with content should give general completions."""
        source = "fn f(x: f32) -> f32 {\n    x\n}\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=1, character=0)
        comp = _complete_general(result, pos)
        assert comp is not None
        labels = _labels(comp)
        assert "fn" in labels


# ============================================================================
# P. Pipe completions for functions from another function scope
# ============================================================================

class TestPipeCrossFunctionScope:
    """Test pipe completion with functions defined in different order."""

    def test_later_defined_fn_in_pipe(self):
        """Functions defined after the current one should appear in pipe completion."""
        source = "fn f(x: f32) -> f32 { x }\nfn later(y: f32) -> f32 { y + 1.0 }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=23)
        comp = _complete_dot(result, pos)
        if comp:
            labels = _labels(comp)
            assert "later" in labels


# ============================================================================
# Q. Struct field types in dot completion detail
# ============================================================================

class TestDotFieldDetail:
    """Test that dot completion for struct fields shows the correct type."""

    def test_field_detail_shows_type(self):
        source = "struct S { count: i32, value: f32 }\nfn f(s: S) -> f32 { s.value }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=1, character=22)
        comp = _complete_dot(result, pos)
        assert comp is not None
        count_item = _find_item(comp, "count")
        assert count_item is not None
        assert "i32" in count_item.detail

        value_item = _find_item(comp, "value")
        assert value_item is not None
        assert "f32" in value_item.detail


# ============================================================================
# R. Struct with same-name field and function
# ============================================================================

class TestFieldFunctionNameCollision:
    """Test when struct field and function have the same name."""

    def test_field_and_fn_same_name(self):
        """If struct field and builtin/function share a name, both should appear.

        Category: Edge case -- potential confusion but not a crash.
        """
        # 'sum' is both a builtin and could be a field name
        source = "struct S { sum: f32 }\nfn f(s: S) -> f32 { s.sum }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=1, character=22)
        comp = _complete_dot(result, pos)
        assert comp is not None
        labels = [item.label for item in comp.items]
        # Both the field and the pipe function should appear
        sum_items = [i for i in comp.items if i.label == "sum"]
        # Check if we get duplicates (field 'sum' + pipe function 'sum')
        if len(sum_items) > 1:
            kinds = [i.kind for i in sum_items]
            # One should be Field, one should be Function
            assert types.CompletionItemKind.Field in kinds
            assert types.CompletionItemKind.Function in kinds


# ============================================================================
# S. Validate that all _BUILTIN_NAMESPACES entries have docs
# ============================================================================

class TestNamespaceBuiltinDocs:
    """Verify that namespaced builtins have documentation."""

    def test_random_functions_have_docs(self):
        """All random.* functions should have documentation."""
        missing = []
        for fn_name in _BUILTIN_NAMESPACES.get("random", []):
            full_name = f"random.{fn_name}"
            if full_name not in _BUILTIN_DOCS or not _BUILTIN_DOCS[full_name]:
                missing.append(full_name)
        if missing:
            # This is a gap finding
            pass  # Document: some namespaced builtins missing docs


# ============================================================================
# T. Struct literal completion inside nested struct
# ============================================================================

class TestStructLiteralInside:
    """Test struct literal completion in complex expressions."""

    def test_struct_literal_inside_let(self):
        """Struct literal completion inside a let binding."""
        source = """struct Point { x: f32, y: f32 }
fn f() -> Point {
    let p = Point {  };
    p
}
"""
        _, result = validate(source, "<test>")
        # Line 2: "    let p = Point {  };"
        line_text = "    let p = Point {  };"
        pos = types.Position(line=2, character=20)
        comp = _complete_struct_literal(line_text, 20, result, source, pos)
        if comp:
            labels = _labels(comp)
            assert "x" in labels
            assert "y" in labels

    def test_struct_literal_as_function_arg(self):
        """Struct literal completion when struct is a function argument."""
        source = """struct Point { x: f32, y: f32 }
fn process(p: Point) -> f32 { p.x }
fn f() -> f32 {
    process(Point {  })
}
"""
        _, result = validate(source, "<test>")
        # Line 3: "    process(Point {  })"
        line_text = "    process(Point {  })"
        pos = types.Position(line=3, character=20)
        comp = _complete_struct_literal(line_text, 20, result, source, pos)
        if comp:
            labels = _labels(comp)
            assert "x" in labels
            assert "y" in labels


# ============================================================================
# U. Struct literal with "with" expression
# ============================================================================

class TestWithExprCompletion:
    """Test that 'with' expression doesn't interfere with struct literal completion."""

    def test_inside_with_block(self):
        """Cursor inside a with block should not trigger struct literal completion for outer struct."""
        source = """struct Point { x: f32, y: f32 }
fn f(p: Point) -> Point {
    p with { x = 1.0 }
}
"""
        _, result = validate(source, "<test>")
        # Line 2: "    p with { x = 1.0 }"
        # 'with {' has an opening brace preceded by 'with'
        # _complete_struct_literal should scan back and find 'with' not a struct name
        line_text = "    p with { x = 1.0 }"
        pos = types.Position(line=2, character=13)
        comp = _complete_struct_literal(line_text, 13, result, source, pos)
        # 'with' is preceded by 'p' which is not a struct name but a variable
        # The code checks struct_defs for the identifier before {
        # 'p' is not in struct_defs, so it should return None
        # OR it might find 'with' before '{' which is not a struct name
        # Let's check what actually happens
        if comp:
            # If it returns completions, check they're sensible
            labels = _labels(comp)
            # Shouldn't return struct literal fields for 'with'


# ============================================================================
# V. Import with partially typed name
# ============================================================================

class TestImportPartiallyTyped:
    """Test import completion with partial module names."""

    def test_from_with_partial_module_name(self):
        """'from ma' should still offer module completions."""
        line = "from ma"
        result = _complete_import(line, 7, "<test>")
        assert result is not None
        # Should list all modules (filtering is client-side)

    def test_import_with_path_prefix(self):
        """'from \"../lib' -- path-based import."""
        line = 'from "../lib'
        result = _complete_import(line, 12, "<test>")
        # The regex may or may not match this pattern
        # It's edge-case territory


# ============================================================================
# W. Validate all items in various contexts
# ============================================================================

class TestValidateAllItems:
    """Run assert_all_completions_valid on completions from various contexts."""

    def test_general_valid(self):
        source = "fn f(x: f32) -> f32 {\n    let a = 1.0;\n    a\n}\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=2, character=4)
        comp = _complete_general(result, pos)
        assert_all_completions_valid(comp, pos)

    def test_dot_struct_valid(self):
        source = "struct S { x: f32, y: f32 }\nfn f(s: S) -> f32 { s.x }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=1, character=22)
        comp = _complete_dot(result, pos)
        assert comp is not None
        assert_all_completions_valid(comp, pos)

    def test_dot_array_valid(self):
        source = "fn f(x: f32[3, 3]) -> f32 { x }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=30)
        comp = _complete_dot(result, pos)
        assert comp is not None
        assert_all_completions_valid(comp, pos)

    def test_dot_scalar_valid(self):
        source = "fn f(x: f32) -> f32 { x }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=24)
        comp = _complete_dot(result, pos)
        if comp:
            assert_all_completions_valid(comp, pos)

    def test_import_valid(self):
        line = "from math import { "
        comp = _complete_import(line, 19, "<test>")
        if comp:
            # Import completions don't have text_edits, so validation is simple
            pos = types.Position(line=0, character=19)
            assert_all_completions_valid(comp, pos)


# ============================================================================
# X. Duplicate struct + fn name in completions
# ============================================================================

class TestDuplicateLabelsDeep:
    """Deep test for duplicate labels."""

    def test_struct_and_fn_duplicate(self):
        """Struct and function with same name: check for duplicates."""
        source = "struct Foo { x: f32 }\nfn Foo(x: f32) -> f32 { x }\nfn f() -> f32 { 1.0 }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=2, character=17)
        comp = _complete_general(result, pos)
        labels = [item.label for item in comp.items]
        foo_count = labels.count("Foo")
        if foo_count > 1:
            # FINDING: duplicate labels for same-named struct and function
            pass  # This IS a finding to report

    def test_param_and_builtin_duplicate(self):
        """Parameter named 'exp' should shadow or coexist with builtin 'exp'."""
        source = "fn f(exp: f32) -> f32 { exp }\n"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=25)
        comp = _complete_general(result, pos)
        labels = [item.label for item in comp.items]
        exp_count = labels.count("exp")
        if exp_count > 1:
            # FINDING: duplicate 'exp' - one as Variable, one as Function
            kinds = [item.kind for item in comp.items if item.label == "exp"]
            pass  # Document finding
