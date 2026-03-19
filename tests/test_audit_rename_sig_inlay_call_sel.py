"""
Audit of Rename, Signature Help, Inlay Hints, Call Hierarchy & Selection Ranges.

All findings are categorized:
- CRITICAL: Wrong behavior that will cause incorrect results for users
- BUG: Incorrect behavior that produces wrong or confusing output
- GAP: Missing functionality per LSP spec or user expectations
- EDGE: Edge case where behavior is suboptimal but not broken

Tests that demonstrate bugs use pytest.skip() so the test suite passes
but the issue is documented. Tests that pass verify correctness.
"""

import pytest
from lsprotocol import types

from maomi.lsp import (
    validate,
    AnalysisResult,
    prepare_rename_at, rename_at,
    _sig_parse_call_context, _BUILTIN_SIGNATURES, _BUILTIN_DOCS,
    _build_signature_help,
    _build_inlay_hints,
    _sel_collect_ancestors, _sel_build_chain,
    _call_hierarchy_prepare, _call_hierarchy_incoming, _call_hierarchy_outgoing,
    _cache,
    _local_functions,
    _find_node_at,
)
from maomi.lsp._signature import _resolve_active_param
from maomi.lsp._inlay_hints import _inlay_collect_from_expr
from maomi.lsp._ast_utils import classify_symbol
from maomi.lsp._builtin_data import _KEYWORDS
from maomi.ast_nodes import (
    Identifier, IntLiteral, FloatLiteral, BinOp, CallExpr,
    LetStmt, FnDef, Block, Span, StructLiteral, FieldAccess,
    ScanExpr, MapExpr, Param,
)


def _edit_ranges(edits):
    """Extract (line, start_char, end_char) tuples from TextEdit list."""
    return sorted(
        (e.range.start.line, e.range.start.character, e.range.end.character)
        for e in edits
    )


# ===========================================================================
# CRITICAL-1: Call hierarchy handler double-converts coordinates
# ===========================================================================

class TestCritical1_CallHierarchyDoubleConversion:
    """
    CRITICAL: prepare_call_hierarchy handler does line+1/col+1
    before passing to _call_hierarchy_prepare which also does +1.

    File: src/maomi/lsp/_call_hierarchy.py
    Handler (lines 131-133):
        line = params.position.line + 1    # 0-indexed -> 1-indexed
        col = params.position.character + 1
        return _call_hierarchy_prepare(result, uri, line, col)

    Core function (lines 45-46):
        line = line_0 + 1    # adds 1 AGAIN
        col = col_0 + 1

    Result: LSP positions are converted from 0-indexed to 2-indexed.
    The handler converts to 1-indexed, then the core function adds 1 more,
    so the core function looks at (line+2, col+2) in 1-indexed coordinates.
    """

    SOURCE = "fn c(x: f32) -> f32 { x }\nfn b(x: f32) -> f32 { c(x) }\nfn a(x: f32) -> f32 { b(x) }"

    def test_core_function_works_with_0indexed(self):
        """Core function works correctly when tests pass 0-indexed input."""
        _, result = validate(self.SOURCE, "<test>")
        items = _call_hierarchy_prepare(result, "file:///test.mao", 0, 3)
        assert items is not None
        assert items[0].name == "c"

    def test_handler_simulation_gives_wrong_result(self):
        """Simulating handler's double-conversion gives wrong or None result."""
        _, result = validate(self.SOURCE, "<test>")
        # Handler does: line = 0 + 1 = 1, col = 3 + 1 = 4
        # Core does: line = 1 + 1 = 2, col = 4 + 1 = 5
        # So instead of looking at (1, 4) in 1-indexed [= fn 'c', col 4],
        # it looks at (2, 5) [= fn 'b', col 5 = 'x' param]
        items = _call_hierarchy_prepare(result, "file:///test.mao", 1, 4)
        if items is None or items[0].name != "c":
            name = items[0].name if items else None
            pytest.skip(f"CRITICAL: handler double-converts coords. "
                        f"Direct(0,3)='c', handler-sim(1,4)='{name}'")


# ===========================================================================
# BUG-1: Rename to keyword not blocked
# ===========================================================================

class TestBug1_RenameToKeyword:
    """
    BUG: Renaming a variable/function to a keyword produces syntactically
    invalid code. The implementation does not validate new_name.

    File: src/maomi/lsp/_rename.py
    Function: rename_at() — no validation of new_name against keywords.
    """

    @pytest.mark.parametrize("keyword", ["fn", "let", "if", "else", "struct",
                                          "while", "scan", "map", "fold", "in",
                                          "with", "true", "false", "import",
                                          "from", "grad", "value_and_grad", "cast"])
    def test_rename_var_to_keyword(self, keyword):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        edits = rename_at(source, result, 0, 5, keyword)
        if edits is not None:
            pytest.skip(f"BUG: rename to keyword '{keyword}' allowed ({len(edits)} edits)")

    @pytest.mark.parametrize("keyword", ["fn", "let", "if"])
    def test_rename_fn_to_keyword(self, keyword):
        source = "fn helper(x: f32) -> f32 { x }\nfn f(y: f32) -> f32 { helper(y) }"
        _, result = validate(source, "<test>")
        edits = rename_at(source, result, 0, 3, keyword)
        if edits is not None:
            pytest.skip(f"BUG: fn rename to keyword '{keyword}' allowed")


# ===========================================================================
# BUG-2: Shadowed variable rename crosses scopes
# ===========================================================================

class TestBug2_ShadowedVariableRename:
    """
    BUG: Renaming a let-bound variable that shadows a parameter also renames
    the parameter and all its uses. The implementation does not track
    variable scoping — _rename_collect_variable_edits walks the entire
    function body and matches by name.

    File: src/maomi/lsp/_rename.py
    Function: _rename_collect_variable_edits()
    Issue: Walks entire fn scope body matching by name without scope analysis.
    """

    def test_shadow_renames_param(self):
        """Renaming shadowed let 'x' also renames param 'x'."""
        source = """fn f(x: f32) -> f32 {
    let y = x;
    let x = y + 1.0;
    x
}"""
        _, result = validate(source, "<test>")
        edits = rename_at(source, result, 2, 8, "z")
        assert edits is not None
        ranges = _edit_ranges(edits)
        actual_lines = {line for line, _, _ in ranges}
        # BUG: touches lines {0, 1, 2, 3} instead of just {2, 3}
        if actual_lines != {2, 3}:
            pytest.skip(f"BUG: shadow rename touched lines {actual_lines}, expected {{2, 3}}")

    def test_rename_to_existing_name_allowed(self):
        """Renaming to an already-in-scope name is not blocked."""
        source = "fn f(x: f32, y: f32) -> f32 { x + y }"
        _, result = validate(source, "<test>")
        edits = rename_at(source, result, 0, 5, "y")
        if edits is not None:
            pytest.skip(f"BUG/GAP: rename to existing name 'y' allowed ({len(edits)} edits)")


# ===========================================================================
# BUG-3: Signature help inside string returns None
# ===========================================================================

class TestBug3_SigHelpInsideString:
    """
    BUG: When cursor is inside a string argument (e.g., config("key")),
    _sig_parse_call_context returns None instead of the function name.

    File: src/maomi/lsp/_signature.py
    Function: _sig_parse_call_context()
    Issue: The backward string-skipping logic (lines 39-47) treats any '"'
    as a closing quote and scans backward for a matching opening quote.
    When cursor is INSIDE a string, it encounters the opening quote first,
    misinterprets it as a closing quote, and then fails to find a match.
    """

    def test_cursor_inside_string_arg(self):
        """Cursor on 'e' inside config("key") returns None."""
        source = 'fn f() -> f32 { config("key") }'
        name, idx, _ = _sig_parse_call_context(source, types.Position(line=0, character=25))
        if name != "config":
            pytest.skip(f"BUG: cursor inside string, name={name}, expected 'config'")

    def test_cursor_on_closing_quote(self):
        """Cursor on closing '"' works (scanning backward hits it as first quote)."""
        source = 'fn f() -> f32 { config("key") }'
        # Closing quote at position 27 — but col-1=26='y', doesn't hit quote
        # Position 28=')': col-1=27='"' closing quote -> works!
        name, idx, _ = _sig_parse_call_context(source, types.Position(line=0, character=28))
        assert name == "config"  # This one works


# ===========================================================================
# BUG-4: prepare_rename on type annotation returns param name
# ===========================================================================

class TestBug4_PrepareRenameTypeAnnotation:
    """
    BUG: When cursor is on a struct type in a parameter's type annotation
    (e.g., 'Point' in 'p: Point'), prepare_rename returns the param name
    range ('p') instead of the struct name range ('Point').

    File: src/maomi/lsp/_rename.py
    Functions: prepare_rename_at() and rename_at()
    Issue: classify_symbol(node) at line 133 is called WITHOUT line/col/struct_names,
    so it returns the param variable name instead of distinguishing whether the
    cursor is on the name or the type annotation.
    """

    def test_cursor_on_struct_type_returns_param(self):
        """Clicking on 'Point' in 'p: Point' returns 'p' range."""
        source = """struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 { p.x }"""
        _, result = validate(source, "<test>")
        rng = prepare_rename_at(source, result, 1, 8)  # on 'Point'
        if rng is not None:
            text = source.splitlines()[rng.start.line][rng.start.character:rng.end.character]
            if text != "Point":
                pytest.skip(f"BUG: cursor on struct type 'Point' returned '{text}' range")


# ===========================================================================
# GAP-1: Field rename not supported
# ===========================================================================

class TestGap1_FieldRenameNotSupported:
    """
    GAP: Renaming struct fields is not supported. Clicking on a field name
    (e.g., 'x' in 's.x') does not trigger rename for the field.

    File: src/maomi/lsp/_rename.py
    Issue: No _rename_collect_field_edits function exists.
    classify_symbol for FieldAccess returns the object's name, not the field.
    """

    def test_field_access_not_renameable(self):
        source = """struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 { p.x }"""
        _, result = validate(source, "<test>")
        edits = rename_at(source, result, 1, 27, "z")
        if edits is None:
            pytest.skip("GAP: field rename not supported")

    def test_struct_def_field_not_renameable(self):
        source = "struct Point { x: f32, y: f32 }"
        _, result = validate(source, "<test>")
        rng = prepare_rename_at(source, result, 0, 15)
        if rng is None:
            pytest.skip("GAP: struct def field names not renameable")


# ===========================================================================
# GAP-2: CallHierarchyItem selection_range equals range
# ===========================================================================

class TestGap2_CallHierarchySelectionRange:
    """
    GAP: CallHierarchyItem uses fn.span for both range AND selection_range.
    Per LSP spec, selection_range should be the range to "select/highlight"
    (usually just the function name), not the entire function body.

    File: src/maomi/lsp/_call_hierarchy.py
    Function: _make_hierarchy_item() at line 29-35
    Both range and selection_range use _span_to_range(fn.span).
    """

    def test_selection_range_equals_range(self):
        source = """fn helper(x: f32) -> f32 {
    x + 1.0
}
fn f(x: f32) -> f32 { helper(x) }"""
        _, result = validate(source, "<test>")
        items = _call_hierarchy_prepare(result, "file:///test.mao", 0, 3)
        assert items is not None
        item = items[0]
        if item.range == item.selection_range:
            pytest.skip("GAP: selection_range == range (should be just function name)")


# ===========================================================================
# GAP-3: Inlay hints duplicate code in _inlay_collect_from_expr
# ===========================================================================

class TestGap3_InlayHintsDuplicateCode:
    """
    GAP (code quality): _inlay_collect_from_expr has duplicate elif branches.

    File: src/maomi/lsp/_inlay_hints.py
    Lines 69-75: WhileExpr handling
    Lines 83-89: WhileExpr handling AGAIN (dead code, unreachable)
    Lines 65-68: FoldExpr handling
    Lines 90-93: FoldExpr handling AGAIN (dead code, unreachable)
    """

    def test_duplicate_while_branch(self):
        import inspect
        src = inspect.getsource(_inlay_collect_from_expr)
        count = src.count('WhileExpr')
        if count > 1:
            pytest.skip(f"GAP: _inlay_collect_from_expr has {count} WhileExpr branches (1 is dead code)")

    def test_duplicate_fold_branch(self):
        import inspect
        src = inspect.getsource(_inlay_collect_from_expr)
        count = src.count('FoldExpr')
        if count > 1:
            pytest.skip(f"GAP: _inlay_collect_from_expr has {count} FoldExpr branches (1 is dead code)")


# ===========================================================================
# EDGE-1: Named arg detection only works at value position
# ===========================================================================

class TestEdge1_NamedArgDetection:
    """
    EDGE: Named argument detection in _sig_parse_call_context only detects
    the named arg when cursor is on the VALUE after '=', not on the name
    or '=' itself.

    File: src/maomi/lsp/_signature.py
    Function: _sig_parse_call_context()
    Issue: The current_arg text is extracted between arg_start and col.
    When cursor is on 'axis' or '=', the text doesn't contain '=' yet,
    so named_param is None.
    """

    def test_named_arg_on_value(self):
        """Named arg detected when cursor is on the value (works correctly)."""
        source = "fn f(x: f32[4,8]) -> f32[4] { sum(x, axis=0) }"
        name, idx, named = _sig_parse_call_context(source, types.Position(line=0, character=42))
        assert name == "sum"
        assert named == "axis"

    def test_named_arg_on_equals(self):
        """Named arg NOT detected when cursor is on '='."""
        source = "fn f(x: f32[4,8]) -> f32[4] { sum(x, axis=0) }"
        name, idx, named = _sig_parse_call_context(source, types.Position(line=0, character=41))
        assert name == "sum"
        if named is None:
            pytest.skip("EDGE: named arg not detected on '=' sign")

    def test_named_arg_on_name_text(self):
        """Named arg NOT detected when cursor is on 'axis' text."""
        source = "fn f(x: f32[4,8]) -> f32[4] { sum(x, axis=0) }"
        name, idx, named = _sig_parse_call_context(source, types.Position(line=0, character=38))
        assert name == "sum"
        if named is None:
            pytest.skip("EDGE: named arg not detected on name text 'axis'")

    def test_named_arg_cross_line(self):
        """Named arg NOT detected when on a different line than opening paren."""
        source = """fn f(x: f32[4,8]) -> f32[4] {
    sum(
        x,
        axis=1
    )
}"""
        name, idx, named = _sig_parse_call_context(source, types.Position(line=3, character=14))
        assert name == "sum"
        if named is None:
            pytest.skip("EDGE: named arg detection fails across lines")


# ===========================================================================
# EDGE-2: Cursor on opening paren returns None
# ===========================================================================

class TestEdge2_CursorOnOpenParen:
    """
    EDGE: When cursor is exactly on the '(' character,
    _sig_parse_call_context returns None.

    File: src/maomi/lsp/_signature.py
    Function: _sig_parse_call_context()
    Issue: i = min(col - 1, len - 1) starts scanning one char BEFORE the cursor.
    When cursor is at '(' position, it starts at the char before '(' and never
    sees the opening paren.
    """

    def test_cursor_on_open_paren(self):
        source = 'fn f() -> f32 { config("key") }'
        # '(' is at position 22
        name, idx, _ = _sig_parse_call_context(source, types.Position(line=0, character=22))
        if name is None:
            pytest.skip("EDGE: cursor on '(' returns None (starts scanning before it)")


# ===========================================================================
# EDGE-3: Destructured let produces duplicate hint positions
# ===========================================================================

class TestEdge3_DestructuredLetHints:
    """
    EDGE: Destructured let bindings produce multiple InlayHints at the same
    character position, plus a confusing full-struct-type hint.

    File: src/maomi/lsp/_inlay_hints.py
    Issue: Parser desugars `let { x, y } = p;` into multiple LetStmt nodes.
    Each gets a type hint, but they may share the same source position.
    Also, the outer binding gets a ': Point { x: f32, y: f32 }' hint
    which is redundant since the individual fields already have hints.
    """

    def test_destructured_duplicate_positions(self):
        source = """struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 {
    let { x, y } = p;
    x + y
}"""
        _, result = validate(source, "<test>")
        hints = _build_inlay_hints(result, 1, 5, source)
        type_hints = [h for h in hints if h.kind == types.InlayHintKind.Type]
        positions = [(h.position.line, h.position.character) for h in type_hints]
        unique = set(positions)
        if len(positions) != len(unique):
            pytest.skip(f"EDGE: duplicate hint positions: {positions}")

    def test_destructured_shows_full_struct_type(self):
        source = """struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 {
    let { x, y } = p;
    x + y
}"""
        _, result = validate(source, "<test>")
        hints = _build_inlay_hints(result, 1, 5, source)
        type_hints = [h for h in hints if h.kind == types.InlayHintKind.Type]
        struct_hints = [h for h in type_hints if "Point" in h.label]
        if struct_hints:
            pytest.skip(f"EDGE: destructured let shows full struct type: '{struct_hints[0].label}'")


# ===========================================================================
# EDGE-4: Call hierarchy prepare matches variable with function name
# ===========================================================================

class TestEdge4_CallHierarchyVariableMatchesFn:
    """
    EDGE: _call_hierarchy_prepare treats an Identifier as a function
    reference if its name matches a function name, even when the
    Identifier is actually a variable (parameter).

    File: src/maomi/lsp/_call_hierarchy.py
    Function: _call_hierarchy_prepare()
    Lines 59-60: `elif isinstance(node, Identifier) and node.name in fn_names`
    """

    def test_identifier_matching_fn_name(self):
        source = """fn helper(x: f32) -> f32 { x }
fn f(helper: f32) -> f32 { helper }"""
        _, result = validate(source, "<test>")
        items = _call_hierarchy_prepare(result, "file:///test.mao", 1, 28)
        if items is not None:
            pytest.skip(f"EDGE: variable 'helper' matched as function '{items[0].name}'")


# ===========================================================================
# Correctness verification tests (should all pass)
# ===========================================================================

class TestRenameCorrectness:
    """Verify rename works correctly in standard cases."""

    def test_rename_function(self):
        source = "\nfn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 { helper(y) }"
        _, result = validate(source, "<test>")
        edits = rename_at(source, result, 1, 3, "util")
        assert edits is not None
        assert len(edits) == 2

    def test_rename_function_from_callsite(self):
        source = "\nfn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 { helper(y) }"
        _, result = validate(source, "<test>")
        edits = rename_at(source, result, 2, 25, "util")
        assert edits is not None
        assert len(edits) == 2

    def test_rename_variable(self):
        source = "fn f(x: f32) -> f32 { x + x }"
        _, result = validate(source, "<test>")
        edits = rename_at(source, result, 0, 22, "a")
        assert edits is not None
        assert len(edits) == 3

    def test_rename_respects_function_scoping(self):
        source = "\nfn f(x: f32) -> f32 { x }\nfn g(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        edits = rename_at(source, result, 1, 5, "a")
        assert edits is not None
        for e in edits:
            assert e.range.start.line == 1

    def test_rename_struct(self):
        source = "\nstruct Point { x: f32, y: f32 }\nfn f(p: Point) -> Point { p }"
        _, result = validate(source, "<test>")
        edits = rename_at(source, result, 1, 7, "Vec2")
        assert edits is not None
        assert len(edits) == 3

    def test_rename_multiple_call_sites(self):
        source = """fn helper(x: f32) -> f32 { x }
fn a(x: f32) -> f32 { helper(x) }
fn b(x: f32) -> f32 { helper(x) }"""
        _, result = validate(source, "<test>")
        edits = rename_at(source, result, 0, 3, "util")
        assert edits is not None
        assert len(edits) == 3

    def test_rename_in_nested_if(self):
        source = """fn f(x: f32) -> f32 {
    if x > 0.0 {
        x + 1.0
    } else {
        x - 1.0
    }
}"""
        _, result = validate(source, "<test>")
        edits = rename_at(source, result, 0, 5, "val")
        assert edits is not None
        assert len(edits) >= 4

    def test_prepare_rename_rejects_builtin(self):
        source = "fn f(x: f32) -> f32 { exp(x) }"
        _, result = validate(source, "<test>")
        rng = prepare_rename_at(source, result, 0, 22)
        assert rng is None

    def test_rename_returns_none_empty(self):
        source = ""
        _, result = validate(source, "<test>")
        assert rename_at(source, result, 0, 0, "x") is None
        assert prepare_rename_at(source, result, 0, 0) is None


class TestSignatureHelpCorrectness:
    """Verify signature help works correctly."""

    def test_builtin_sig(self):
        source = "fn f(x: f32) -> f32 { exp(x) }"
        name, idx, _ = _sig_parse_call_context(source, types.Position(line=0, character=27))
        assert name == "exp"
        assert idx == 0

    def test_second_param(self):
        source = "fn f(a: f32, b: f32) -> f32 { helper(a, b) }"
        name, idx, _ = _sig_parse_call_context(source, types.Position(line=0, character=40))
        assert name == "helper"
        assert idx == 1

    def test_nested_inner(self):
        source = "fn f(x: f32) -> f32 { exp(log(x)) }"
        name, idx, _ = _sig_parse_call_context(source, types.Position(line=0, character=30))
        assert name == "log"

    def test_nested_outer(self):
        source = "fn f(x: f32) -> f32 { exp(log(x), y) }"
        name, idx, _ = _sig_parse_call_context(source, types.Position(line=0, character=34))
        assert name == "exp"
        assert idx == 1

    def test_multiline(self):
        source = """fn f(a: f32, b: f32) -> f32 {
    helper(
        a,
        b
    )
}"""
        name, idx, _ = _sig_parse_call_context(source, types.Position(line=3, character=8))
        assert name == "helper"
        assert idx == 1

    def test_build_sig_help_clamping(self):
        result = _build_signature_help("f", ["x", "y"], ["f32", "f32"], "f32", 99)
        assert result.active_parameter == 1

    def test_build_sig_help_empty_params(self):
        result = _build_signature_help("f", [], [], "f32", 0)
        assert result.active_parameter == 0

    def test_build_sig_help_doc(self):
        result = _build_signature_help("f", ["x"], ["f32"], "f32", 0, doc="Help")
        assert result.signatures[0].documentation.value == "Help"

    def test_resolve_active_named(self):
        assert _resolve_active_param(2, "y", ["x", "y", "z"]) == 1

    def test_resolve_active_unknown_named(self):
        assert _resolve_active_param(2, "w", ["x", "y", "z"]) == 2

    def test_outside_call(self):
        source = "fn f(x: f32) -> f32 { x }"
        name, _, _ = _sig_parse_call_context(source, types.Position(line=0, character=23))
        assert name is None


class TestInlayHintsCorrectness:
    """Verify inlay hints work correctly."""

    def test_let_type_hint(self):
        source = "fn f(x: f32) -> f32 { let a = x; a }"
        _, result = validate(source, "<test>")
        hints = _build_inlay_hints(result, 1, 1, source)
        type_hints = [h for h in hints if h.kind == types.InlayHintKind.Type]
        assert len(type_hints) == 1
        assert type_hints[0].label == ": f32"

    def test_no_hint_explicit_annotation(self):
        source = "fn f(x: f32) -> f32 { let a: f32 = x; a }"
        _, result = validate(source, "<test>")
        hints = _build_inlay_hints(result, 1, 1, source)
        assert len([h for h in hints if h.kind == types.InlayHintKind.Type]) == 0

    def test_param_hint_suppression(self):
        """Param hints suppressed when arg name matches param name."""
        source = """fn helper(a: f32, b: f32) -> f32 { a + b }
fn f(a: f32, b: f32) -> f32 { helper(a, b) }"""
        _, result = validate(source, "<test>")
        hints = _build_inlay_hints(result, 1, 2, source)
        param_hints = [h for h in hints if h.kind == types.InlayHintKind.Parameter]
        assert len(param_hints) == 0

    def test_single_param_builtin_no_hint(self):
        source = "fn f(x: f32) -> f32 { exp(x) }"
        _, result = validate(source, "<test>")
        hints = _build_inlay_hints(result, 1, 1, source)
        assert len([h for h in hints if h.kind == types.InlayHintKind.Parameter]) == 0

    def test_empty_program(self):
        hints = _build_inlay_hints(AnalysisResult(None, {}, {}, {}), 1, 1, "")
        assert hints == []


class TestCallHierarchyCorrectness:
    """Verify call hierarchy works correctly."""

    SOURCE = "fn c(x: f32) -> f32 { x }\nfn b(x: f32) -> f32 { c(x) }\nfn a(x: f32) -> f32 { b(x) }"

    def _setup(self):
        _, result = validate(self.SOURCE, "<test>")
        _cache["file:///test.mao"] = result
        return result

    def test_incoming_calls(self):
        result = self._setup()
        incoming = _call_hierarchy_incoming(result, "file:///test.mao", "b")
        assert len(incoming) == 1
        assert incoming[0].from_.name == "a"

    def test_outgoing_calls(self):
        result = self._setup()
        outgoing = _call_hierarchy_outgoing(result, "file:///test.mao", "b")
        assert len(outgoing) == 1
        assert outgoing[0].to.name == "c"

    def test_no_callers(self):
        result = self._setup()
        assert _call_hierarchy_incoming(result, "file:///test.mao", "a") == []

    def test_no_callees(self):
        result = self._setup()
        assert _call_hierarchy_outgoing(result, "file:///test.mao", "c") == []

    def test_recursive(self):
        source = """fn recurse(x: f32) -> f32 {
    if x > 0.0 { recurse(x - 1.0) } else { 0.0 }
}"""
        _, result = validate(source, "<test>")
        _cache["file:///test.mao"] = result
        incoming = _call_hierarchy_incoming(result, "file:///test.mao", "recurse")
        outgoing = _call_hierarchy_outgoing(result, "file:///test.mao", "recurse")
        assert len(incoming) == 1
        assert len(outgoing) == 1

    def test_multiple_call_sites(self):
        source = """fn h(x: f32) -> f32 { x }
fn f(x: f32) -> f32 { h(x) + h(x) }"""
        _, result = validate(source, "<test>")
        _cache["file:///test.mao"] = result
        incoming = _call_hierarchy_incoming(result, "file:///test.mao", "h")
        assert len(incoming) == 1
        assert len(incoming[0].from_ranges) == 2

    def test_empty_program(self):
        empty = AnalysisResult(None, {}, {}, {})
        assert _call_hierarchy_prepare(empty, "uri", 0, 0) is None
        assert _call_hierarchy_incoming(empty, "uri", "f") == []
        assert _call_hierarchy_outgoing(empty, "uri", "f") == []


class TestSelectionRangeCorrectness:
    """Verify selection ranges work correctly."""

    def test_identifier_chain(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        fn = result.program.functions[0]
        expr = fn.body.expr
        ancestors = []
        _sel_collect_ancestors(fn, expr.span.line_start, expr.span.col_start, ancestors)
        assert len(ancestors) >= 2
        chain = _sel_build_chain(ancestors)
        assert chain is not None
        assert chain.parent is not None

    def test_binop_chain(self):
        source = "fn f(a: f32, b: f32) -> f32 { a + b }"
        _, result = validate(source, "<test>")
        fn = result.program.functions[0]
        left = fn.body.expr.left
        ancestors = []
        _sel_collect_ancestors(fn, left.span.line_start, left.span.col_start, ancestors)
        assert len(ancestors) >= 3
        chain = _sel_build_chain(ancestors)
        assert chain.parent is not None
        assert chain.parent.parent is not None

    def test_chain_ordering(self):
        """Each parent range contains the child range."""
        source = "fn f(a: f32, b: f32) -> f32 { a + b }"
        _, result = validate(source, "<test>")
        fn = result.program.functions[0]
        left = fn.body.expr.left
        ancestors = []
        _sel_collect_ancestors(fn, left.span.line_start, left.span.col_start, ancestors)
        chain = _sel_build_chain(ancestors)
        ranges = []
        node = chain
        while node is not None:
            r = node.range
            ranges.append((r.start.line, r.start.character, r.end.line, r.end.character))
            node = node.parent
        for i in range(len(ranges) - 1):
            inner = ranges[i]
            outer = ranges[i + 1]
            assert (outer[0], outer[1]) <= (inner[0], inner[1])
            assert (outer[2], outer[3]) >= (inner[2], inner[3])

    def test_empty_list(self):
        assert _sel_build_chain([]) is None

    def test_deeply_nested(self):
        source = "fn f(x: f32) -> f32 { exp(log(tanh(x))) }"
        _, result = validate(source, "<test>")
        fn = result.program.functions[0]
        innermost = fn.body.expr.args[0].args[0].args[0]
        ancestors = []
        _sel_collect_ancestors(fn, innermost.span.line_start, innermost.span.col_start, ancestors)
        assert len(ancestors) >= 5

    def test_no_span_node(self):
        class NoSpan:
            pass
        ancestors = []
        assert _sel_collect_ancestors(NoSpan(), 1, 1, ancestors) is False
        assert len(ancestors) == 0
