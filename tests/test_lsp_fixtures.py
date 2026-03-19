"""Data-driven parameterized LSP correctness tests for the Maomi language server.

Each test case specifies: source code, cursor position, and expected outcome.
Uses @pytest.mark.parametrize for each LSP feature category.
"""

import pytest
from lsprotocol import types

from maomi.lsp import (
    validate,
    _find_node_at,
    _get_hover_text,
    _goto_find_definition,
    _goto_type_definition,
    _complete_general,
    _complete_dot,
    _refs_classify_node,
    _refs_collect_all,
    _span_contains,
    prepare_rename_at,
    rename_at,
    _sig_parse_call_context,
    _build_inlay_hints,
    _build_document_symbols,
    _build_document_highlights,
    _format_document,
    _build_folding_ranges,
    _build_code_lenses,
    _local_functions,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _validate(source):
    """Validate source and return (diags, result)."""
    return validate(source, "<test>")


def _hover_at(source, line_0, col_0):
    """Get hover text at 0-indexed position."""
    _, result = _validate(source)
    if not result.program:
        return None
    line_1, col_1 = line_0 + 1, col_0 + 1
    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line_1, col_1)
        if node is not None:
            return _get_hover_text(node, fn, result)
    return None


def _completions_at(source, line_0, col_0):
    """Get completion labels at 0-indexed position."""
    _, result = _validate(source)
    pos = types.Position(line=line_0, character=col_0)
    comp = _complete_general(result, pos)
    return {item.label for item in comp.items} if comp else set()


def _goto_def_at(source, line_0, col_0):
    """Get definition location (line_0) at 0-indexed position, or None."""
    _, result = _validate(source)
    if not result.program:
        return None
    line_1, col_1 = line_0 + 1, col_0 + 1
    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line_1, col_1)
        if node is not None:
            found = _goto_find_definition(node, fn, result)
            if found is not None:
                span, _ = found
                return span.line_start - 1  # return 0-indexed
    return None


def _refs_at(source, line_1, col_1, include_decl):
    """Get reference count at 1-indexed position."""
    _, result = _validate(source)
    if not result.program:
        return 0

    # Check struct defs
    for sd in result.program.struct_defs:
        if _span_contains(sd.span, line_1, col_1):
            spans = _refs_collect_all(result, sd.name, "struct", include_decl)
            return len(spans)

    # Check functions
    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line_1, col_1)
        if node is not None:
            name, kind = _refs_classify_node(node, line_1, col_1)
            if name:
                fn_scope = fn if kind == "variable" else None
                spans = _refs_collect_all(
                    result, name, kind, include_decl, fn_scope=fn_scope
                )
                return len(spans)
    return 0


def _rename_count(source, line_0, col_0, new_name):
    """Get count of rename edits, or None if rename not possible."""
    _, result = _validate(source)
    edits = rename_at(source, result, line_0, col_0, new_name)
    if edits is None:
        return None
    return len(edits)


def _sig_help_at(source, line_0, col_0):
    """Get (fn_name, param_index) from signature help context."""
    pos = types.Position(line=line_0, character=col_0)
    name, idx, _ = _sig_parse_call_context(source, pos)
    return name, idx


def _inlay_hint_count(source, start_line_1, end_line_1):
    """Get number of inlay hints in line range (1-indexed)."""
    _, result = _validate(source)
    if not result.program:
        return 0
    hints = _build_inlay_hints(result, start_line_1, end_line_1, source)
    return len(hints)


def _doc_symbol_names(source):
    """Get set of top-level document symbol names."""
    _, result = _validate(source)
    symbols = _build_document_symbols(result)
    if symbols is None:
        return set()
    return {s.name for s in symbols}


def _highlight_count(source, line_1, col_1):
    """Get number of highlights at 1-indexed position, or None."""
    _, result = _validate(source)
    highlights = _build_document_highlights(result, line_1, col_1)
    if highlights is None:
        return None
    return len(highlights)


def _format_has_edits(source):
    """Return whether formatting produces any edits."""
    edits = _format_document(source)
    return len(edits) > 0


def _folding_range_count(source):
    """Get number of folding ranges."""
    _, result = _validate(source)
    ranges = _build_folding_ranges(result)
    return len(ranges)


# ---------------------------------------------------------------------------
# Hover tests
# ---------------------------------------------------------------------------

HOVER_CASES = [
    # (id, source, line_0, col_0, expected_substring_or_None)
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0, 22, "f32",
        id="var-in-body",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0, 5, "f32",
        id="param-hover",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0, 3, "fn f",
        id="function-name",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { exp(x) }",
        0, 22, "exp",
        id="builtin-call",
    ),
    pytest.param(
        "struct P { x: f32 }\nfn f(p: P) -> f32 { p.x }",
        1, 22, "f32",
        id="field-access-type",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { let a = x; a }",
        0, 23, "let a",
        id="let-binding",
    ),
    pytest.param(
        "/// A doc\nfn f(x: f32) -> f32 { x }",
        1, 3, "A doc",
        id="doc-comment-in-hover",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x + 1.0 }",
        0, 22, "f32",
        id="var-in-binop",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0, 2, "fn f",
        id="on-fn-keyword-area",
    ),
    pytest.param(
        "fn f( -> f32 { x }",
        0, 15, None,
        id="parse-error-no-hover",
    ),
    pytest.param(
        "fn f(x: f32, y: f32) -> f32 { x + y }",
        0, 31, "f32",
        id="first-param-in-binop",
    ),
    pytest.param(
        "fn f(x: f32, y: f32) -> f32 { x + y }",
        0, 35, "f32",
        id="second-param-in-binop",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { tanh(x) }",
        0, 22, "tanh",
        id="tanh-builtin",
    ),
    pytest.param(
        "struct P { x: f32, y: f32 }\nfn f() -> P { P { x: 1.0, y: 2.0 } }",
        1, 14, "P",
        id="struct-literal-hover",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { let a = x; let b = a; b }",
        0, 38, "let b",
        id="second-let-binding",
    ),
]


@pytest.mark.parametrize("source, line_0, col_0, expected", HOVER_CASES)
def test_hover(source, line_0, col_0, expected):
    result = _hover_at(source, line_0, col_0)
    if expected is None:
        assert result is None, f"Expected no hover, got: {result}"
    else:
        assert result is not None, f"Expected hover containing '{expected}', got None"
        assert expected in result, f"Expected '{expected}' in hover text, got: {result}"


# ---------------------------------------------------------------------------
# Completion tests
# ---------------------------------------------------------------------------

COMPLETION_CASES = [
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0, 23, {"exp", "let", "x"},
        id="basic-completions",
    ),
    pytest.param(
        "fn a(x: f32) -> f32 { x }\nfn b(y: f32) -> f32 { y }",
        1, 23, {"a"},
        id="user-fn-visible",
    ),
    pytest.param(
        "struct P { x: f32 }\nfn f(p: P) -> f32 { p.x }",
        1, 23, {"P"},
        id="struct-name-completion",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0, 23, {"if", "scan", "map"},
        id="keyword-completions",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0, 23, {"fn"},
        id="fn-keyword-completion",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0, 23, {"f32", "i32", "bool"},
        id="type-name-completions",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0, 23, {"tanh", "mean", "sum"},
        id="builtin-completions",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0, 23, {"random"},
        id="namespace-completion",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 {\n    let a = x;\n    a\n}",
        2, 4, {"a"},
        id="let-binding-in-scope",
    ),
    pytest.param(
        "fn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 { y }",
        1, 26, {"helper"},
        id="cross-fn-completion",
    ),
]


@pytest.mark.parametrize("source, line_0, col_0, expected_subset", COMPLETION_CASES)
def test_completions(source, line_0, col_0, expected_subset):
    labels = _completions_at(source, line_0, col_0)
    for label in expected_subset:
        assert label in labels, f"Expected '{label}' in completions, got: {labels}"


# ---------------------------------------------------------------------------
# Go-to-definition tests
# ---------------------------------------------------------------------------

GOTO_DEF_CASES = [
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0, 22, 0,
        id="var-to-param",
    ),
    pytest.param(
        "fn a(x: f32) -> f32 { x }\nfn b(y: f32) -> f32 { a(y) }",
        1, 22, 0,
        id="call-to-fn-def",
    ),
    pytest.param(
        "struct P { x: f32 }\nfn f() -> P { P { x: 1.0 } }",
        1, 14, 0,
        id="struct-lit-to-struct-def",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { exp(x) }",
        0, 22, None,
        id="builtin-no-def",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { let a = x; a }",
        0, 34, 0,
        id="var-to-let-binding",
    ),
    pytest.param(
        "fn f() -> f32 { 1.0 }",
        0, 16, None,
        id="literal-no-def",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 {\n    let a = x;\n    a\n}",
        2, 4, 1,
        id="var-to-let-multiline",
    ),
    pytest.param(
        "fn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 { helper(y) }",
        1, 25, 0,
        id="call-to-fn-first-line",
    ),
    pytest.param(
        "fn f(x: f32, y: f32) -> f32 { x + y }",
        0, 35, 0,
        id="second-param-to-def",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { let a = x; let b = a; b }",
        0, 44, 0,
        id="chained-let-to-binding",
    ),
]


@pytest.mark.parametrize("source, line_0, col_0, expected_def_line_0", GOTO_DEF_CASES)
def test_goto_definition(source, line_0, col_0, expected_def_line_0):
    result = _goto_def_at(source, line_0, col_0)
    if expected_def_line_0 is None:
        assert result is None, f"Expected no definition, got line {result}"
    else:
        assert result is not None, "Expected a definition, got None"
        assert result == expected_def_line_0, (
            f"Expected definition on line {expected_def_line_0}, got line {result}"
        )


# ---------------------------------------------------------------------------
# References tests
# ---------------------------------------------------------------------------

REFERENCES_CASES = [
    pytest.param(
        "fn f(x: f32) -> f32 { x + x }",
        1, 23, False, 2,
        id="two-uses-of-x",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x + x }",
        1, 23, True, 3,
        id="two-uses-plus-decl",
    ),
    pytest.param(
        "fn a(x: f32) -> f32 { x }\nfn b(y: f32) -> f32 { a(y) + a(y) }",
        1, 4, True, 3,
        id="fn-decl-plus-two-calls",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { let a = x; a + a }",
        1, 35, False, 2,
        id="let-var-two-uses",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { let a = x; a + a }",
        1, 35, True, 3,
        id="let-var-uses-plus-decl",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }\nfn g(x: f32) -> f32 { x }",
        1, 23, False, 1,
        id="var-scoped-to-fn-f",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }\nfn g(x: f32) -> f32 { x }",
        2, 23, False, 1,
        id="var-scoped-to-fn-g",
    ),
    pytest.param(
        "struct P { x: f32 }\nfn f(p: P) -> f32 { p.x }",
        1, 9, True, 2,
        id="struct-refs-with-decl",
    ),
]


@pytest.mark.parametrize(
    "source, line_1, col_1, include_decl, expected_count", REFERENCES_CASES
)
def test_references(source, line_1, col_1, include_decl, expected_count):
    count = _refs_at(source, line_1, col_1, include_decl)
    assert count == expected_count, (
        f"Expected {expected_count} references, got {count}"
    )


# ---------------------------------------------------------------------------
# Rename tests
# ---------------------------------------------------------------------------

RENAME_CASES = [
    pytest.param(
        "fn f(x: f32) -> f32 { x + x }",
        0, 22, "a", 3,
        id="rename-param-three-edits",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { exp(x) }",
        0, 22, "nope", None,
        id="builtin-cannot-rename",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { 1.0 }",
        0, 22, "nope", None,
        id="literal-cannot-rename",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { let y = x; y + y }",
        0, 27, "z", 3,
        id="let-binding-rename",
    ),
    pytest.param(
        "fn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 { helper(y) }",
        0, 3, "h", 2,
        id="rename-function",
    ),
    pytest.param(
        "struct P { x: f32 }\nfn f(p: P) -> P { p }",
        0, 7, "Q", 3,
        id="rename-struct",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0, 5, "y", 2,
        id="rename-single-use-param",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { let a = x; let b = a; b }",
        0, 43, "c", 2,
        id="rename-second-let",
    ),
]


@pytest.mark.parametrize(
    "source, line_0, col_0, new_name, expected_count", RENAME_CASES
)
def test_rename(source, line_0, col_0, new_name, expected_count):
    count = _rename_count(source, line_0, col_0, new_name)
    if expected_count is None:
        assert count is None, f"Expected rename to be impossible, got {count} edits"
    else:
        assert count is not None, "Expected rename edits, got None (rename not possible)"
        assert count == expected_count, (
            f"Expected {expected_count} rename edits, got {count}"
        )


# ---------------------------------------------------------------------------
# Signature help tests
# ---------------------------------------------------------------------------

SIG_HELP_CASES = [
    pytest.param(
        "fn f(x: f32) -> f32 { exp( }",
        0, 27, "exp", 0,
        id="builtin-first-param",
    ),
    pytest.param(
        "fn f(x: f32, y: f32) -> f32 { x }\nfn g(a: f32) -> f32 { f(a, }",
        1, 27, "f", 1,
        id="user-fn-second-param",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0, 23, None, 0,
        id="not-in-call",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { sum( }",
        0, 27, "sum", 0,
        id="sum-builtin",
    ),
    pytest.param(
        "fn f(x: f32, y: f32) -> f32 { x }\nfn g(a: f32) -> f32 { f( }",
        1, 24, "f", 0,
        id="user-fn-first-param",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { tanh(exp( }",
        0, 32, "exp", 0,
        id="nested-call-inner",
    ),
]


@pytest.mark.parametrize(
    "source, line_0, col_0, expected_fn_name, expected_param_idx",
    SIG_HELP_CASES,
)
def test_signature_help(source, line_0, col_0, expected_fn_name, expected_param_idx):
    fn_name, param_idx = _sig_help_at(source, line_0, col_0)
    assert fn_name == expected_fn_name, (
        f"Expected fn_name={expected_fn_name}, got {fn_name}"
    )
    assert param_idx == expected_param_idx, (
        f"Expected param_idx={expected_param_idx}, got {param_idx}"
    )


# ---------------------------------------------------------------------------
# Inlay hints tests
# ---------------------------------------------------------------------------

INLAY_HINTS_CASES = [
    pytest.param(
        "fn f(x: f32) -> f32 { let a = x; a }",
        1, 1, 1,
        id="one-untyped-let",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { let a: f32 = x; a }",
        1, 1, 0,
        id="explicit-annotation-no-hint",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        1, 1, 0,
        id="no-let-no-hints",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 {\n    let a = x;\n    let b = a;\n    b\n}",
        1, 4, 2,
        id="two-lets-two-hints",
    ),
    pytest.param(
        "fn f(x: f32, y: f32) -> f32 {\n    let s = x + y;\n    s\n}",
        1, 3, 1,
        id="let-binop-hint",
    ),
]


@pytest.mark.parametrize(
    "source, start_line_1, end_line_1, expected_count", INLAY_HINTS_CASES
)
def test_inlay_hints(source, start_line_1, end_line_1, expected_count):
    count = _inlay_hint_count(source, start_line_1, end_line_1)
    assert count == expected_count, (
        f"Expected {expected_count} inlay hints, got {count}"
    )


# ---------------------------------------------------------------------------
# Document symbols tests
# ---------------------------------------------------------------------------

DOC_SYMBOLS_CASES = [
    pytest.param(
        "fn f(x: f32) -> f32 { x }\nfn g(y: f32) -> f32 { y }",
        {"f", "g"},
        id="two-functions",
    ),
    pytest.param(
        "struct P { x: f32 }\nfn f(p: P) -> f32 { p.x }",
        {"P", "f"},
        id="struct-and-fn",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        {"f"},
        id="single-fn",
    ),
    pytest.param(
        "struct A { a: f32 }\nstruct B { b: i32 }\nfn f(x: f32) -> f32 { x }",
        {"A", "B", "f"},
        id="two-structs-and-fn",
    ),
]


@pytest.mark.parametrize("source, expected_names", DOC_SYMBOLS_CASES)
def test_document_symbols(source, expected_names):
    names = _doc_symbol_names(source)
    assert names == expected_names, f"Expected {expected_names}, got {names}"


# ---------------------------------------------------------------------------
# Highlights tests
# ---------------------------------------------------------------------------

HIGHLIGHTS_CASES = [
    pytest.param(
        "fn f(x: f32) -> f32 { x + x }",
        1, 23, 3,
        id="param-decl-plus-two-uses",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { 1.0 }",
        1, 23, None,
        id="literal-no-highlights",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        1, 23, 2,
        id="param-decl-plus-one-use",
    ),
    pytest.param(
        "fn a(x: f32) -> f32 { x }\nfn b(y: f32) -> f32 { a(y) + a(y) }",
        1, 4, 3,
        id="fn-decl-plus-two-calls",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { let a = x; a + a }",
        1, 35, 3,
        id="let-var-highlights",
    ),
]


@pytest.mark.parametrize(
    "source, line_1, col_1, expected_count", HIGHLIGHTS_CASES
)
def test_highlights(source, line_1, col_1, expected_count):
    count = _highlight_count(source, line_1, col_1)
    if expected_count is None:
        assert count is None, f"Expected no highlights, got {count}"
    else:
        assert count is not None, "Expected highlights, got None"
        assert count == expected_count, (
            f"Expected {expected_count} highlights, got {count}"
        )


# ---------------------------------------------------------------------------
# Formatting tests
# ---------------------------------------------------------------------------

FORMAT_CASES = [
    pytest.param(
        "fn f() {\n    x;\n}\n",
        False,
        id="already-formatted",
    ),
    pytest.param(
        "fn f() {\nx;\n}\n",
        True,
        id="needs-indent",
    ),
    pytest.param(
        "fn f(){\n x;\n}\n",
        True,
        id="needs-brace-spacing",
    ),
    pytest.param(
        "fn f() {\n    let a = 1;\n    a\n}\n",
        False,
        id="multi-stmt-formatted",
    ),
    pytest.param(
        "fn f() {\nlet a = 1;\na\n}\n",
        True,
        id="multi-stmt-needs-indent",
    ),
]


@pytest.mark.parametrize("source, has_edits", FORMAT_CASES)
def test_formatting(source, has_edits):
    result = _format_has_edits(source)
    assert result == has_edits, (
        f"Expected has_edits={has_edits}, got {result}"
    )


# ---------------------------------------------------------------------------
# Folding ranges tests
# ---------------------------------------------------------------------------

FOLDING_CASES = [
    pytest.param(
        "fn f(x: f32) -> f32 {\n    x\n}",
        1,
        id="multiline-fn-one-fold",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 { x }",
        0,
        id="single-line-no-fold",
    ),
    pytest.param(
        "fn f(x: f32) -> f32 {\n    x\n}\nfn g(y: f32) -> f32 {\n    y\n}",
        2,
        id="two-multiline-fns",
    ),
]


@pytest.mark.parametrize("source, expected_count", FOLDING_CASES)
def test_folding_ranges(source, expected_count):
    count = _folding_range_count(source)
    assert count == expected_count, (
        f"Expected {expected_count} folding ranges, got {count}"
    )
