"""Fuzz-all-positions LSP test.

Sweeps every (line, col) position in a diverse corpus of .mao sources
and calls every LSP helper function.  Any unhandled exception = test failure.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from lsprotocol import types

from maomi.lsp import (
    validate,
    _cache,
    _local_functions,
    _find_node_at,
    _get_hover_text,
    _goto_find_definition,
    _goto_type_definition,
    _build_document_highlights,
    _sel_collect_ancestors,
    _complete_general,
    _complete_dot,
    _sig_parse_call_context,
    prepare_rename_at,
    rename_at,
    _build_document_symbols,
    _build_folding_ranges,
    _sem_collect_tokens,
    _build_inlay_hints,
    _format_document,
    _build_code_lenses,
    _call_hierarchy_prepare,
    _call_hierarchy_incoming,
    _call_hierarchy_outgoing,
    _workspace_symbols,
    code_actions,
)

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

INLINE_CORPUS: list[str] = [
    # 1  Simple function
    "fn f(x: f32) -> f32 { x }",
    # 2  Struct + with + field access
    "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> Point { p with { x = p.y } }",
    # 3  Scan
    "fn f(xs: f32[3]) -> f32[3] { scan (acc, x) in (0.0, xs) { acc + x } }",
    # 4  Map
    "fn f(xs: f32[5]) -> f32[5] { map x in xs { x * 2.0 } }",
    # 5  Grad
    "fn loss(w: f32) -> f32 { w * w }\nfn g(w: f32) -> f32 { grad(loss(w), w) }",
    # 6  Indexing
    "fn f(x: f32[10]) -> f32 { x[0] }",
    # 7  Cast
    "fn f(x: f32) -> i32 { cast(x, i32) }",
    # 8  If/else
    "fn f(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 - x } }",
    # 9  Multiple functions
    "fn a(x: f32) -> f32 { x }\nfn b(y: f32) -> f32 { a(y) + a(y) }",
    # 10 Nested structs
    "struct Inner { v: f32 }\nstruct Outer { i: Inner }\nfn f(o: Outer) -> f32 { o.i.v }",
    # 11 Builtins
    "fn f(x: f32) -> f32 { exp(log(tanh(x))) }",
    # 12 Empty function
    "fn f() -> f32 { 0.0 }",
    # 13 Doc comments
    "/// A docstring\nfn f(x: f32) -> f32 { x }",
    # 14 Parse error: broken fn
    "fn f( -> { }",
    # 15 Parse error: broken struct
    "struct { }",
    # 16 Empty source
    "",
    # 17 Parse error: incomplete fn
    "fn",
    # 18 Type error: return type mismatch
    "fn f(x: f32) -> i32 { x }",
    # 19 Fold
    "fn f(xs: f32[3]) -> f32 { fold (acc, x) in (0.0, xs) { acc + x } }",
    # 20 Let binding
    "fn f(x: f32) -> f32 { let a = x + 1.0; a * a }",
    # 21 Multi-line function
    (
        "fn add(a: f32[2, 3], b: f32[2, 3]) -> f32[2, 3] {\n"
        "    let c = a + b;\n"
        "    let d = c * c;\n"
        "    d\n"
        "}"
    ),
    # 22 Slicing
    "fn f(x: f32[10]) -> f32[3] { x[1:4] }",
    # 23 Neg / abs pattern
    "fn f(x: f32) -> f32 { neg(x) }",
    # 24 Transpose
    "fn f(x: f32[2, 3]) -> f32[3, 2] { transpose(x) }",
    # 25 Matmul
    "fn f(x: f32[2, 3], y: f32[3, 4]) -> f32[2, 4] { matmul(x, y) }",
    # 26 Where
    "fn f(c: bool[3], x: f32[3], y: f32[3]) -> f32[3] { where(c, x, y) }",
    # 27 Iota
    "fn f() -> i32[5] { iota(5) }",
    # 28 Multiple let bindings + nested calls
    (
        "fn f(x: f32) -> f32 {\n"
        "    let a = exp(x);\n"
        "    let b = log(a);\n"
        "    let c = sqrt(b + 1.0);\n"
        "    c * c\n"
        "}"
    ),
    # 29 Struct with multiple fields and operations
    (
        "struct Vec2 { x: f32, y: f32 }\n"
        "fn dot(a: Vec2, b: Vec2) -> f32 {\n"
        "    a.x * b.x + a.y * b.y\n"
        "}"
    ),
    # 30 Comments mixed in
    (
        "// A comment\n"
        "fn f(x: f32) -> f32 {\n"
        "    // inner comment\n"
        "    x + 1.0\n"
        "}"
    ),
    # 31 Value and grad
    "fn loss(w: f32) -> f32 { w * w }\n"
    "fn g(w: f32) -> f32 { let vg = value_and_grad(loss(w), w); vg.gradient }",
    # 32 Stop gradient
    "fn f(x: f32) -> f32 { stop_gradient(x) + x }",
    # 33 Sigmoid / cos / sin
    "fn f(x: f32) -> f32 { sigmoid(cos(sin(x))) }",
    # 34 Reduction with axis
    "fn f(x: f32[3, 4]) -> f32[3] { sum(x, axis=1) }",
    # 35 Mean with keepdims
    "fn f(x: f32[3, 4]) -> f32[3, 1] { mean(x, axis=1, keepdims=true) }",
    # 36 Argmax
    "fn f(x: f32[5]) -> i32 { argmax(x) }",
    # 37 Various parse errors to stress LSP robustness
    "fn f(x: f32 { }",
    "fn (x: f32) -> f32 { x }",
    "struct Point { x: , y: f32 }",
    "fn f(x: f32) -> f32 { let = x; x }",
]


def _load_file_corpus() -> list[str]:
    """Load all .mao files from fixtures and examples directories."""
    corpus: list[str] = []
    for d in [FIXTURES_DIR, EXAMPLES_DIR]:
        if d.exists():
            for p in sorted(d.rglob("*.mao")):
                try:
                    text = p.read_text()
                    if text:
                        corpus.append(text)
                except Exception:
                    pass
    return corpus


ALL_CORPUS: list[str] = INLINE_CORPUS + _load_file_corpus()


# ---------------------------------------------------------------------------
# Fuzz driver
# ---------------------------------------------------------------------------

def _fuzz_source(source: str) -> None:
    """Call every LSP helper function at every position in *source*.

    Any unhandled exception propagates as a test failure.
    """
    uri = "file:///fuzz.mao"
    diags, result = validate(source, "<fuzz>")
    lines = source.splitlines() if source else []
    num_lines = len(lines)

    # ------------------------------------------------------------------
    # Position-insensitive calls (once per source)
    # ------------------------------------------------------------------
    _build_document_symbols(result)
    _build_folding_ranges(result)
    _format_document(source)
    _build_inlay_hints(result, 1, max(num_lines, 1), source)

    if result.program:
        fns = _local_functions(result.program)
        for fn in fns:
            tokens: list = []
            _sem_collect_tokens(fn, tokens, set())
        _build_code_lenses(result, uri)
        for fn in fns:
            _call_hierarchy_incoming(result, uri, fn.name)
            _call_hierarchy_outgoing(result, uri, fn.name)

    # Workspace symbols — requires populating _cache
    _cache[uri] = result
    try:
        _workspace_symbols("")
        _workspace_symbols("f")
    finally:
        _cache.pop(uri, None)

    # Code actions — requires _cache + diagnostics
    if diags:
        _cache[uri] = result
        try:
            params = types.CodeActionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                range=diags[0].range,
                context=types.CodeActionContext(diagnostics=diags),
            )
            code_actions(None, params)
        finally:
            _cache.pop(uri, None)

    # ------------------------------------------------------------------
    # Position-sensitive: sweep every (line, col)
    # ------------------------------------------------------------------
    for line_0 in range(num_lines):
        line_1 = line_0 + 1
        line_text = lines[line_0]
        for col_0 in range(len(line_text) + 1):
            col_1 = col_0 + 1
            pos = types.Position(line=line_0, character=col_0)

            # 0-indexed helpers
            _complete_general(result, pos)
            _complete_dot(result, pos)
            _sig_parse_call_context(source, pos)
            prepare_rename_at(source, result, line_0, col_0)
            rename_at(source, result, line_0, col_0, "zzz")

            if result.program:
                _call_hierarchy_prepare(result, uri, line_0, col_0)

            # 1-indexed helpers
            _build_document_highlights(result, line_1, col_1)

            if result.program:
                for fn in _local_functions(result.program):
                    ancestors: list = []
                    _sel_collect_ancestors(fn, line_1, col_1, ancestors)

                    node = _find_node_at(fn, line_1, col_1)
                    if node is not None:
                        _get_hover_text(node, fn, result)
                        _goto_find_definition(node, fn, result)
                        _goto_type_definition(node, fn, result)


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------

def _corpus_id(source: str) -> str:
    """Generate a short, readable test id from the first line of source."""
    if not source:
        return "<empty>"
    first_line = source.split("\n", 1)[0]
    return first_line[:60].replace("\n", "\\n")


@pytest.mark.parametrize("source", ALL_CORPUS, ids=_corpus_id)
def test_no_crash_at_any_position(source: str) -> None:
    """Every LSP function must survive at every cursor position without crashing."""
    _fuzz_source(source)
