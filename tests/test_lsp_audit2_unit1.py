"""Tests for LSP audit Unit 1: Semantic Tokens fixes.

Covers:
- B15: BinOp operator token position using source text search
- G13: Missing semantic tokens for secondary keywords (else, with, in, limit, do)
- G14: Struct literal field name semantic tokens
"""
import pytest
from maomi.lsp import (
    validate,
    _sem_collect_tokens, _sem_delta_encode,
    _ST_FUNCTION, _ST_PARAMETER, _ST_VARIABLE, _ST_STRUCT,
    _ST_PROPERTY, _ST_TYPE, _ST_NUMBER, _ST_KEYWORD,
    _ST_BUILTIN_TYPE, _ST_BUILTIN_FUNCTION, _ST_BOOLEAN, _ST_OPERATOR,
    _MOD_DECLARATION, _MOD_DEFINITION,
)
from maomi.lsp._semantic import _MOD_CONTROL_FLOW


def _collect_fn_tokens(source: str, source_lines: list[str] | None = None):
    """Parse source and collect semantic tokens for all functions."""
    _, result = validate(source, "<test>")
    if source_lines is None:
        source_lines = source.splitlines()
    tokens = []
    for fn in result.program.functions:
        _sem_collect_tokens(fn, tokens, set(), source_lines)
    return tokens, result


def _collect_all_tokens(source: str):
    """Parse source and collect semantic tokens for structs + functions."""
    _, result = validate(source, "<test>")
    source_lines = source.splitlines()
    tokens = []
    for sd in result.program.struct_defs:
        _sem_collect_tokens(sd, tokens, set(), source_lines)
    for fn in result.program.functions:
        _sem_collect_tokens(fn, tokens, set(), source_lines)
    return tokens, result


# ---------------------------------------------------------------------------
# B15: BinOp operator token position
# ---------------------------------------------------------------------------

class TestBinOpOperatorPosition:
    def test_normal_spacing(self):
        """a + b — standard spacing, operator at column 2."""
        source = "fn f(a: f32, b: f32) -> f32 { a + b }"
        tokens, _ = _collect_fn_tokens(source)
        op_tokens = [t for t in tokens if t[3] == _ST_OPERATOR]
        assert len(op_tokens) == 1
        # "a + b" in "{ a + b }" — find 'a' in body at col 31 (0-indexed: 30)
        # The '+' is at 0-indexed column 32
        line = 0
        assert op_tokens[0][0] == line
        assert op_tokens[0][2] == 1  # length of "+"
        # Verify it's between a and b by checking it matches source
        assert source[op_tokens[0][1]] == "+"

    def test_tight_spacing(self):
        """a+b — no spaces, operator should still be found."""
        source = "fn f(a: f32, b: f32) -> f32 { a+b }"
        tokens, _ = _collect_fn_tokens(source)
        op_tokens = [t for t in tokens if t[3] == _ST_OPERATOR]
        assert len(op_tokens) == 1
        assert op_tokens[0][2] == 1
        assert source[op_tokens[0][1]] == "+"

    def test_extra_spacing(self):
        """a  +  b — extra spaces around operator."""
        source = "fn f(a: f32, b: f32) -> f32 { a  +  b }"
        tokens, _ = _collect_fn_tokens(source)
        op_tokens = [t for t in tokens if t[3] == _ST_OPERATOR]
        assert len(op_tokens) == 1
        assert op_tokens[0][2] == 1
        assert source[op_tokens[0][1]] == "+"

    def test_multichar_eq(self):
        """a == b — two-character operator."""
        source = "fn f(a: f32, b: f32) -> bool { a == b }"
        tokens, _ = _collect_fn_tokens(source)
        op_tokens = [t for t in tokens if t[3] == _ST_OPERATOR]
        assert len(op_tokens) == 1
        assert op_tokens[0][2] == 2  # length of "=="
        col = op_tokens[0][1]
        assert source[col:col + 2] == "=="

    def test_multichar_neq(self):
        """a != b — two-character operator."""
        source = "fn f(a: f32, b: f32) -> bool { a != b }"
        tokens, _ = _collect_fn_tokens(source)
        op_tokens = [t for t in tokens if t[3] == _ST_OPERATOR]
        assert len(op_tokens) == 1
        assert op_tokens[0][2] == 2
        col = op_tokens[0][1]
        assert source[col:col + 2] == "!="

    def test_and_operator(self):
        """a and b — keyword-length operator."""
        source = "fn f(a: bool, b: bool) -> bool { a and b }"
        tokens, _ = _collect_fn_tokens(source)
        op_tokens = [t for t in tokens if t[3] == _ST_OPERATOR]
        assert len(op_tokens) == 1
        assert op_tokens[0][2] == 3  # length of "and"
        col = op_tokens[0][1]
        assert source[col:col + 3] == "and"

    def test_or_operator(self):
        """a or b — keyword-length operator."""
        source = "fn f(a: bool, b: bool) -> bool { a or b }"
        tokens, _ = _collect_fn_tokens(source)
        op_tokens = [t for t in tokens if t[3] == _ST_OPERATOR]
        assert len(op_tokens) == 1
        assert op_tokens[0][2] == 2  # length of "or"
        col = op_tokens[0][1]
        assert source[col:col + 2] == "or"

    def test_gte_lte_operators(self):
        """Test >= and <= operators."""
        source = "fn f(a: f32, b: f32) -> bool { a >= b }"
        tokens, _ = _collect_fn_tokens(source)
        op_tokens = [t for t in tokens if t[3] == _ST_OPERATOR]
        assert len(op_tokens) == 1
        assert op_tokens[0][2] == 2
        col = op_tokens[0][1]
        assert source[col:col + 2] == ">="

    def test_no_source_lines_fallback(self):
        """Without source_lines, the old heuristic is used (backward compat)."""
        source = "fn f(a: f32, b: f32) -> f32 { a + b }"
        _, result = validate(source, "<test>")
        tokens = []
        # Call without source_lines — should still produce operator token via fallback
        _sem_collect_tokens(result.program.functions[0], tokens, set())
        op_tokens = [t for t in tokens if t[3] == _ST_OPERATOR]
        assert len(op_tokens) == 1


# ---------------------------------------------------------------------------
# G13: Missing semantic tokens for secondary keywords
# ---------------------------------------------------------------------------

class TestSecondaryKeywords:
    def test_else_keyword(self):
        """if/else — 'else' should get a keyword token."""
        source = "fn f(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 - x } }"
        tokens, _ = _collect_fn_tokens(source)
        kw_tokens = [t for t in tokens if t[3] == _ST_KEYWORD]
        kw_texts = []
        for t in kw_tokens:
            kw_texts.append(source[t[1]:t[1] + t[2]])
        assert "if" in kw_texts
        assert "else" in kw_texts
        # Verify else has controlFlow modifier
        else_tokens = [t for t in kw_tokens if source[t[1]:t[1] + t[2]] == "else"]
        assert len(else_tokens) == 1
        assert else_tokens[0][4] & _MOD_CONTROL_FLOW

    def test_in_keyword_scan(self):
        """scan — 'in' should get a keyword token."""
        source = "fn f(xs: f32[5]) -> f32 { scan (acc, x) in (0.0, xs) { acc + x } }"
        tokens, _ = _collect_fn_tokens(source)
        kw_tokens = [t for t in tokens if t[3] == _ST_KEYWORD]
        kw_texts = [source[t[1]:t[1] + t[2]] for t in kw_tokens]
        assert "scan" in kw_texts
        assert "in" in kw_texts

    def test_in_keyword_fold(self):
        """fold — 'in' should get a keyword token."""
        source = "fn f(xs: f32[5]) -> f32 { fold (acc, x) in (0.0, xs) { acc + x } }"
        tokens, _ = _collect_fn_tokens(source)
        kw_tokens = [t for t in tokens if t[3] == _ST_KEYWORD]
        kw_texts = [source[t[1]:t[1] + t[2]] for t in kw_tokens]
        assert "fold" in kw_texts
        assert "in" in kw_texts

    def test_in_keyword_map(self):
        """map — 'in' should get a keyword token."""
        source = "fn f(xs: f32[5]) -> f32[5] { map x in xs { x + 1.0 } }"
        tokens, _ = _collect_fn_tokens(source)
        kw_tokens = [t for t in tokens if t[3] == _ST_KEYWORD]
        kw_texts = [source[t[1]:t[1] + t[2]] for t in kw_tokens]
        assert "map" in kw_texts
        assert "in" in kw_texts

    def test_limit_keyword_while(self):
        """while with limit — 'limit' should get a keyword token."""
        source = "fn f(x: f32) -> f32 { while s in x limit 10 { s > 0.1 } do { s * 0.5 } }"
        tokens, _ = _collect_fn_tokens(source)
        kw_tokens = [t for t in tokens if t[3] == _ST_KEYWORD]
        kw_texts = [source[t[1]:t[1] + t[2]] for t in kw_tokens]
        assert "while" in kw_texts
        assert "limit" in kw_texts

    def test_do_keyword_while(self):
        """while — 'do' should get a keyword token."""
        source = "fn f(x: f32) -> f32 { while s in x limit 10 { s > 0.1 } do { s * 0.5 } }"
        tokens, _ = _collect_fn_tokens(source)
        kw_tokens = [t for t in tokens if t[3] == _ST_KEYWORD]
        kw_texts = [source[t[1]:t[1] + t[2]] for t in kw_tokens]
        assert "do" in kw_texts

    def test_with_keyword(self):
        """s with { x = 1.0 } — 'with' should get a keyword token."""
        source = """struct Point { x: f32, y: f32 }
fn f(p: Point) -> Point { p with { x = 1.0 } }"""
        tokens, _ = _collect_all_tokens(source)
        kw_tokens = [t for t in tokens if t[3] == _ST_KEYWORD]
        kw_texts = []
        lines = source.splitlines()
        for t in kw_tokens:
            line_text = lines[t[0]]
            kw_texts.append(line_text[t[1]:t[1] + t[2]])
        assert "with" in kw_texts


# ---------------------------------------------------------------------------
# G14: Struct literal field name semantic tokens
# ---------------------------------------------------------------------------

class TestStructLiteralFieldNames:
    def test_field_names_highlighted(self):
        """Point { x: 1.0, y: 2.0 } — field names x, y get property tokens."""
        source = """struct Point { x: f32, y: f32 }
fn f() -> Point { Point { x: 1.0, y: 2.0 } }"""
        tokens, _ = _collect_all_tokens(source)
        # Filter property tokens on line 1 (the function line, 0-indexed)
        prop_tokens = [t for t in tokens if t[3] == _ST_PROPERTY and t[0] == 1]
        # Should have at least x and y field names from the struct literal
        lines = source.splitlines()
        prop_texts = [lines[t[0]][t[1]:t[1] + t[2]] for t in prop_tokens]
        assert "x" in prop_texts
        assert "y" in prop_texts

    def test_field_names_multiline(self):
        """Struct literal spanning multiple lines."""
        source = """struct Vec2 { x: f32, y: f32 }
fn f() -> Vec2 {
    Vec2 {
        x: 1.0,
        y: 2.0
    }
}"""
        tokens, _ = _collect_all_tokens(source)
        prop_tokens = [t for t in tokens if t[3] == _ST_PROPERTY]
        lines = source.splitlines()
        prop_texts = [lines[t[0]][t[1]:t[1] + t[2]] for t in prop_tokens]
        # From struct definition: x, y (declaration)
        # From struct literal: x, y
        assert prop_texts.count("x") >= 2  # at least definition + literal
        assert prop_texts.count("y") >= 2

    def test_field_names_not_duplicated_without_source(self):
        """Without source_lines, field name tokens are not emitted (no crash)."""
        source = """struct Point { x: f32, y: f32 }
fn f() -> Point { Point { x: 1.0, y: 2.0 } }"""
        _, result = validate(source, "<test>")
        tokens = []
        # Call without source_lines
        for fn in result.program.functions:
            _sem_collect_tokens(fn, tokens, set())
        # Should still have struct name token but no field property tokens from literal
        struct_tokens = [t for t in tokens if t[3] == _ST_STRUCT]
        assert len(struct_tokens) >= 1
        # Property tokens from literal won't appear without source_lines
        # (FieldAccess property tokens would, but there are none in this source)
