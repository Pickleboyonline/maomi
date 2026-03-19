"""Tests for small LSP fixes: folding, builtin sort, named args, string handling."""

from lsprotocol import types

from maomi.lsp import (
    validate,
    _build_folding_ranges,
    _sig_parse_call_context,
    _BUILTINS,
)


# ---------------------------------------------------------------------------
# G23: FoldExpr and WhileExpr should produce folding ranges
# ---------------------------------------------------------------------------

class TestFoldingFoldWhile:
    def test_fold_expr_produces_folding_range(self):
        source = """fn f(xs: f32[10]) -> f32 {
    fold (carry, x) in (0.0, xs) {
        carry + x
    }
}"""
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        # Should have ranges for the function AND the fold body
        assert len(ranges) >= 2

    def test_while_expr_produces_folding_range(self):
        source = """fn f(x: f32) -> f32 {
    while s in x limit 10 {
        s > 0.0
    } do {
        s - 1.0
    }
}"""
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        # Should have ranges for the function AND the while body
        assert len(ranges) >= 2


# ---------------------------------------------------------------------------
# Builtin sort order: "config" should be in sorted position
# ---------------------------------------------------------------------------

class TestBuiltinSortOrder:
    def test_builtins_is_sorted(self):
        assert _BUILTINS == sorted(_BUILTINS), (
            f"_BUILTINS is not sorted; 'config' at index {_BUILTINS.index('config')}"
        )

    def test_config_in_builtins(self):
        assert "config" in _BUILTINS


# ---------------------------------------------------------------------------
# G18: Named arguments in signature help
# ---------------------------------------------------------------------------

class TestSignatureHelpNamedArgs:
    def test_named_arg_returns_param_name(self):
        source = "fn f(x: f32) -> f32 { sum(x, axis=1) }"
        # Cursor after "axis=" — inside the named argument
        # "sum(x, axis=1)" starts at col 22
        # 's' at 22, 'u' at 23, 'm' at 24, '(' at 25, 'x' at 26, ',' at 27
        # ' ' at 28, 'a' at 29, 'x' at 30, 'i' at 31, 's' at 32, '=' at 33, '1' at 34
        name, comma_count, named_param = _sig_parse_call_context(
            source, types.Position(line=0, character=34),
        )
        assert name == "sum"
        assert named_param == "axis"

    def test_positional_arg_returns_no_named_param(self):
        source = "fn f(x: f32) -> f32 { exp(x) }"
        name, comma_count, named_param = _sig_parse_call_context(
            source, types.Position(line=0, character=27),
        )
        assert name == "exp"
        assert named_param is None


# ---------------------------------------------------------------------------
# E6: Commas/parens inside strings should be ignored
# ---------------------------------------------------------------------------

class TestSignatureHelpStringHandling:
    def test_commas_in_string_ignored(self):
        source = 'fn f() -> f32 { helper("a,b,c", x) }'
        # "helper(" starts at col 16, '(' at 22
        # '"a,b,c"' is cols 23-29, ',' at 30, ' ' at 31, 'x' at 32
        # Cursor at 'x' — should be param index 1 (one real comma), not 3
        name, idx, _ = _sig_parse_call_context(
            source, types.Position(line=0, character=32),
        )
        assert name == "helper"
        assert idx == 1

    def test_parens_in_string_ignored(self):
        source = 'fn f() -> f32 { helper("()", x) }'
        # Parens inside the string should not affect depth tracking
        name, idx, _ = _sig_parse_call_context(
            source, types.Position(line=0, character=28),
        )
        assert name == "helper"
        assert idx == 1
