"""Tests for expand_dims, squeeze, and broadcast_to builtins."""
import pytest

import numpy as np

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.codegen.stablehlo import StableHLOCodegen
from maomi.ad import transform_grad


def _parse(src: str):
    tokens = Lexer(src).tokenize()
    return Parser(tokens).parse()


def _check(src: str):
    prog = _parse(src)
    tc = TypeChecker()
    errors = tc.check(prog)
    return prog, tc, errors


def _check_ok(src: str):
    _, _, errors = _check(src)
    assert not errors, errors


def _check_err(src: str, fragment: str = ""):
    _, _, errors = _check(src)
    assert errors, f"expected error containing '{fragment}'"
    if fragment:
        assert any(fragment in str(e) for e in errors), (
            f"expected '{fragment}' in errors, got: {errors}"
        )


def _compile(src: str) -> str:
    prog = _parse(src)
    tc = TypeChecker()
    errors = tc.check(prog)
    assert not errors, errors
    prog = transform_grad(prog, tc.type_map)
    gen = StableHLOCodegen(prog, tc.type_map)
    return gen.generate()


# ---------- Type checker: expand_dims ----------


class TestExpandDimsTypeChecker:
    def test_front(self):
        _check_ok("fn f(x: f32[3, 4]) -> f32[1, 3, 4] { expand_dims(x, 0) }")

    def test_middle(self):
        _check_ok("fn f(x: f32[3, 4]) -> f32[3, 1, 4] { expand_dims(x, 1) }")

    def test_end(self):
        _check_ok("fn f(x: f32[3, 4]) -> f32[3, 4, 1] { expand_dims(x, 2) }")

    def test_scalar(self):
        _check_ok("fn f(x: f32) -> f32[1] { expand_dims(x, 0) }")

    def test_1d(self):
        _check_ok("fn f(x: f32[5]) -> f32[1, 5] { expand_dims(x, 0) }")

    def test_1d_end(self):
        _check_ok("fn f(x: f32[5]) -> f32[5, 1] { expand_dims(x, 1) }")

    def test_axis_out_of_range(self):
        _check_err(
            "fn f(x: f32[3, 4]) -> f32[3, 4, 1] { expand_dims(x, 3) }",
            "out of range",
        )

    def test_negative_axis(self):
        # -1 is parsed as UnaryOp("-", 1), not IntLiteral(-1), so it's not a literal
        _check_err(
            "fn f(x: f32[3, 4]) -> f32[3, 4] { expand_dims(x, -1) }",
            "integer literal",
        )

    def test_wrong_arg_count(self):
        _check_err(
            "fn f(x: f32[3, 4]) -> f32[3, 4] { expand_dims(x) }",
            "exactly 2 arguments",
        )

    def test_non_literal_axis(self):
        _check_err(
            "fn f(x: f32[3, 4], a: i32) -> f32[3, 4] { expand_dims(x, a) }",
            "integer literal",
        )


# ---------- Type checker: squeeze ----------


class TestSqueezeTypeChecker:
    def test_middle(self):
        _check_ok("fn f(x: f32[3, 1, 4]) -> f32[3, 4] { squeeze(x, 1) }")

    def test_front(self):
        _check_ok("fn f(x: f32[1, 4]) -> f32[4] { squeeze(x, 0) }")

    def test_end(self):
        _check_ok("fn f(x: f32[3, 1]) -> f32[3] { squeeze(x, 1) }")

    def test_to_scalar(self):
        _check_ok("fn f(x: f32[1]) -> f32 { squeeze(x, 0) }")

    def test_not_size_1(self):
        _check_err(
            "fn f(x: f32[3, 4]) -> f32[3] { squeeze(x, 0) }",
            "size 1",
        )

    def test_axis_out_of_range(self):
        _check_err(
            "fn f(x: f32[3, 1, 4]) -> f32[3, 4] { squeeze(x, 3) }",
            "out of range",
        )

    def test_wrong_arg_count(self):
        _check_err(
            "fn f(x: f32[3, 1, 4]) -> f32[3, 4] { squeeze(x) }",
            "exactly 2 arguments",
        )

    def test_non_array(self):
        _check_err(
            "fn f(x: f32) -> f32 { squeeze(x, 0) }",
            "must be an array",
        )


# ---------- Type checker: broadcast_to ----------


class TestBroadcastToTypeChecker:
    def test_1d_to_2d(self):
        _check_ok("fn f(x: f32[4]) -> f32[3, 4] { broadcast_to(x, 3, 4) }")

    def test_scalar_to_2d(self):
        _check_ok("fn f(x: f32) -> f32[3, 4] { broadcast_to(x, 3, 4) }")

    def test_size1_broadcast(self):
        _check_ok("fn f(x: f32[1, 4]) -> f32[3, 4] { broadcast_to(x, 3, 4) }")

    def test_same_shape(self):
        _check_ok("fn f(x: f32[3, 4]) -> f32[3, 4] { broadcast_to(x, 3, 4) }")

    def test_scalar_to_1d(self):
        _check_ok("fn f(x: f32) -> f32[5] { broadcast_to(x, 5) }")

    def test_incompatible(self):
        _check_err(
            "fn f(x: f32[3]) -> f32[4, 4] { broadcast_to(x, 4, 4) }",
            "not compatible",
        )

    def test_rank_too_high(self):
        _check_err(
            "fn f(x: f32[3, 4]) -> f32[4] { broadcast_to(x, 4) }",
            "rank",
        )

    def test_too_few_args(self):
        _check_err(
            "fn f(x: f32[4]) -> f32[4] { broadcast_to(x) }",
            "at least 2 arguments",
        )


# ---------- Codegen ----------


class TestExpandDimsCodegen:
    def test_reshape_emitted(self):
        out = _compile("fn f(x: f32[3, 4]) -> f32[1, 3, 4] { expand_dims(x, 0) }")
        assert "stablehlo.reshape" in out

    def test_middle(self):
        out = _compile("fn f(x: f32[3, 4]) -> f32[3, 1, 4] { expand_dims(x, 1) }")
        assert "stablehlo.reshape" in out

    def test_scalar(self):
        out = _compile("fn f(x: f32) -> f32[1] { expand_dims(x, 0) }")
        assert "stablehlo.reshape" in out


class TestSqueezeCodegen:
    def test_reshape_emitted(self):
        out = _compile("fn f(x: f32[3, 1, 4]) -> f32[3, 4] { squeeze(x, 1) }")
        assert "stablehlo.reshape" in out

    def test_to_scalar(self):
        out = _compile("fn f(x: f32[1]) -> f32 { squeeze(x, 0) }")
        assert "stablehlo.reshape" in out


class TestBroadcastToCodegen:
    def test_broadcast_in_dim_emitted(self):
        out = _compile("fn f(x: f32[4]) -> f32[3, 4] { broadcast_to(x, 3, 4) }")
        assert "stablehlo.broadcast_in_dim" in out

    def test_scalar(self):
        out = _compile("fn f(x: f32) -> f32[3, 4] { broadcast_to(x, 3, 4) }")
        assert "stablehlo.broadcast_in_dim" in out

    def test_size1(self):
        out = _compile("fn f(x: f32[1, 4]) -> f32[3, 4] { broadcast_to(x, 3, 4) }")
        assert "stablehlo.broadcast_in_dim" in out


# ---------- AD ----------


class TestExpandDimsAD:
    def test_grad_through_expand_dims(self):
        out = _compile(
            "fn f(x: f32[2, 2]) -> f32 { sum(expand_dims(x, 0)) }\n"
            "fn grad_f(x: f32[2, 2]) -> f32[2, 2] { grad(f(x), x) }"
        )
        assert "stablehlo.reshape" in out

    def test_grad_expand_dims_end(self):
        out = _compile(
            "fn f(x: f32[3]) -> f32 { sum(expand_dims(x, 1)) }\n"
            "fn grad_f(x: f32[3]) -> f32[3] { grad(f(x), x) }"
        )
        assert "stablehlo.reshape" in out


class TestSqueezeAD:
    def test_grad_through_squeeze(self):
        out = _compile(
            "fn f(x: f32[3, 1, 4]) -> f32 { sum(squeeze(x, 1)) }\n"
            "fn grad_f(x: f32[3, 1, 4]) -> f32[3, 1, 4] { grad(f(x), x) }"
        )
        assert "stablehlo.reshape" in out


class TestBroadcastToAD:
    def test_grad_through_broadcast_to(self):
        out = _compile(
            "fn f(x: f32[4]) -> f32 { sum(broadcast_to(x, 3, 4)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }"
        )
        assert "grad_f" in out

    def test_grad_scalar_broadcast(self):
        out = _compile(
            "fn f(x: f32) -> f32 { sum(broadcast_to(x, 3, 4)) }\n"
            "fn grad_f(x: f32) -> f32 { grad(f(x), x) }"
        )
        assert "grad_f" in out
