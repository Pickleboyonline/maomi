"""Tests for reshape and concat builtins."""
import pytest

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


# ---------- Type checker: reshape ----------


class TestReshapeTypeChecker:
    def test_flatten(self):
        _check_ok("fn f(x: f32[4, 8]) -> f32[32] { reshape(x, 32) }")

    def test_unflatten(self):
        _check_ok("fn f(x: f32[32]) -> f32[4, 8] { reshape(x, 4, 8) }")

    def test_3d(self):
        _check_ok("fn f(x: f32[24]) -> f32[2, 3, 4] { reshape(x, 2, 3, 4) }")

    def test_same_shape(self):
        _check_ok("fn f(x: f32[4, 8]) -> f32[4, 8] { reshape(x, 4, 8) }")

    def test_element_count_mismatch(self):
        _check_err(
            "fn f(x: f32[4, 8]) -> f32[10] { reshape(x, 10) }",
            "input has 32 elements but target shape has 10",
        )

    def test_too_few_args(self):
        _check_err(
            "fn f(x: f32[4]) -> f32[4] { reshape(x) }",
            "at least 2 arguments",
        )

    def test_non_literal_dim(self):
        _check_err(
            "fn f(x: f32[4], n: i32) -> f32[4] { reshape(x, n) }",
            "integer literals",
        )

    def test_scalar_input(self):
        _check_err(
            "fn f(x: f32) -> f32[1] { reshape(x, 1) }",
            "first argument must be an array",
        )


# ---------- Type checker: concat ----------


class TestConcatTypeChecker:
    def test_1d(self):
        _check_ok("fn f(a: f32[4], b: f32[6]) -> f32[10] { concat(a, b) }")

    def test_2d(self):
        _check_ok("fn f(a: f32[4, 8], b: f32[6, 8]) -> f32[10, 8] { concat(a, b) }")

    def test_three_arrays(self):
        _check_ok("fn f(a: f32[2], b: f32[3], c: f32[5]) -> f32[10] { concat(a, b, c) }")

    def test_axis_1(self):
        _check_ok("fn f(a: f32[4, 3], b: f32[4, 5]) -> f32[4, 8] { concat(a, b, 1) }")

    def test_rank_mismatch(self):
        _check_err(
            "fn f(a: f32[4], b: f32[4, 8]) -> f32[8] { concat(a, b) }",
            "rank mismatch",
        )

    def test_dim_mismatch(self):
        _check_err(
            "fn f(a: f32[4, 8], b: f32[4, 10]) -> f32[8, 8] { concat(a, b) }",
            "dimension mismatch",
        )

    def test_type_mismatch(self):
        _check_err(
            "fn f(a: f32[4], b: i32[4]) -> f32[8] { concat(a, b) }",
            "base type mismatch",
        )

    def test_too_few_args(self):
        _check_err(
            "fn f(a: f32[4]) -> f32[4] { concat(a) }",
            "at least 2",
        )

    def test_axis_out_of_range(self):
        _check_err(
            "fn f(a: f32[4], b: f32[4]) -> f32[8] { concat(a, b, 2) }",
            "axis 2 out of range",
        )

    def test_scalar_args(self):
        _check_err(
            "fn f(a: f32, b: f32) -> f32 { concat(a, b) }",
            "must be an array",
        )


# ---------- Codegen: reshape ----------


class TestReshapeCodegen:
    def test_emits_reshape_op(self):
        out = _compile("fn f(x: f32[4, 8]) -> f32[32] { reshape(x, 32) }")
        assert "stablehlo.reshape" in out

    def test_correct_types(self):
        out = _compile("fn f(x: f32[4, 8]) -> f32[32] { reshape(x, 32) }")
        assert "tensor<4x8xf32>" in out
        assert "tensor<32xf32>" in out


# ---------- Codegen: concat ----------


class TestConcatCodegen:
    def test_emits_concatenate_op(self):
        out = _compile("fn f(a: f32[4], b: f32[6]) -> f32[10] { concat(a, b) }")
        assert "stablehlo.concatenate" in out

    def test_dim_attr(self):
        out = _compile("fn f(a: f32[4], b: f32[6]) -> f32[10] { concat(a, b) }")
        assert "dim = 0" in out

    def test_axis_1(self):
        out = _compile("fn f(a: f32[4, 3], b: f32[4, 5]) -> f32[4, 8] { concat(a, b, 1) }")
        assert "dim = 1" in out
        assert "stablehlo.concatenate" in out

    def test_correct_types(self):
        out = _compile("fn f(a: f32[4], b: f32[6]) -> f32[10] { concat(a, b) }")
        assert "tensor<4xf32>" in out
        assert "tensor<6xf32>" in out
        assert "tensor<10xf32>" in out


# ---------- AD: reshape ----------


class TestADReshape:
    def test_grad_through_reshape(self):
        out = _compile(
            "fn f(x: f32[2, 4]) -> f32[2, 4] { grad(sum(reshape(x, 8)), x) }"
        )
        assert "stablehlo.reshape" in out

    def test_grad_reshape_with_mean(self):
        out = _compile(
            "fn f(x: f32[2, 4]) -> f32[2, 4] { grad(mean(reshape(x, 8)), x) }"
        )
        assert "stablehlo.reshape" in out
        assert "stablehlo.divide" in out


# ---------- AD: concat ----------


class TestADConcat:
    def test_grad_concat_wrt_first(self):
        # grad(sum(concat(a, b)), a) = ones(4) — scalar adj broadcasts
        out = _compile(
            "fn f(a: f32[4], b: f32[6]) -> f32[4] { grad(sum(concat(a, b)), a) }"
        )
        assert "dense<1.000000e+00>" in out

    def test_grad_concat_wrt_second(self):
        # grad(sum(concat(a, b)), b) = ones(6)
        out = _compile(
            "fn f(a: f32[4], b: f32[6]) -> f32[6] { grad(sum(concat(a, b)), b) }"
        )
        assert "dense<1.000000e+00>" in out

    def test_grad_concat_with_multiply(self):
        # When adj is an array (not scalar), concat backward slices
        # sum(concat(a, b) * c) → adj of concat = 1.0 * c = c (array)
        # then concat slices c into c[0:4] and c[4:10]
        out = _compile(
            "fn f(a: f32[4], b: f32[6], c: f32[10]) -> f32[4] { grad(sum(concat(a, b) * c), a) }"
        )
        assert "stablehlo.slice" in out

    def test_grad_concat_compiles(self):
        # Just verify the full pipeline doesn't crash
        out = _compile(
            "fn f(a: f32[3], b: f32[7]) -> f32[3] { grad(mean(concat(a, b)), a) }"
        )
        assert "module {" in out
