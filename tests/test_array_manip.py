"""Tests for flip, tril, triu array manipulation builtins."""
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


# ==========================================================================
# flip
# ==========================================================================


class TestFlipTypeChecker:
    def test_1d(self):
        _check_ok("fn f(x: f32[4]) -> f32[4] { flip(x, 0) }")

    def test_2d_axis0(self):
        _check_ok("fn f(x: f32[3, 4]) -> f32[3, 4] { flip(x, 0) }")

    def test_2d_axis1(self):
        _check_ok("fn f(x: f32[3, 4]) -> f32[3, 4] { flip(x, 1) }")

    def test_3d(self):
        _check_ok("fn f(x: f32[2, 3, 4]) -> f32[2, 3, 4] { flip(x, 2) }")

    def test_axis_out_of_range(self):
        _check_err(
            "fn f(x: f32[3, 4]) -> f32[3, 4] { flip(x, 2) }",
            "out of range",
        )

    def test_missing_axis(self):
        _check_err(
            "fn f(x: f32[4]) -> f32[4] { flip(x) }",
            "expects exactly 2 arguments",
        )

    def test_scalar_input(self):
        _check_err(
            "fn f(x: f32) -> f32 { flip(x, 0) }",
            "requires an array",
        )


class TestFlipCodegen:
    def test_1d_reverse(self):
        mlir = _compile("fn f(x: f32[4]) -> f32[4] { flip(x, 0) }")
        assert "stablehlo.reverse" in mlir
        assert "dims = [0]" in mlir

    def test_2d_axis1(self):
        mlir = _compile("fn f(x: f32[3, 4]) -> f32[3, 4] { flip(x, 1) }")
        assert "stablehlo.reverse" in mlir
        assert "dims = [1]" in mlir


class TestFlipAD:
    def test_grad_sum_flip(self):
        src = """
        fn loss(x: f32[4]) -> f32 {
            sum(flip(x, 0))
        }
        fn dloss(x: f32[4]) -> f32[4] {
            grad(loss(x), x)
        }
        """
        mlir = _compile(src)
        assert "stablehlo.reverse" in mlir

    def test_grad_preserves_shape(self):
        src = """
        fn loss(x: f32[3, 4]) -> f32 {
            sum(flip(x, 1))
        }
        fn dloss(x: f32[3, 4]) -> f32[3, 4] {
            grad(loss(x), x)
        }
        """
        mlir = _compile(src)
        assert "stablehlo.reverse" in mlir


# ==========================================================================
# tril
# ==========================================================================


class TestTrilTypeChecker:
    def test_square(self):
        _check_ok("fn f(x: f32[3, 3]) -> f32[3, 3] { tril(x) }")

    def test_rectangular(self):
        _check_ok("fn f(x: f32[2, 4]) -> f32[2, 4] { tril(x) }")

    def test_f64(self):
        _check_ok("fn f(x: f64[3, 3]) -> f64[3, 3] { tril(x) }")

    def test_1d_error(self):
        _check_err(
            "fn f(x: f32[4]) -> f32[4] { tril(x) }",
            "2D",
        )

    def test_3d_error(self):
        _check_err(
            "fn f(x: f32[2, 3, 4]) -> f32[2, 3, 4] { tril(x) }",
            "2D",
        )

    def test_scalar_error(self):
        _check_err(
            "fn f(x: f32) -> f32 { tril(x) }",
            "2D array",
        )

    def test_int_array_error(self):
        _check_err(
            "fn f(x: i32[3, 3]) -> i32[3, 3] { tril(x) }",
            "float",
        )

    def test_too_many_args(self):
        _check_err(
            "fn f(x: f32[3, 3]) -> f32[3, 3] { tril(x, 0) }",
            "expects exactly 1 argument",
        )


class TestTrilCodegen:
    def test_square(self):
        mlir = _compile("fn f(x: f32[3, 3]) -> f32[3, 3] { tril(x) }")
        assert "stablehlo.iota" in mlir
        assert "stablehlo.compare GE" in mlir
        assert "stablehlo.select" in mlir

    def test_rectangular(self):
        mlir = _compile("fn f(x: f32[2, 4]) -> f32[2, 4] { tril(x) }")
        assert "stablehlo.iota" in mlir
        assert "stablehlo.compare GE" in mlir
        assert "stablehlo.select" in mlir


class TestTrilAD:
    def test_grad_sum_tril(self):
        src = """
        fn loss(x: f32[3, 3]) -> f32 {
            sum(tril(x))
        }
        fn dloss(x: f32[3, 3]) -> f32[3, 3] {
            grad(loss(x), x)
        }
        """
        mlir = _compile(src)
        # The gradient should apply tril to the adjoint
        # There should be at least 2 tril patterns (forward + backward)
        assert mlir.count("stablehlo.compare GE") >= 2


# ==========================================================================
# triu
# ==========================================================================


class TestTriuTypeChecker:
    def test_square(self):
        _check_ok("fn f(x: f32[3, 3]) -> f32[3, 3] { triu(x) }")

    def test_rectangular(self):
        _check_ok("fn f(x: f32[2, 4]) -> f32[2, 4] { triu(x) }")

    def test_f64(self):
        _check_ok("fn f(x: f64[3, 3]) -> f64[3, 3] { triu(x) }")

    def test_1d_error(self):
        _check_err(
            "fn f(x: f32[4]) -> f32[4] { triu(x) }",
            "2D",
        )

    def test_3d_error(self):
        _check_err(
            "fn f(x: f32[2, 3, 4]) -> f32[2, 3, 4] { triu(x) }",
            "2D",
        )

    def test_scalar_error(self):
        _check_err(
            "fn f(x: f32) -> f32 { triu(x) }",
            "2D array",
        )

    def test_int_array_error(self):
        _check_err(
            "fn f(x: i32[3, 3]) -> i32[3, 3] { triu(x) }",
            "float",
        )


class TestTriuCodegen:
    def test_square(self):
        mlir = _compile("fn f(x: f32[3, 3]) -> f32[3, 3] { triu(x) }")
        assert "stablehlo.iota" in mlir
        assert "stablehlo.compare LE" in mlir
        assert "stablehlo.select" in mlir

    def test_rectangular(self):
        mlir = _compile("fn f(x: f32[2, 4]) -> f32[2, 4] { triu(x) }")
        assert "stablehlo.iota" in mlir
        assert "stablehlo.compare LE" in mlir
        assert "stablehlo.select" in mlir


class TestTriuAD:
    def test_grad_sum_triu(self):
        src = """
        fn loss(x: f32[3, 3]) -> f32 {
            sum(triu(x))
        }
        fn dloss(x: f32[3, 3]) -> f32[3, 3] {
            grad(loss(x), x)
        }
        """
        mlir = _compile(src)
        # The gradient should apply triu to the adjoint
        assert mlir.count("stablehlo.compare LE") >= 2
