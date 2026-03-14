"""Tests for linear algebra builtins: cholesky, triangular_solve."""
from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.codegen.stablehlo import StableHLOCodegen
from maomi.errors import MaomiError
from maomi.ad import transform_grad
import pytest


def codegen(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    return StableHLOCodegen(program, checker.type_map).generate()


def type_check_fails(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert len(errors) > 0, "Expected type error"
    return errors[0].message


def codegen_ad(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    from maomi.resolver import resolve
    program = resolve(program, "/tmp/test.mao")
    tc = TypeChecker()
    errors = tc.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    program = transform_grad(program, tc.type_map)
    return StableHLOCodegen(program, tc.type_map).generate()


# ============================================================
# Cholesky
# ============================================================


class TestCholeskyCodegen:
    def test_basic_cholesky(self):
        out = codegen("fn f(x: f32[3,3]) -> f32[3,3] { cholesky(x) }")
        assert "stablehlo.cholesky" in out
        assert "tensor<3x3xf32>" in out

    def test_cholesky_f64(self):
        out = codegen("fn f(x: f64[4,4]) -> f64[4,4] { cholesky(x) }")
        assert "stablehlo.cholesky" in out
        assert "tensor<4x4xf64>" in out

    def test_cholesky_in_let_binding(self):
        out = codegen("""
            fn f(x: f32[3,3]) -> f32[3,3] {
                let L = cholesky(x);
                L
            }
        """)
        assert "stablehlo.cholesky" in out

    def test_cholesky_lower_attribute(self):
        """Verify the lower=true attribute is present in output."""
        out = codegen("fn f(x: f32[3,3]) -> f32[3,3] { cholesky(x) }")
        assert "lower = true" in out


class TestCholeskyTypeErrors:
    def test_not_2d(self):
        msg = type_check_fails("fn f(x: f32[3]) -> f32[3] { cholesky(x) }")
        assert "2D" in msg

    def test_not_square(self):
        msg = type_check_fails("fn f(x: f32[3,4]) -> f32[3,4] { cholesky(x) }")
        assert "square" in msg

    def test_integer_type(self):
        msg = type_check_fails("fn f(x: i32[3,3]) -> i32[3,3] { cholesky(x) }")
        assert "float" in msg

    def test_wrong_arg_count(self):
        msg = type_check_fails("fn f(x: f32[3,3], y: f32[3,3]) -> f32[3,3] { cholesky(x, y) }")
        assert "1 argument" in msg

    def test_3d_input(self):
        msg = type_check_fails("fn f(x: f32[2,3,3]) -> f32[2,3,3] { cholesky(x) }")
        assert "2D" in msg


# ============================================================
# Triangular Solve
# ============================================================


class TestTriangularSolveCodegen:
    def test_basic_left_side_lower(self):
        out = codegen(
            "fn f(a: f32[3,3], b: f32[3,2]) -> f32[3,2] { triangular_solve(a, b, true, true) }"
        )
        assert "stablehlo.triangular_solve" in out
        assert "left_side = true" in out
        assert "lower = true" in out
        assert "tensor<3x3xf32>" in out
        assert "tensor<3x2xf32>" in out

    def test_upper_triangular(self):
        out = codegen(
            "fn f(a: f32[3,3], b: f32[3,2]) -> f32[3,2] { triangular_solve(a, b, false, true) }"
        )
        assert "lower = false" in out
        assert "left_side = true" in out

    def test_right_side(self):
        out = codegen(
            "fn f(a: f32[3,3], b: f32[2,3]) -> f32[2,3] { triangular_solve(a, b, true, false) }"
        )
        assert "left_side = false" in out
        assert "lower = true" in out

    def test_right_side_upper(self):
        out = codegen(
            "fn f(a: f32[4,4], b: f32[2,4]) -> f32[2,4] { triangular_solve(a, b, false, false) }"
        )
        assert "left_side = false" in out
        assert "lower = false" in out

    def test_unit_diagonal_false(self):
        """Verify unit_diagonal is always false."""
        out = codegen(
            "fn f(a: f32[3,3], b: f32[3,2]) -> f32[3,2] { triangular_solve(a, b, true, true) }"
        )
        assert "unit_diagonal = false" in out

    def test_no_transpose(self):
        """Verify transpose_a = NO_TRANSPOSE is in the output."""
        out = codegen(
            "fn f(a: f32[3,3], b: f32[3,2]) -> f32[3,2] { triangular_solve(a, b, true, true) }"
        )
        assert "NO_TRANSPOSE" in out

    def test_f64(self):
        out = codegen(
            "fn f(a: f64[3,3], b: f64[3,2]) -> f64[3,2] { triangular_solve(a, b, true, true) }"
        )
        assert "stablehlo.triangular_solve" in out
        assert "tensor<3x3xf64>" in out
        assert "tensor<3x2xf64>" in out

    def test_in_let_binding(self):
        out = codegen("""
            fn f(a: f32[3,3], b: f32[3,2]) -> f32[3,2] {
                let x = triangular_solve(a, b, true, true);
                x
            }
        """)
        assert "stablehlo.triangular_solve" in out


class TestTriangularSolveTypeErrors:
    def test_dim_mismatch_left_side(self):
        msg = type_check_fails(
            "fn f(a: f32[3,3], b: f32[4,2]) -> f32[4,2] { triangular_solve(a, b, true, true) }"
        )
        assert "dimension" in msg.lower() or "match" in msg.lower() or "4" in msg

    def test_dim_mismatch_right_side(self):
        msg = type_check_fails(
            "fn f(a: f32[3,3], b: f32[2,4]) -> f32[2,4] { triangular_solve(a, b, true, false) }"
        )
        assert "dimension" in msg.lower() or "match" in msg.lower() or "4" in msg

    def test_a_not_square(self):
        msg = type_check_fails(
            "fn f(a: f32[3,4], b: f32[3,2]) -> f32[3,2] { triangular_solve(a, b, true, true) }"
        )
        assert "square" in msg

    def test_a_not_2d(self):
        msg = type_check_fails(
            "fn f(a: f32[3], b: f32[3,2]) -> f32[3,2] { triangular_solve(a, b, true, true) }"
        )
        assert "2D" in msg

    def test_b_not_2d(self):
        msg = type_check_fails(
            "fn f(a: f32[3,3], b: f32[3]) -> f32[3] { triangular_solve(a, b, true, true) }"
        )
        assert "2D" in msg

    def test_wrong_arg_count(self):
        msg = type_check_fails(
            "fn f(a: f32[3,3], b: f32[3,2]) -> f32[3,2] { triangular_solve(a, b, true) }"
        )
        assert "4 arguments" in msg

    def test_type_mismatch(self):
        msg = type_check_fails(
            "fn f(a: f32[3,3], b: f64[3,2]) -> f64[3,2] { triangular_solve(a, b, true, true) }"
        )
        assert "base type" in msg.lower() or "same" in msg.lower()

    def test_integer_a(self):
        msg = type_check_fails(
            "fn f(a: i32[3,3], b: f32[3,2]) -> f32[3,2] { triangular_solve(a, b, true, true) }"
        )
        assert "float" in msg


# ============================================================
# Cholesky + Triangular Solve combined
# ============================================================


class TestLinalgCombined:
    def test_cholesky_then_solve(self):
        """Classic pattern: Cholesky decompose, then solve."""
        out = codegen("""
            fn f(a: f32[3,3], b: f32[3,2]) -> f32[3,2] {
                let L = cholesky(a);
                triangular_solve(L, b, true, true)
            }
        """)
        assert "stablehlo.cholesky" in out
        assert "stablehlo.triangular_solve" in out


# ============================================================
# AD behavior (nondiff)
# ============================================================


class TestLinalgAD:
    def test_cholesky_grad_errors(self):
        """Differentiating through cholesky should raise an error."""
        with pytest.raises(MaomiError, match="not yet supported"):
            codegen_ad("""
                fn f(x: f32[3,3]) -> f32 {
                    let L = cholesky(x);
                    sum(L)
                }
                fn main(x: f32[3,3]) -> f32[3,3] {
                    grad(f(x), x)
                }
            """)

    def test_triangular_solve_grad_errors(self):
        """Differentiating through triangular_solve should raise an error."""
        with pytest.raises(MaomiError, match="not yet supported"):
            codegen_ad("""
                fn f(a: f32[3,3], b: f32[3,2]) -> f32 {
                    let x = triangular_solve(a, b, true, true);
                    sum(x)
                }
                fn main(a: f32[3,3], b: f32[3,2]) -> f32[3,3] {
                    grad(f(a, b), a)
                }
            """)
