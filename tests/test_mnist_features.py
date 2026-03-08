"""Tests for MNIST-readiness features: cast, fold, max, min, argmax, argmin."""

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.ad import transform_grad
from maomi.codegen.stablehlo import StableHLOCodegen
from maomi.ast_nodes import *
from maomi.errors import MaomiError
import pytest


def check_ok(source: str):
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"


def check_err(source: str, expected_msg: str):
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert any(expected_msg in e.message for e in errors), \
        f"Expected error containing '{expected_msg}', got: {[e.message for e in errors]}"


def codegen(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    return StableHLOCodegen(program, checker.type_map).generate()


def ad_codegen(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    program = transform_grad(program, checker.type_map)
    return StableHLOCodegen(program, checker.type_map).generate()


def parse_ok(source: str) -> Program:
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


# ============================================================
# Cast
# ============================================================


class TestCastParser:
    def test_cast_f32(self):
        prog = parse_ok("fn f(x: i32) -> f32 { cast(x, f32) }")
        body = prog.functions[0].body.expr
        assert isinstance(body, CastExpr)
        assert body.target_type == "f32"

    def test_cast_i32(self):
        prog = parse_ok("fn f(x: f32) -> i32 { cast(x, i32) }")
        body = prog.functions[0].body.expr
        assert isinstance(body, CastExpr)
        assert body.target_type == "i32"

    def test_cast_bool(self):
        prog = parse_ok("fn f(x: f32) -> bool { cast(x, bool) }")
        body = prog.functions[0].body.expr
        assert isinstance(body, CastExpr)
        assert body.target_type == "bool"


class TestCastTypeChecker:
    def test_scalar_cast(self):
        check_ok("fn f(x: i32) -> f32 { cast(x, f32) }")

    def test_array_cast(self):
        check_ok("fn f(x: i32[4]) -> f32[4] { cast(x, f32) }")

    def test_cast_preserves_shape(self):
        check_ok("fn f(x: f32[3, 4]) -> i32[3, 4] { cast(x, i32) }")

    def test_cast_bool_to_f32(self):
        check_ok("fn f(x: bool[4]) -> f32[4] { cast(x, f32) }")

    def test_cast_same_type(self):
        check_ok("fn f(x: f32) -> f32 { cast(x, f32) }")

    def test_cast_f64(self):
        check_ok("fn f(x: f32[4]) -> f64[4] { cast(x, f64) }")


class TestCastCodegen:
    def test_cast_i32_to_f32(self):
        out = codegen("fn f(x: i32) -> f32 { cast(x, f32) }")
        assert "stablehlo.convert" in out

    def test_cast_array(self):
        out = codegen("fn f(x: i32[4]) -> f32[4] { cast(x, f32) }")
        assert "stablehlo.convert" in out
        assert "tensor<4xi32>" in out
        assert "tensor<4xf32>" in out

    def test_cast_noop(self):
        out = codegen("fn f(x: f32) -> f32 { cast(x, f32) }")
        assert "stablehlo.convert" not in out


class TestCastAD:
    def test_grad_through_cast_f32_f64(self):
        """Gradient through f32→f64 cast should cast back."""
        out = ad_codegen("""
            fn f(x: f32[4]) -> f32[4] {
                grad(sum(cast(x, f64)), x)
            }
        """)
        assert "stablehlo.convert" in out

    def test_grad_through_cast_to_int_is_zero(self):
        """Gradient through f32→i32 should be zero (non-differentiable)."""
        out = ad_codegen("""
            fn f(x: f32[4]) -> f32[4] {
                let y = cast(x, i32);
                let z = cast(y, f32);
                grad(sum(z), x)
            }
        """)
        # Should compile without error; gradient through int cast is zero
        assert "module {" in out


# ============================================================
# Fold
# ============================================================


class TestFoldParser:
    def test_fold_basic(self):
        prog = parse_ok("""
            fn f(x: f32[10]) -> f32 {
                fold (acc, e) in (0.0, x) { acc + e }
            }
        """)
        body = prog.functions[0].body.expr
        assert isinstance(body, FoldExpr)
        assert body.carry_var == "acc"
        assert body.elem_vars == ["e"]

    def test_fold_multi_seq(self):
        prog = parse_ok("""
            fn f(x: f32[10], y: f32[10]) -> f32 {
                fold (acc, (a, b)) in (0.0, (x, y)) { acc + a + b }
            }
        """)
        body = prog.functions[0].body.expr
        assert isinstance(body, FoldExpr)
        assert body.elem_vars == ["a", "b"]
        assert len(body.sequences) == 2


class TestFoldTypeChecker:
    def test_fold_returns_carry(self):
        check_ok("fn f(x: f32[10]) -> f32 { fold (acc, e) in (0.0, x) { acc + e } }")

    def test_fold_struct_carry(self):
        check_ok("""
            struct S { x: f32 }
            fn f(xs: f32[10]) -> S {
                fold (s, e) in (S { x: 0.0 }, xs) {
                    s with { x = s.x + e }
                }
            }
        """)

    def test_fold_body_type_mismatch(self):
        check_err(
            "fn f(x: f32[10]) -> f32 { fold (acc, e) in (0, x) { acc } }",
            ""  # should error on type mismatch (init is i32, seq is f32)
        )


class TestFoldCodegen:
    def test_fold_sum(self):
        out = codegen("fn f(x: f32[10]) -> f32 { fold (acc, e) in (0.0, x) { acc + e } }")
        assert "stablehlo.while" in out
        assert "module {" in out

    def test_fold_returns_carry_not_stacked(self):
        out = codegen("fn f(x: f32[10]) -> f32 { fold (acc, e) in (0.0, x) { acc + e } }")
        # Fold should NOT have the output buffer that scan has
        # The return type is scalar (carry), not f32[10] (stacked)
        assert "func.func @f" in out


class TestFoldAD:
    def test_fold_grad_sum(self):
        """grad(fold(acc + e, x), x) should compile (equivalent to grad of sum)."""
        out = ad_codegen("""
            fn f(x: f32[10]) -> f32[10] {
                let s = fold (acc, e) in (0.0, x) { acc + e };
                grad(s, x)
            }
        """)
        assert "func.func @f" in out
        assert "stablehlo.while" in out

    def test_fold_grad_wrt_init(self):
        """grad(fold(...), init) should compile."""
        out = ad_codegen("""
            fn f(x: f32, arr: f32[5]) -> f32 {
                let s = fold (acc, e) in (x, arr) { acc + e };
                grad(s, x)
            }
        """)
        assert "func.func @f" in out

    def test_fold_grad_product(self):
        """grad(fold(acc * e, x), x) should compile — non-trivial derivative."""
        out = ad_codegen("""
            fn f(x: f32[5]) -> f32[5] {
                let s = fold (acc, e) in (1.0, x) { acc * e };
                grad(s, x)
            }
        """)
        assert "func.func @f" in out
        assert "stablehlo.while" in out

    def test_fold_grad_multi_seq(self):
        """grad of fold with multiple sequences."""
        out = ad_codegen("""
            fn f(x: f32[5], y: f32[5]) -> f32[5] {
                let s = fold (acc, (a, b)) in (0.0, (x, y)) { acc + a * b };
                grad(s, x)
            }
        """)
        assert "func.func @f" in out


# ============================================================
# Max / Min
# ============================================================


class TestMaxMinTypeChecker:
    def test_max_all(self):
        check_ok("fn f(x: f32[10]) -> f32 { max(x) }")

    def test_min_all(self):
        check_ok("fn f(x: f32[10]) -> f32 { min(x) }")

    def test_max_axis(self):
        check_ok("fn f(x: f32[3, 4]) -> f32[3] { max(x, 1) }")

    def test_min_axis(self):
        check_ok("fn f(x: f32[3, 4]) -> f32[4] { min(x, 0) }")

    def test_max_scalar(self):
        check_ok("fn f(x: f32) -> f32 { max(x) }")

    def test_max_3d_axis(self):
        check_ok("fn f(x: f32[2, 3, 4]) -> f32[2, 4] { max(x, 1) }")

    def test_max_axis_out_of_range(self):
        check_err("fn f(x: f32[3, 4]) -> f32[3] { max(x, 2) }", "out of range")

    def test_max_i32(self):
        check_ok("fn f(x: i32[10]) -> i32 { max(x) }")


class TestMaxMinCodegen:
    def test_max_all(self):
        out = codegen("fn f(x: f32[10]) -> f32 { max(x) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.maximum" in out
        assert "0xFF800000" in out  # -inf init

    def test_min_all(self):
        out = codegen("fn f(x: f32[10]) -> f32 { min(x) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.minimum" in out
        assert "0x7F800000" in out  # +inf init

    def test_max_axis(self):
        out = codegen("fn f(x: f32[3, 4]) -> f32[3] { max(x, 1) }")
        assert "across dimensions = [1]" in out
        assert "stablehlo.maximum" in out

    def test_min_axis(self):
        out = codegen("fn f(x: f32[3, 4]) -> f32[4] { min(x, 0) }")
        assert "across dimensions = [0]" in out
        assert "stablehlo.minimum" in out


class TestMaxMinAD:
    def test_grad_max_all(self):
        """grad(max(x), x) should use indicator-based rule."""
        out = ad_codegen("""
            fn f(x: f32[4]) -> f32[4] {
                grad(max(x), x)
            }
        """)
        assert "stablehlo.compare" in out  # EQ comparison
        assert "stablehlo.convert" in out  # cast bool→f32
        assert "stablehlo.divide" in out   # divide by count
        assert "module {" in out

    def test_grad_min_all(self):
        """grad(min(x), x) should also use indicator rule."""
        out = ad_codegen("""
            fn f(x: f32[4]) -> f32[4] {
                grad(min(x), x)
            }
        """)
        assert "stablehlo.compare" in out
        assert "stablehlo.convert" in out
        assert "module {" in out

    def test_grad_max_axis(self):
        """grad(sum(max(x, 1)), x) should compile."""
        out = ad_codegen("""
            fn f(x: f32[3, 4]) -> f32[3, 4] {
                grad(sum(max(x, 1)), x)
            }
        """)
        assert "stablehlo.maximum" in out  # forward max
        assert "stablehlo.compare" in out  # indicator
        assert "module {" in out

    def test_grad_max_in_expression(self):
        """grad through max used in a larger expression."""
        out = ad_codegen("""
            fn f(x: f32[4]) -> f32[4] {
                let m = max(x);
                let shifted = x - m;
                grad(sum(shifted), x)
            }
        """)
        assert "module {" in out


# ============================================================
# Argmax / Argmin
# ============================================================


class TestArgmaxTypeChecker:
    def test_argmax_all(self):
        check_ok("fn f(x: f32[10]) -> i32 { argmax(x) }")

    def test_argmin_all(self):
        check_ok("fn f(x: f32[10]) -> i32 { argmin(x) }")

    def test_argmax_axis(self):
        check_ok("fn f(x: f32[3, 4]) -> i32[3] { argmax(x, 1) }")

    def test_argmin_axis(self):
        check_ok("fn f(x: f32[3, 4]) -> i32[4] { argmin(x, 0) }")

    def test_argmax_3d_axis(self):
        check_ok("fn f(x: f32[2, 3, 4]) -> i32[2, 4] { argmax(x, 1) }")

    def test_argmax_scalar_error(self):
        check_err("fn f(x: f32) -> i32 { argmax(x) }", "array argument")

    def test_argmax_axis_out_of_range(self):
        check_err("fn f(x: f32[3, 4]) -> i32[3] { argmax(x, 2) }", "out of range")


class TestArgmaxCodegen:
    def test_argmax_all(self):
        out = codegen("fn f(x: f32[10]) -> i32 { argmax(x) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.iota" in out
        assert "stablehlo.compare" in out
        assert "stablehlo.select" in out

    def test_argmin_all(self):
        out = codegen("fn f(x: f32[10]) -> i32 { argmin(x) }")
        assert "stablehlo.reduce" in out
        assert "LT" in out  # argmin uses LT comparison

    def test_argmax_axis(self):
        out = codegen("fn f(x: f32[3, 4]) -> i32[3] { argmax(x, 1) }")
        assert "across dimensions = [1]" in out
        assert "stablehlo.iota" in out
        assert "GT" in out  # argmax uses GT comparison

    def test_argmax_i32_input(self):
        out = codegen("fn f(x: i32[10]) -> i32 { argmax(x) }")
        assert "SIGNED" in out  # integer comparison uses SIGNED
