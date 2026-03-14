"""Tests for new two-arg elementwise builtins: logaddexp, hypot, remainder, copysign."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.codegen.stablehlo import StableHLOCodegen
from maomi.ad import transform_grad
from maomi.resolver import resolve
from maomi.errors import MaomiError


# -- Helpers --

def codegen(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    return StableHLOCodegen(program, checker.type_map).generate()


def codegen_ad(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    program = resolve(program, "/tmp/test.mao")
    tc = TypeChecker()
    errors = tc.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    program = transform_grad(program, tc.type_map)
    return StableHLOCodegen(program, tc.type_map).generate()


def type_check_fails(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert len(errors) > 0, "Expected type error"
    return errors[0].message


# ============================================================
# logaddexp
# ============================================================

class TestLogaddexp:
    def test_scalar_codegen(self):
        out = codegen("fn f(a: f32, b: f32) -> f32 { logaddexp(a, b) }")
        assert "stablehlo.maximum" in out
        assert "stablehlo.subtract" in out
        assert "stablehlo.abs" in out
        assert "stablehlo.negate" in out
        assert "stablehlo.exponential" in out
        assert "stablehlo.log_plus_one" in out
        assert "stablehlo.add" in out

    def test_array_codegen(self):
        out = codegen("fn f(a: f32[4], b: f32[4]) -> f32[4] { logaddexp(a, b) }")
        assert "stablehlo.maximum" in out
        assert "stablehlo.log_plus_one" in out

    def test_type_error_wrong_arg_count(self):
        msg = type_check_fails("fn f(a: f32) -> f32 { logaddexp(a) }")
        assert "2 arguments" in msg

    def test_type_error_int_args(self):
        msg = type_check_fails("fn f(a: i32, b: i32) -> i32 { logaddexp(a, b) }")
        assert "float" in msg.lower()

    def test_broadcasting(self):
        out = codegen("fn f(a: f32[4], b: f32) -> f32[4] { logaddexp(a, b) }")
        assert "stablehlo.broadcast_in_dim" in out

    def test_grad(self):
        out = codegen_ad("""
            fn f(a: f32, b: f32) -> f32 {
                grad(logaddexp(a, b), a)
            }
        """)
        # Gradient uses sigmoid(a-b) — which involves exponential, negate, divide, etc.
        assert "stablehlo" in out

    def test_grad_array(self):
        out = codegen_ad("""
            fn f(a: f32[4], b: f32[4]) -> f32[4] {
                grad(sum(logaddexp(a, b)), a)
            }
        """)
        assert "stablehlo" in out


# ============================================================
# hypot
# ============================================================

class TestHypot:
    def test_scalar_codegen(self):
        out = codegen("fn f(a: f32, b: f32) -> f32 { hypot(a, b) }")
        assert "stablehlo.multiply" in out
        assert "stablehlo.add" in out
        assert "stablehlo.sqrt" in out

    def test_array_codegen(self):
        out = codegen("fn f(a: f32[3], b: f32[3]) -> f32[3] { hypot(a, b) }")
        assert "stablehlo.multiply" in out
        assert "stablehlo.sqrt" in out

    def test_type_error_wrong_arg_count(self):
        msg = type_check_fails("fn f(a: f32) -> f32 { hypot(a) }")
        assert "2 arguments" in msg

    def test_type_error_int_args(self):
        msg = type_check_fails("fn f(a: i32, b: i32) -> i32 { hypot(a, b) }")
        assert "float" in msg.lower()

    def test_broadcasting(self):
        out = codegen("fn f(a: f32[3], b: f32) -> f32[3] { hypot(a, b) }")
        assert "stablehlo.broadcast_in_dim" in out

    def test_grad(self):
        out = codegen_ad("""
            fn f(a: f32, b: f32) -> f32 {
                grad(hypot(a, b), a)
            }
        """)
        # Gradient: a / hypot(a, b) — involves divide
        assert "stablehlo.divide" in out

    def test_grad_array(self):
        out = codegen_ad("""
            fn f(a: f32[4], b: f32[4]) -> f32[4] {
                grad(sum(hypot(a, b)), a)
            }
        """)
        assert "stablehlo" in out


# ============================================================
# remainder
# ============================================================

class TestRemainder:
    def test_scalar_codegen(self):
        out = codegen("fn f(a: f32, b: f32) -> f32 { remainder(a, b) }")
        assert "stablehlo.divide" in out
        assert "stablehlo.floor" in out
        assert "stablehlo.multiply" in out
        assert "stablehlo.subtract" in out

    def test_array_codegen(self):
        out = codegen("fn f(a: f32[3], b: f32[3]) -> f32[3] { remainder(a, b) }")
        assert "stablehlo.divide" in out
        assert "stablehlo.floor" in out

    def test_type_error_wrong_arg_count(self):
        msg = type_check_fails("fn f(a: f32) -> f32 { remainder(a) }")
        assert "2 arguments" in msg

    def test_type_error_int_args(self):
        msg = type_check_fails("fn f(a: i32, b: i32) -> i32 { remainder(a, b) }")
        assert "float" in msg.lower()

    def test_broadcasting(self):
        out = codegen("fn f(a: f32[3], b: f32) -> f32[3] { remainder(a, b) }")
        assert "stablehlo.broadcast_in_dim" in out

    def test_grad(self):
        out = codegen_ad("""
            fn f(a: f32, b: f32) -> f32 {
                grad(remainder(a, b), a)
            }
        """)
        # d/da remainder = 1, so the adjoint should pass through directly
        assert "stablehlo" in out

    def test_grad_b(self):
        out = codegen_ad("""
            fn f(a: f32, b: f32) -> f32 {
                grad(remainder(a, b), b)
            }
        """)
        # d/db remainder = -floor(a/b)
        assert "stablehlo.floor" in out


# ============================================================
# copysign
# ============================================================

class TestCopysign:
    def test_scalar_codegen(self):
        out = codegen("fn f(a: f32, b: f32) -> f32 { copysign(a, b) }")
        assert "stablehlo.abs" in out
        assert "stablehlo.sign" in out
        assert "stablehlo.multiply" in out

    def test_array_codegen(self):
        out = codegen("fn f(a: f32[3], b: f32[3]) -> f32[3] { copysign(a, b) }")
        assert "stablehlo.abs" in out
        assert "stablehlo.sign" in out

    def test_type_error_wrong_arg_count(self):
        msg = type_check_fails("fn f(a: f32) -> f32 { copysign(a) }")
        assert "2 arguments" in msg

    def test_type_error_int_args(self):
        msg = type_check_fails("fn f(a: i32, b: i32) -> i32 { copysign(a, b) }")
        assert "float" in msg.lower()

    def test_broadcasting(self):
        out = codegen("fn f(a: f32[3], b: f32) -> f32[3] { copysign(a, b) }")
        assert "stablehlo.broadcast_in_dim" in out

    def test_grad_a(self):
        out = codegen_ad("""
            fn f(a: f32, b: f32) -> f32 {
                grad(copysign(a, b), a)
            }
        """)
        # d/da = sign(b) — involves sign op
        assert "stablehlo.sign" in out

    def test_grad_b_zero(self):
        out = codegen_ad("""
            fn f(a: f32, b: f32) -> f32 {
                grad(copysign(a, b), b)
            }
        """)
        # d/db = 0
        assert "stablehlo" in out


# ============================================================
# Cross-cutting tests
# ============================================================

class TestCrossCutting:
    """Tests that apply to all new two-arg builtins."""

    def test_all_in_single_fn(self):
        """All four builtins can be used in the same function."""
        out = codegen("""
            fn f(a: f32, b: f32) -> f32 {
                let la = logaddexp(a, b);
                let h = hypot(a, b);
                let r = remainder(a, b);
                let c = copysign(a, b);
                la + h + r + c
            }
        """)
        assert "stablehlo.maximum" in out  # logaddexp
        assert "stablehlo.sqrt" in out     # hypot
        assert "stablehlo.floor" in out    # remainder
        assert "stablehlo.sign" in out     # copysign

    def test_shape_match_2d(self):
        """2D arrays work correctly."""
        out = codegen("fn f(a: f32[3,4], b: f32[3,4]) -> f32[3,4] { logaddexp(a, b) }")
        assert "tensor<3x4xf32>" in out

    def test_shape_mismatch_error(self):
        """Mismatched shapes (non-broadcastable) produce a type error."""
        msg = type_check_fails("fn f(a: f32[3], b: f32[4]) -> f32[3] { logaddexp(a, b) }")
        assert "shape" in msg.lower() or "broadcast" in msg.lower() or "mismatch" in msg.lower()
