"""Tests for new elementwise builtins: cbrt, round, trunc."""

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


# ============================================================
# cbrt
# ============================================================

class TestCbrt:
    """Test cbrt builtin."""

    def test_cbrt_codegen_array(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { cbrt(x) }")
        assert "stablehlo.cbrt" in out

    def test_cbrt_codegen_scalar(self):
        out = codegen("fn f(x: f32) -> f32 { cbrt(x) }")
        assert "stablehlo.cbrt" in out
        assert "tensor<f32>" in out

    def test_cbrt_codegen_2d(self):
        out = codegen("fn f(x: f32[3, 4]) -> f32[3, 4] { cbrt(x) }")
        assert "stablehlo.cbrt" in out
        assert "tensor<3x4xf32>" in out

    def test_cbrt_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(cbrt(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        # Gradient uses cbrt(x)^2, so cbrt should appear in the output
        assert "stablehlo.cbrt" in out
        assert out is not None


# ============================================================
# round
# ============================================================

class TestRound:
    """Test round builtin."""

    def test_round_codegen_array(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { round(x) }")
        assert "stablehlo.round_nearest_even" in out

    def test_round_codegen_scalar(self):
        out = codegen("fn f(x: f32) -> f32 { round(x) }")
        assert "stablehlo.round_nearest_even" in out
        assert "tensor<f32>" in out

    def test_round_codegen_2d(self):
        out = codegen("fn f(x: f32[3, 4]) -> f32[3, 4] { round(x) }")
        assert "stablehlo.round_nearest_even" in out
        assert "tensor<3x4xf32>" in out

    def test_round_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(round(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        # Gradient is zero (piecewise constant)
        assert out is not None


# ============================================================
# trunc
# ============================================================

class TestTrunc:
    """Test trunc builtin."""

    def test_trunc_codegen_array(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { trunc(x) }")
        # trunc = sign(x) * floor(abs(x))
        assert "stablehlo.abs" in out
        assert "stablehlo.floor" in out
        assert "stablehlo.sign" in out
        assert "stablehlo.multiply" in out

    def test_trunc_codegen_scalar(self):
        out = codegen("fn f(x: f32) -> f32 { trunc(x) }")
        assert "stablehlo.abs" in out
        assert "stablehlo.floor" in out
        assert "tensor<f32>" in out

    def test_trunc_codegen_2d(self):
        out = codegen("fn f(x: f32[3, 4]) -> f32[3, 4] { trunc(x) }")
        assert "stablehlo.abs" in out
        assert "stablehlo.floor" in out
        assert "tensor<3x4xf32>" in out

    def test_trunc_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(trunc(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        # Gradient is zero (piecewise constant)
        assert out is not None
