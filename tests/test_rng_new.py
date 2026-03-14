"""Tests for new RNG builtins: random.exponential, random.randint."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker, KEY_TYPE
from maomi.ad import transform_grad
from maomi.codegen.stablehlo import StableHLOCodegen
from maomi.cli import compile_source
from maomi.types import ArrayType, ScalarType


# -- Helpers --

def check(source: str) -> list[str]:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    errors = TypeChecker().check(program)
    return [e.message for e in errors]


def check_ok(source: str):
    errors = check(source)
    assert errors == [], f"Expected no errors, got: {errors}"


def check_err(source: str, *fragments: str):
    errors = check(source)
    assert len(errors) > 0, "Expected errors, got none"
    for frag in fragments:
        assert any(frag in e for e in errors), f"Expected error containing {frag!r}, got: {errors}"


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


def infer_type(source: str, fn_name: str = None):
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    checker.check(program)
    if fn_name is None:
        fn_name = program.functions[0].name
    sig = checker.fn_table.get(fn_name)
    return sig.return_type if sig else None


# =====================================================================
# Type Checker Tests — random.exponential
# =====================================================================


class TestExponentialTypeCheck:
    def test_basic(self):
        check_ok("fn f(k: i32[4]) -> f32[10] { random.exponential(k, 10) }")

    def test_2d(self):
        check_ok("fn f(k: i32[4]) -> f32[3, 4] { random.exponential(k, 3, 4) }")

    def test_3d(self):
        check_ok("fn f(k: i32[4]) -> f32[2, 3, 4] { random.exponential(k, 2, 3, 4) }")

    def test_return_type(self):
        ret = infer_type("fn f(k: i32[4]) -> f32[10] { random.exponential(k, 10) }")
        assert ret == ArrayType("f32", (10,))

    def test_return_type_2d(self):
        ret = infer_type("fn f(k: i32[4]) -> f32[3, 4] { random.exponential(k, 3, 4) }")
        assert ret == ArrayType("f32", (3, 4))

    def test_key_alias(self):
        check_ok("fn f(k: Key) -> f32[10] { random.exponential(k, 10) }")

    def test_wrong_key_type(self):
        check_err("fn f(k: f32[4]) -> f32[10] { random.exponential(k, 10) }", "Key")

    def test_missing_dims(self):
        check_err("fn f(k: i32[4]) -> f32[10] { random.exponential(k) }", "at least 2")

    def test_non_literal_dim(self):
        check_err("fn f(k: i32[4], n: i32) -> f32[10] { random.exponential(k, n) }", "integer literal")

    def test_wrong_return_type(self):
        """random.exponential returns f32, not i32."""
        check_err("fn f(k: i32[4]) -> i32[10] { random.exponential(k, 10) }", "f32")


# =====================================================================
# Type Checker Tests — random.randint
# =====================================================================


class TestRandintTypeCheck:
    def test_basic(self):
        check_ok("fn f(k: i32[4]) -> i32[10] { random.randint(k, 0, 10, 10) }")

    def test_2d(self):
        check_ok("fn f(k: i32[4]) -> i32[3, 4] { random.randint(k, 1, 100, 3, 4) }")

    def test_3d(self):
        check_ok("fn f(k: i32[4]) -> i32[2, 3, 4] { random.randint(k, 0, 5, 2, 3, 4) }")

    def test_return_type(self):
        ret = infer_type("fn f(k: i32[4]) -> i32[10] { random.randint(k, 0, 10, 10) }")
        assert ret == ArrayType("i32", (10,))

    def test_return_type_2d(self):
        ret = infer_type("fn f(k: i32[4]) -> i32[3, 4] { random.randint(k, 0, 10, 3, 4) }")
        assert ret == ArrayType("i32", (3, 4))

    def test_key_alias(self):
        check_ok("fn f(k: Key) -> i32[10] { random.randint(k, 0, 10, 10) }")

    def test_wrong_key_type(self):
        check_err("fn f(k: f32[4]) -> i32[10] { random.randint(k, 0, 10, 10) }", "Key")

    def test_wrong_low_type(self):
        check_err("fn f(k: i32[4]) -> i32[10] { random.randint(k, 0.0, 10, 10) }", "i32")

    def test_wrong_high_type(self):
        check_err("fn f(k: i32[4]) -> i32[10] { random.randint(k, 0, 10.0, 10) }", "i32")

    def test_missing_dims(self):
        check_err("fn f(k: i32[4]) -> i32[10] { random.randint(k, 0, 10) }", "at least 4")

    def test_non_literal_dim(self):
        check_err("fn f(k: i32[4], n: i32) -> i32[10] { random.randint(k, 0, 10, n) }", "integer literal")


# =====================================================================
# Codegen Tests — random.exponential
# =====================================================================


class TestExponentialCodegen:
    def test_emits_rng_bit_generator(self):
        out = codegen("fn f(k: i32[4]) -> f32[10] { random.exponential(k, 10) }")
        assert "stablehlo.rng_bit_generator" in out

    def test_emits_log_and_negate(self):
        out = codegen("fn f(k: i32[4]) -> f32[10] { random.exponential(k, 10) }")
        assert "stablehlo.log" in out
        assert "stablehlo.negate" in out

    def test_output_shape(self):
        out = codegen("fn f(k: i32[4]) -> f32[10] { random.exponential(k, 10) }")
        assert "tensor<10xf32>" in out

    def test_2d_output_shape(self):
        out = codegen("fn f(k: i32[4]) -> f32[3, 4] { random.exponential(k, 3, 4) }")
        assert "tensor<3x4xf32>" in out


# =====================================================================
# Codegen Tests — random.randint
# =====================================================================


class TestRandintCodegen:
    def test_emits_rng_bit_generator(self):
        out = codegen("fn f(k: i32[4]) -> i32[10] { random.randint(k, 0, 10, 10) }")
        assert "stablehlo.rng_bit_generator" in out

    def test_emits_floor(self):
        out = codegen("fn f(k: i32[4]) -> i32[10] { random.randint(k, 0, 10, 10) }")
        assert "stablehlo.floor" in out

    def test_emits_convert(self):
        """Should cast back to i32."""
        out = codegen("fn f(k: i32[4]) -> i32[10] { random.randint(k, 0, 10, 10) }")
        assert "stablehlo.convert" in out

    def test_output_shape(self):
        out = codegen("fn f(k: i32[4]) -> i32[10] { random.randint(k, 0, 10, 10) }")
        assert "tensor<10xi32>" in out

    def test_2d_output_shape(self):
        out = codegen("fn f(k: i32[4]) -> i32[3, 4] { random.randint(k, 1, 100, 3, 4) }")
        assert "tensor<3x4xi32>" in out


# =====================================================================
# AD Tests
# =====================================================================


class TestNewRngAD:
    def test_exponential_in_grad(self):
        """random.exponential inside grad: zero gradient through RNG."""
        out = ad_codegen("""
            fn f(k: i32[4], x: f32[4]) -> f32[4] {
                let noise = random.exponential(k, 4);
                grad(sum(x + noise), x)
            }
        """)
        assert "stablehlo.rng_bit_generator" in out


# =====================================================================
# End-to-end Runner Tests (require JAX)
# =====================================================================


jax = pytest.importorskip("jax")
np = pytest.importorskip("numpy")

from maomi.jax_runner import run_stablehlo


class TestExponentialRunner:
    def test_shape(self):
        src = "fn f(k: i32[4]) -> f32[1000] { random.exponential(k, 1000) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.shape == (1000,)
        assert out.dtype == np.float32

    def test_non_negative(self):
        """Exponential distribution values should be >= 0."""
        src = "fn f(k: i32[4]) -> f32[10000] { random.exponential(k, 10000) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.min() >= 0.0

    def test_mean_approx_1(self):
        """Exponential(rate=1) has mean=1."""
        src = "fn f(k: i32[4]) -> f32[10000] { random.exponential(k, 10000) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert abs(out.mean() - 1.0) < 0.1

    def test_2d_shape(self):
        src = "fn f(k: i32[4]) -> f32[3, 4] { random.exponential(k, 3, 4) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.shape == (3, 4)

    def test_deterministic(self):
        src = "fn f(k: i32[4]) -> f32[10] { random.exponential(k, 10) }"
        result = compile_source(src)
        _, out1 = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        _, out2 = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        np.testing.assert_array_equal(out1, out2)


class TestRandintRunner:
    def test_shape(self):
        src = "fn f(k: i32[4]) -> i32[1000] { random.randint(k, 0, 10, 1000) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.shape == (1000,)
        assert out.dtype == np.int32

    def test_range(self):
        """Values should be in [0, 10)."""
        src = "fn f(k: i32[4]) -> i32[10000] { random.randint(k, 0, 10, 10000) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.min() >= 0
        assert out.max() < 10

    def test_range_offset(self):
        """Values should be in [5, 15)."""
        src = "fn f(k: i32[4]) -> i32[10000] { random.randint(k, 5, 15, 10000) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.min() >= 5
        assert out.max() < 15

    def test_2d_shape(self):
        src = "fn f(k: i32[4]) -> i32[3, 4] { random.randint(k, 0, 100, 3, 4) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.shape == (3, 4)

    def test_deterministic(self):
        src = "fn f(k: i32[4]) -> i32[10] { random.randint(k, 0, 100, 10) }"
        result = compile_source(src)
        _, out1 = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        _, out2 = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        np.testing.assert_array_equal(out1, out2)
