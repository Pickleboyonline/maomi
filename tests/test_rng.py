"""Tests for RNG builtins: random.key, random.split, random.uniform, random.normal."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker, KEY_TYPE
from maomi.ad import transform_grad
from maomi.codegen_stablehlo import StableHLOCodegen
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
# Type Checker Tests
# =====================================================================


class TestKeyTypeAlias:
    def test_key_as_param(self):
        check_ok("fn f(k: Key) -> i32[4] { k }")

    def test_key_as_return(self):
        check_ok("fn f(seed: i32) -> Key { random.key(seed) }")

    def test_key_in_let(self):
        check_ok("fn f(seed: i32) -> i32[4] { let k: Key = random.key(seed); k }")

    def test_key_resolves_to_i32_4(self):
        ret = infer_type("fn f(k: Key) -> Key { k }")
        assert ret == KEY_TYPE
        assert ret == ArrayType("i32", (4,))

    def test_key_with_dims_error(self):
        """Key[2] is a parse error — parser only allows dims on base types."""
        from maomi.errors import MaomiError
        with pytest.raises(MaomiError):
            codegen("fn f(k: Key[2]) -> Key { k }")


class TestRngKeyTypeCheck:
    def test_basic(self):
        check_ok("fn f(s: i32) -> i32[4] { random.key(s) }")

    def test_return_type(self):
        ret = infer_type("fn f(s: i32) -> i32[4] { random.key(s) }")
        assert ret == ArrayType("i32", (4,))

    def test_literal_seed(self):
        check_ok("fn f() -> i32[4] { random.key(42) }")

    def test_wrong_seed_type(self):
        check_err("fn f(s: f32) -> i32[4] { random.key(s) }", "i32")

    def test_too_many_args(self):
        check_err("fn f(s: i32) -> i32[4] { random.key(s, s) }", "1 argument")


class TestRngSplitTypeCheck:
    def test_basic(self):
        check_ok("fn f(k: i32[4]) -> i32[3, 4] { random.split(k, 3) }")

    def test_return_type(self):
        ret = infer_type("fn f(k: i32[4]) -> i32[3, 4] { random.split(k, 3) }")
        assert ret == ArrayType("i32", (3, 4))

    def test_split_2(self):
        ret = infer_type("fn f(k: i32[4]) -> i32[2, 4] { random.split(k, 2) }")
        assert ret == ArrayType("i32", (2, 4))

    def test_wrong_key_type(self):
        check_err("fn f(k: f32[4]) -> i32[3, 4] { random.split(k, 3) }", "Key")

    def test_non_literal_count(self):
        check_err("fn f(k: i32[4], n: i32) -> i32[3, 4] { random.split(k, n) }", "integer literal")

    def test_wrong_arg_count(self):
        check_err("fn f(k: i32[4]) -> i32[3, 4] { random.split(k) }", "2 arguments")


class TestRngUniformTypeCheck:
    def test_basic(self):
        check_ok("fn f(k: i32[4]) -> f32[4, 4] { random.uniform(k, 0.0, 1.0, 4, 4) }")

    def test_return_type_2d(self):
        ret = infer_type("fn f(k: i32[4]) -> f32[4, 4] { random.uniform(k, 0.0, 1.0, 4, 4) }")
        assert ret == ArrayType("f32", (4, 4))

    def test_return_type_1d(self):
        ret = infer_type("fn f(k: i32[4]) -> f32[8] { random.uniform(k, 0.0, 1.0, 8) }")
        assert ret == ArrayType("f32", (8,))

    def test_return_type_3d(self):
        ret = infer_type("fn f(k: i32[4]) -> f32[2, 3, 4] { random.uniform(k, 0.0, 1.0, 2, 3, 4) }")
        assert ret == ArrayType("f32", (2, 3, 4))

    def test_wrong_key_type(self):
        check_err("fn f(k: f32[4]) -> f32[4] { random.uniform(k, 0.0, 1.0, 4) }", "Key")

    def test_wrong_param_type(self):
        check_err("fn f(k: i32[4]) -> f32[4] { random.uniform(k, 0, 1.0, 4) }", "f32")

    def test_missing_dims(self):
        check_err("fn f(k: i32[4]) -> f32[4] { random.uniform(k, 0.0, 1.0) }", "at least 4")

    def test_non_literal_dim(self):
        check_err("fn f(k: i32[4], n: i32) -> f32[4] { random.uniform(k, 0.0, 1.0, n) }", "integer literal")

    def test_expression_params(self):
        """low/high can be any f32 expression, not just literals."""
        check_ok("""
            fn f(k: i32[4], lo: f32, hi: f32) -> f32[4] {
                random.uniform(k, lo, hi, 4)
            }
        """)


class TestRngNormalTypeCheck:
    def test_basic(self):
        check_ok("fn f(k: i32[4]) -> f32[8] { random.normal(k, 0.0, 1.0, 8) }")

    def test_return_type(self):
        ret = infer_type("fn f(k: i32[4]) -> f32[4, 4] { random.normal(k, 0.0, 1.0, 4, 4) }")
        assert ret == ArrayType("f32", (4, 4))

    def test_expression_params(self):
        check_ok("""
            fn f(k: i32[4], mu: f32, sigma: f32) -> f32[4] {
                random.normal(k, mu, sigma, 4)
            }
        """)


# =====================================================================
# Codegen Tests
# =====================================================================


class TestRngKeyCodegen:
    def test_emits_concatenate(self):
        out = codegen("fn f(s: i32) -> i32[4] { random.key(s) }")
        assert "stablehlo.concatenate" in out
        assert "tensor<4xi32>" in out

    def test_emits_zeros(self):
        out = codegen("fn f(s: i32) -> i32[4] { random.key(s) }")
        assert "dense<0> : tensor<3xi32>" in out


class TestRngSplitCodegen:
    def test_emits_rng_bit_generator(self):
        out = codegen("fn f(k: i32[4]) -> i32[3, 4] { random.split(k, 3) }")
        assert "stablehlo.rng_bit_generator" in out
        assert "algorithm = DEFAULT" in out

    def test_emits_bitcast(self):
        out = codegen("fn f(k: i32[4]) -> i32[3, 4] { random.split(k, 3) }")
        assert "stablehlo.bitcast_convert" in out


class TestRngUniformCodegen:
    def test_emits_rng_bit_generator(self):
        out = codegen("fn f(k: i32[4]) -> f32[4, 4] { random.uniform(k, 0.0, 1.0, 4, 4) }")
        assert "stablehlo.rng_bit_generator" in out

    def test_emits_bits_to_float(self):
        out = codegen("fn f(k: i32[4]) -> f32[4, 4] { random.uniform(k, 0.0, 1.0, 4, 4) }")
        assert "shift_right_logical" in out
        assert "1065353216" in out  # 0x3F800000 = 1.0f bit pattern
        assert "stablehlo.bitcast_convert" in out


class TestRngNormalCodegen:
    def test_emits_rng_bit_generator(self):
        out = codegen("fn f(k: i32[4]) -> f32[8] { random.normal(k, 0.0, 1.0, 8) }")
        assert "stablehlo.rng_bit_generator" in out

    def test_emits_box_muller(self):
        out = codegen("fn f(k: i32[4]) -> f32[8] { random.normal(k, 0.0, 1.0, 8) }")
        assert "stablehlo.cosine" in out
        assert "stablehlo.log" in out
        assert "stablehlo.sqrt" in out

    def test_2d_emits_reshape(self):
        out = codegen("fn f(k: i32[4]) -> f32[4, 4] { random.normal(k, 0.0, 1.0, 4, 4) }")
        assert "tensor<4x4xf32>" in out


# =====================================================================
# AD Tests
# =====================================================================


class TestRngAD:
    def test_rng_uniform_in_grad(self):
        """rng_uniform inside grad: values used but zero gradient through RNG."""
        out = ad_codegen("""
            fn f(k: i32[4], x: f32[4]) -> f32[4] {
                let mask = random.uniform(k, 0.0, 1.0, 4);
                grad(sum(mask * x), x)
            }
        """)
        # Should compile without errors
        assert "stablehlo.rng_bit_generator" in out

    def test_rng_normal_in_grad(self):
        """rng_normal inside grad compiles fine."""
        out = ad_codegen("""
            fn f(k: i32[4], x: f32[4]) -> f32[4] {
                let noise = random.normal(k, 0.0, 0.01, 4);
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


class TestRngRunner:
    def test_rng_key_deterministic(self):
        src = "fn f(seed: i32) -> i32[4] { random.key(seed) }"
        result = compile_source(src)
        _, out1 = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        _, out2 = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        np.testing.assert_array_equal(out1, out2)

    def test_rng_split_shape(self):
        src = "fn f(k: i32[4]) -> i32[3, 4] { random.split(k, 3) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.shape == (3, 4)
        assert out.dtype == np.int32

    def test_rng_uniform_range(self):
        src = "fn f(k: i32[4]) -> f32[1000] { random.uniform(k, 0.0, 1.0, 1000) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.shape == (1000,)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() < 1.0

    def test_rng_uniform_custom_range(self):
        src = "fn f(k: i32[4]) -> f32[1000] { random.uniform(k, -1.0, 1.0, 1000) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.min() >= -1.0
        assert out.max() < 1.0

    def test_rng_uniform_deterministic(self):
        src = "fn f(k: i32[4]) -> f32[4, 4] { random.uniform(k, 0.0, 1.0, 4, 4) }"
        result = compile_source(src)
        _, out1 = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        _, out2 = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        np.testing.assert_array_equal(out1, out2)

    def test_rng_uniform_different_seed(self):
        src = "fn f(k: i32[4]) -> f32[4, 4] { random.uniform(k, 0.0, 1.0, 4, 4) }"
        result = compile_source(src)
        _, out1 = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        _, out2 = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=99)
        assert not np.array_equal(out1, out2)

    def test_rng_normal_stats(self):
        src = "fn f(k: i32[4]) -> f32[10000] { random.normal(k, 0.0, 1.0, 10000) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.shape == (10000,)
        assert abs(out.mean()) < 0.1  # mean ~0
        assert abs(np.std(out) - 1.0) < 0.15  # std ~1

    def test_rng_normal_custom_params(self):
        src = "fn f(k: i32[4]) -> f32[10000] { random.normal(k, 5.0, 0.1, 10000) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert abs(out.mean() - 5.0) < 0.05
        assert abs(np.std(out) - 0.1) < 0.02

    def test_rng_normal_2d(self):
        src = "fn f(k: i32[4]) -> f32[4, 4] { random.normal(k, 0.0, 1.0, 4, 4) }"
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.shape == (4, 4)

    def test_full_pipeline_split_and_index(self):
        src = """
            fn f(seed: i32) -> f32[4, 4] {
                let key = random.key(seed);
                let keys = random.split(key, 2);
                random.uniform(keys[0], 0.0, 1.0, 4, 4)
            }
        """
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.shape == (4, 4)
        assert out.min() >= 0.0
        assert out.max() < 1.0

    def test_key_type_alias_runs(self):
        src = """
            fn f(key: Key) -> f32[4, 4] {
                random.normal(key, 0.0, 1.0, 4, 4)
            }
        """
        result = compile_source(src)
        _, out = run_stablehlo(result.mlir_text, "f", result.fn_table["f"], seed=42)
        assert out.shape == (4, 4)
        assert out.dtype == np.float32
