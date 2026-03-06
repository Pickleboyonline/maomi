"""Tests for the IREE execution backend."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maomi.cli import compile_source
from maomi.types import ScalarType, ArrayType
from maomi.type_checker import FnSignature

# Skip all IREE tests if not installed
pytest.importorskip("iree.compiler")

from maomi.iree_runner import run_stablehlo, generate_inputs, _prepare_module


class TestPrepareModule:
    def test_adds_sym_name(self):
        text = "module {\n  func.func @foo() -> tensor<f32> {\n  }\n}"
        result = _prepare_module(text)
        assert "module @main {" in result
        assert "@foo" in result  # function name preserved

    def test_preserves_all_functions(self):
        text = (
            "module {\n"
            "  func.func @helper() -> tensor<f32> { }\n"
            "  func.func @target() -> tensor<f32> { }\n"
            "}"
        )
        result = _prepare_module(text)
        assert "@helper" in result
        assert "@target" in result


class TestGenerateInputs:
    def test_scalar_f32(self):
        sig = FnSignature(["x"], [ScalarType("f32")], ScalarType("f32"))
        inputs = generate_inputs(sig, seed=42)
        assert len(inputs) == 1
        assert inputs[0].dtype == np.float32
        assert inputs[0].shape == ()

    def test_array_f32(self):
        sig = FnSignature(["x"], [ArrayType("f32", (3, 4))], ScalarType("f32"))
        inputs = generate_inputs(sig, seed=42)
        assert inputs[0].shape == (3, 4)
        assert inputs[0].dtype == np.float32

    def test_i32(self):
        sig = FnSignature(["x"], [ScalarType("i32")], ScalarType("i32"))
        inputs = generate_inputs(sig, seed=42)
        assert inputs[0].dtype == np.int32

    def test_bool(self):
        sig = FnSignature(["x"], [ScalarType("bool")], ScalarType("bool"))
        inputs = generate_inputs(sig, seed=42)
        assert inputs[0].dtype == np.bool_

    def test_deterministic(self):
        sig = FnSignature(["a", "b"], [ScalarType("f32"), ScalarType("f32")], ScalarType("f32"))
        inputs1 = generate_inputs(sig, seed=42)
        inputs2 = generate_inputs(sig, seed=42)
        assert inputs1[0] == inputs2[0]
        assert inputs1[1] == inputs2[1]

    def test_different_seeds(self):
        sig = FnSignature(["x"], [ArrayType("f32", (10,))], ScalarType("f32"))
        inputs1 = generate_inputs(sig, seed=1)
        inputs2 = generate_inputs(sig, seed=2)
        assert not np.array_equal(inputs1[0], inputs2[0])


class TestRunStableHLO:
    def test_scalar_add(self):
        source = "fn add(a: f32, b: f32) -> f32 { a + b }"
        result = compile_source(source)
        sig = result.fn_table["add"]
        inputs, output = run_stablehlo(result.mlir_text, "add", sig, seed=42)
        expected = inputs[0] + inputs[1]
        assert np.isclose(output, expected)

    def test_linear_layer(self):
        source = """
fn linear(x: f32[4, 8], w: f32[8, 3], b: f32[3]) -> f32[4, 3] {
    x @ w + b
}
"""
        result = compile_source(source)
        sig = result.fn_table["linear"]
        inputs, output = run_stablehlo(result.mlir_text, "linear", sig, seed=42)
        x, w, b = inputs
        expected = x @ w + b
        assert np.allclose(output, expected, atol=1e-5)

    def test_deterministic_with_seed(self):
        source = "fn add(a: f32, b: f32) -> f32 { a + b }"
        result = compile_source(source)
        sig = result.fn_table["add"]
        _, out1 = run_stablehlo(result.mlir_text, "add", sig, seed=99)
        _, out2 = run_stablehlo(result.mlir_text, "add", sig, seed=99)
        assert out1 == out2

    def test_multiply(self):
        source = "fn mul(a: f32, b: f32) -> f32 { a * b }"
        result = compile_source(source)
        sig = result.fn_table["mul"]
        inputs, output = run_stablehlo(result.mlir_text, "mul", sig, seed=42)
        assert np.isclose(output, inputs[0] * inputs[1])

    def test_unary_negate(self):
        source = "fn neg(x: f32) -> f32 { -x }"
        result = compile_source(source)
        sig = result.fn_table["neg"]
        inputs, output = run_stablehlo(result.mlir_text, "neg", sig, seed=42)
        assert np.isclose(output, -inputs[0])


    def test_relu_gradient(self):
        """End-to-end: ReLU gradient through function call + if/else AD."""
        source = """
fn relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
fn grad_relu(x: f32) -> f32 { grad(relu(x), x) }
"""
        result = compile_source(source)
        sig = result.fn_table["grad_relu"]

        # Positive input → gradient 1.0
        from maomi.iree_runner import run_stablehlo as run_fn
        from iree.compiler import compile_str
        from iree import runtime as ireert

        mlir_text = result.mlir_text.replace("module {", "module @main {", 1)
        compiled = compile_str(mlir_text, target_backends=["llvm-cpu"], input_type="stablehlo")
        config = ireert.Config("local-task")
        ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled)
        ctx.add_vm_module(vm_module)

        f = ctx.modules.main["grad_relu"]
        assert f(np.float32(5.0)).to_host() == 1.0
        assert f(np.float32(-3.0)).to_host() == 0.0


class TestScanGrad:
    def test_cumulative_sum_gradient(self):
        """grad(sum(scan(acc+x)), xs) should be [5, 4, 3, 2, 1] for xs: f32[5]."""
        source = """
fn f(xs: f32[5]) -> f32[5] {
    let s = scan (acc, x) in (0.0, xs) { acc + x };
    grad(sum(s), xs)
}
"""
        result = compile_source(source)
        sig = result.fn_table["f"]
        inputs, output = run_stablehlo(result.mlir_text, "f", sig, seed=42)
        expected = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        assert np.allclose(output, expected, atol=1e-5)


class TestCompileSource:
    def test_returns_mlir_and_fn_table(self):
        result = compile_source("fn f(x: f32) -> f32 { x }")
        assert "func.func" in result.mlir_text
        assert "f" in result.fn_table
        assert result.fn_table["f"].param_names == ["x"]

    def test_error_on_invalid_source(self):
        from maomi.errors import MaomiError
        with pytest.raises(MaomiError):
            compile_source("fn f(x: f32) -> i32 { x }")
