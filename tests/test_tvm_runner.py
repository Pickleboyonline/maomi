"""Tests for the TVM Relax execution backend."""

import numpy as np
import pytest

tvm = pytest.importorskip("tvm")

from maomi.cli import compile_source_relax
from maomi.types import ScalarType, ArrayType
from maomi.type_checker import FnSignature
from maomi.tvm_runner import run_relax
from maomi.runner_utils import generate_inputs


class TestRunRelaxCPU:
    def test_scalar_add(self):
        source = "fn add(a: f32, b: f32) -> f32 { a + b }"
        result = compile_source_relax(source)
        sig = result.fn_table["add"]
        inputs, output = run_relax(result.ir_mod, "add", sig, seed=42)
        expected = inputs[0] + inputs[1]
        assert np.isclose(output, expected)

    def test_scalar_multiply(self):
        source = "fn mul(a: f32, b: f32) -> f32 { a * b }"
        result = compile_source_relax(source)
        sig = result.fn_table["mul"]
        inputs, output = run_relax(result.ir_mod, "mul", sig, seed=42)
        assert np.isclose(output, inputs[0] * inputs[1])

    def test_scalar_subtract(self):
        source = "fn sub(a: f32, b: f32) -> f32 { a - b }"
        result = compile_source_relax(source)
        sig = result.fn_table["sub"]
        inputs, output = run_relax(result.ir_mod, "sub", sig, seed=42)
        assert np.isclose(output, inputs[0] - inputs[1])

    def test_scalar_divide(self):
        source = "fn div(a: f32, b: f32) -> f32 { a / b }"
        result = compile_source_relax(source)
        sig = result.fn_table["div"]
        inputs, output = run_relax(result.ir_mod, "div", sig, seed=42)
        assert np.isclose(output, inputs[0] / inputs[1])

    def test_unary_negate(self):
        source = "fn neg(x: f32) -> f32 { -x }"
        result = compile_source_relax(source)
        sig = result.fn_table["neg"]
        inputs, output = run_relax(result.ir_mod, "neg", sig, seed=42)
        assert np.isclose(output, -inputs[0])

    def test_array_add(self):
        source = "fn add(a: f32[4, 3], b: f32[4, 3]) -> f32[4, 3] { a + b }"
        result = compile_source_relax(source)
        sig = result.fn_table["add"]
        inputs, output = run_relax(result.ir_mod, "add", sig, seed=42)
        expected = inputs[0] + inputs[1]
        assert np.allclose(output, expected, atol=1e-6)

    def test_matmul(self):
        source = "fn f(a: f32[4, 8], b: f32[8, 3]) -> f32[4, 3] { a @ b }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        inputs, output = run_relax(result.ir_mod, "f", sig, seed=42)
        expected = inputs[0] @ inputs[1]
        assert np.allclose(output, expected, atol=1e-5)

    def test_linear_layer(self):
        source = """
fn linear(x: f32[4, 8], w: f32[8, 3], b: f32[3]) -> f32[4, 3] {
    x @ w + b
}
"""
        result = compile_source_relax(source)
        sig = result.fn_table["linear"]
        inputs, output = run_relax(result.ir_mod, "linear", sig, seed=42)
        x, w, b = inputs
        expected = x @ w + b
        assert np.allclose(output, expected, atol=1e-5)

    def test_let_binding(self):
        source = "fn f(a: f32, b: f32) -> f32 { let x = a + b; x * x }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        inputs, output = run_relax(result.ir_mod, "f", sig, seed=42)
        x = inputs[0] + inputs[1]
        assert np.isclose(output, x * x)

    def test_deterministic_with_seed(self):
        source = "fn add(a: f32, b: f32) -> f32 { a + b }"
        result = compile_source_relax(source)
        sig = result.fn_table["add"]
        _, out1 = run_relax(result.ir_mod, "add", sig, seed=99)
        _, out2 = run_relax(result.ir_mod, "add", sig, seed=99)
        assert out1 == out2

    def test_custom_inputs(self):
        source = "fn add(a: f32, b: f32) -> f32 { a + b }"
        result = compile_source_relax(source)
        sig = result.fn_table["add"]
        a = np.float32(3.0)
        b = np.float32(4.0)
        inputs, output = run_relax(result.ir_mod, "add", sig, inputs=[a, b])
        assert np.isclose(output, 7.0)


class TestRelaxBuiltins:
    def test_exp(self):
        source = "fn f(x: f32) -> f32 { exp(x) }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[np.float32(1.0)])
        assert np.isclose(output, np.exp(1.0), atol=1e-5)

    def test_log(self):
        source = "fn f(x: f32) -> f32 { log(x) }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[np.float32(2.718)])
        assert np.isclose(output, np.log(2.718), atol=1e-3)

    def test_tanh(self):
        source = "fn f(x: f32) -> f32 { tanh(x) }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[np.float32(0.5)])
        assert np.isclose(output, np.tanh(0.5), atol=1e-5)

    def test_sqrt(self):
        source = "fn f(x: f32) -> f32 { sqrt(x) }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[np.float32(4.0)])
        assert np.isclose(output, 2.0, atol=1e-5)

    def test_abs(self):
        source = "fn f(x: f32) -> f32 { abs(x) }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[np.float32(-3.0)])
        assert np.isclose(output, 3.0, atol=1e-5)


class TestRelaxComparisons:
    def test_greater_than(self):
        source = "fn f(a: f32, b: f32) -> bool { a > b }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[np.float32(3.0), np.float32(2.0)])
        assert output == True

    def test_less_than(self):
        source = "fn f(a: f32, b: f32) -> bool { a < b }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[np.float32(1.0), np.float32(2.0)])
        assert output == True


class TestRelaxIfExpr:
    def test_relu_positive(self):
        source = "fn f(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[np.float32(3.0)])
        assert np.isclose(output, 3.0)

    def test_relu_negative(self):
        source = "fn f(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[np.float32(-2.0)])
        assert np.isclose(output, 0.0)


class TestRelaxReductions:
    def test_sum(self):
        source = "fn f(x: f32[5]) -> f32 { sum(x) }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[x])
        assert np.isclose(output, 15.0, atol=1e-5)

    def test_mean(self):
        source = "fn f(x: f32[4]) -> f32 { mean(x) }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        x = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[x])
        assert np.isclose(output, 5.0, atol=1e-5)

class TestRelaxTransposeReshape:
    def test_transpose(self):
        source = "fn f(x: f32[3, 4]) -> f32[4, 3] { transpose(x) }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[x])
        assert np.allclose(output, x.T)

    def test_reshape(self):
        source = "fn f(x: f32[3, 4]) -> f32[12] { reshape(x, 12) }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[x])
        assert np.allclose(output, x.reshape(12))


class TestRelaxConcat:
    def test_concat_axis0(self):
        source = "fn f(a: f32[3, 4], b: f32[3, 4]) -> f32[6, 4] { concat(a, b) }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        a = np.ones((3, 4), dtype=np.float32)
        b = np.zeros((3, 4), dtype=np.float32)
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[a, b])
        expected = np.concatenate([a, b], axis=0)
        assert np.allclose(output, expected)

    def test_concat_axis1(self):
        source = "fn f(a: f32[3, 4], b: f32[3, 4]) -> f32[3, 8] { concat(a, b, 1) }"
        result = compile_source_relax(source)
        sig = result.fn_table["f"]
        a = np.ones((3, 4), dtype=np.float32)
        b = np.zeros((3, 4), dtype=np.float32)
        inputs, output = run_relax(result.ir_mod, "f", sig, inputs=[a, b])
        expected = np.concatenate([a, b], axis=1)
        assert np.allclose(output, expected)


class TestRelaxUserFnCall:
    def test_user_fn_call(self):
        source = """
fn add(a: f32, b: f32) -> f32 { a + b }
fn main(x: f32, y: f32) -> f32 { add(x, y) }
"""
        result = compile_source_relax(source)
        sig = result.fn_table["main"]
        inputs, output = run_relax(result.ir_mod, "main", sig,
                                   inputs=[np.float32(3.0), np.float32(4.0)])
        assert np.isclose(output, 7.0)


class TestCompileSourceRelax:
    def test_returns_ir_mod_and_fn_table(self):
        result = compile_source_relax("fn f(x: f32) -> f32 { x }")
        assert result.ir_mod is not None
        assert "f" in result.fn_table
        assert result.fn_table["f"].param_names == ["x"]

    def test_error_on_invalid_source(self):
        from maomi.errors import MaomiError
        with pytest.raises(MaomiError):
            compile_source_relax("fn f(x: f32) -> i32 { x }")
