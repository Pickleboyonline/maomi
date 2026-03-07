"""Tests for the JAX/XLA execution backend."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maomi.cli import compile_source
from maomi.types import ScalarType, ArrayType
from maomi.type_checker import FnSignature

# Skip all JAX tests if not installed
jax = pytest.importorskip("jax")

from maomi.jax_runner import run_stablehlo, generate_inputs, _prepare_module


class TestPrepareModule:
    def test_adds_sym_name(self):
        text = "module {\n  func.func @foo() -> tensor<f32> {\n  }\n}"
        result = _prepare_module(text, "foo")
        assert "module @main {" in result

    def test_renames_function(self):
        text = "module {\n  func.func @add(%a: tensor<f32>) -> tensor<f32> {\n  }\n}"
        result = _prepare_module(text, "add")
        assert "@main(" in result
        assert "@add" not in result

    def test_preserves_other_functions(self):
        text = (
            "module {\n"
            "  func.func @helper() -> tensor<f32> { }\n"
            "  func.func @target() -> tensor<f32> { }\n"
            "}"
        )
        result = _prepare_module(text, "target")
        assert "@helper" in result
        assert "func.func @main()" in result

    def test_main_noop(self):
        text = "module {\n  func.func @main() -> tensor<f32> { }\n}"
        result = _prepare_module(text, "main")
        assert "module @main {" in result
        assert "func.func @main()" in result


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
        sig_pos = FnSignature(["x"], [ScalarType("f32")], ScalarType("f32"))

        # Positive input → gradient 1.0
        _, out_pos = run_stablehlo(result.mlir_text, "grad_relu", sig_pos, seed=0)
        # seed=0 gives a positive input; check gradient is 1.0 or 0.0 based on sign
        inp_pos = generate_inputs(sig_pos, seed=0)[0]
        expected = 1.0 if float(inp_pos) > 0 else 0.0
        assert float(out_pos) == expected


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


class TestConv2dRunner:
    def test_conv2d_forward(self):
        """conv2d forward pass: compare with numpy conv2d."""
        source = """
fn f(x: f32[1, 1, 4, 4], w: f32[1, 1, 3, 3]) -> f32[1, 1, 2, 2] {
    conv2d(x, w)
}
"""
        result = compile_source(source)
        sig = result.fn_table["f"]

        # Create deterministic inputs
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        w = np.ones((1, 1, 3, 3), dtype=np.float32)

        _, output = run_stablehlo(result.mlir_text, "f", sig, inputs=[x, w])

        # Manual: each output = sum of 3x3 window
        # (0,0): sum of x[0,0,0:3,0:3] = 0+1+2+4+5+6+8+9+10 = 45
        # (0,1): 1+2+3+5+6+7+9+10+11 = 54
        # (1,0): 4+5+6+8+9+10+12+13+14 = 81
        # (1,1): 5+6+7+9+10+11+13+14+15 = 90
        expected = np.array([[[[45., 54.], [81., 90.]]]], dtype=np.float32)
        assert np.allclose(output, expected, atol=1e-4)


class TestMaxPoolRunner:
    def test_max_pool_forward(self):
        """max_pool forward: pick max from each 2x2 window."""
        source = """
fn f(x: f32[1, 1, 4, 4]) -> f32[1, 1, 2, 2] {
    max_pool(x, 2, 2, 2, 2)
}
"""
        result = compile_source(source)
        sig = result.fn_table["f"]
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        _, output = run_stablehlo(result.mlir_text, "f", sig, inputs=[x])
        # max of each 2x2: [5, 7, 13, 15]
        expected = np.array([[[[5., 7.], [13., 15.]]]], dtype=np.float32)
        assert np.allclose(output, expected, atol=1e-4)


class TestAvgPoolRunner:
    def test_avg_pool_forward(self):
        """avg_pool forward: average of each 2x2 window."""
        source = """
fn f(x: f32[1, 1, 4, 4]) -> f32[1, 1, 2, 2] {
    avg_pool(x, 2, 2, 2, 2)
}
"""
        result = compile_source(source)
        sig = result.fn_table["f"]
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        _, output = run_stablehlo(result.mlir_text, "f", sig, inputs=[x])
        # avg of each 2x2: [(0+1+4+5)/4, (2+3+6+7)/4, (8+9+12+13)/4, (10+11+14+15)/4]
        expected = np.array([[[[2.5, 4.5], [10.5, 12.5]]]], dtype=np.float32)
        assert np.allclose(output, expected, atol=1e-4)


class TestConv2dGradRunner:
    def test_conv2d_grad_wrt_input(self):
        """grad of sum(reshape(conv2d(x,w))) w.r.t. x, compare with JAX."""
        source = """
fn f(x: f32[1, 1, 4, 4], w: f32[1, 1, 3, 3]) -> f32[1, 1, 4, 4] {
    let y = conv2d(x, w);
    let flat = reshape(y, 4);
    grad(sum(flat), x)
}
"""
        result = compile_source(source)
        sig = result.fn_table["f"]
        x = np.ones((1, 1, 4, 4), dtype=np.float32)
        w = np.ones((1, 1, 3, 3), dtype=np.float32)
        _, output = run_stablehlo(result.mlir_text, "f", sig, inputs=[x, w])

        # Compare with JAX
        from jax import lax, grad as jax_grad
        import jax.numpy as jnp

        def jax_fn(x):
            y = lax.conv_general_dilated(x, jnp.array(w), (1, 1), 'VALID',
                                          dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
            return jnp.sum(y)

        expected = jax_grad(jax_fn)(jnp.array(x))
        assert np.allclose(output, np.array(expected), atol=1e-5), f"got {output}, expected {expected}"

    def test_conv2d_grad_wrt_kernel(self):
        """grad of sum(reshape(conv2d(x,w))) w.r.t. w, compare with JAX."""
        source = """
fn f(x: f32[1, 1, 4, 4], w: f32[1, 1, 3, 3]) -> f32[1, 1, 3, 3] {
    let y = conv2d(x, w);
    let flat = reshape(y, 4);
    grad(sum(flat), w)
}
"""
        result = compile_source(source)
        sig = result.fn_table["f"]
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        w = np.ones((1, 1, 3, 3), dtype=np.float32)
        _, output = run_stablehlo(result.mlir_text, "f", sig, inputs=[x, w])

        from jax import lax, grad as jax_grad
        import jax.numpy as jnp

        def jax_fn(w):
            y = lax.conv_general_dilated(jnp.array(x), w, (1, 1), 'VALID',
                                          dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
            return jnp.sum(y)

        expected = jax_grad(jax_fn)(jnp.array(w))
        assert np.allclose(output, np.array(expected), atol=1e-5), f"got {output}, expected {expected}"


class TestAvgPoolGradRunner:
    def test_avg_pool_grad(self):
        """grad of sum(reshape(avg_pool(x))) w.r.t. x, compare with JAX."""
        source = """
fn f(x: f32[1, 1, 4, 4]) -> f32[1, 1, 4, 4] {
    let y = avg_pool(x, 2, 2, 2, 2);
    let flat = reshape(y, 4);
    grad(sum(flat), x)
}
"""
        result = compile_source(source)
        sig = result.fn_table["f"]
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        _, output = run_stablehlo(result.mlir_text, "f", sig, inputs=[x])

        from jax import lax, grad as jax_grad
        import jax.numpy as jnp

        def jax_fn(x):
            y = lax.reduce_window(x, 0.0, lax.add, (1,1,2,2), (1,1,2,2), 'VALID')
            y = y / 4.0  # avg
            return jnp.sum(y)

        expected = jax_grad(jax_fn)(jnp.array(x))
        assert np.allclose(output, np.array(expected), atol=1e-5), f"got {output}, expected {expected}"


class TestMaxPoolGradRunner:
    def test_max_pool_grad(self):
        """grad of sum(reshape(max_pool(x))) w.r.t. x, compare with JAX."""
        source = """
fn f(x: f32[1, 1, 4, 4]) -> f32[1, 1, 4, 4] {
    let y = max_pool(x, 2, 2, 2, 2);
    let flat = reshape(y, 4);
    grad(sum(flat), x)
}
"""
        result = compile_source(source)
        sig = result.fn_table["f"]
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        _, output = run_stablehlo(result.mlir_text, "f", sig, inputs=[x])

        from jax import lax, grad as jax_grad
        import jax.numpy as jnp

        def jax_fn(x):
            y = lax.reduce_window(x, -jnp.inf, lax.max, (1,1,2,2), (1,1,2,2), 'VALID')
            return jnp.sum(y)

        expected = jax_grad(jax_fn)(jnp.array(x))
        assert np.allclose(output, np.array(expected), atol=1e-5), f"got {output}, expected {expected}"


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

    def test_callback_count(self):
        result = compile_source("fn f(x: f32) -> f32 { callback(x); x }")
        assert result.callback_count == 1

    def test_callback_count_multiple(self):
        result = compile_source("fn f(x: f32) -> f32 { callback(x); callback(x); x }")
        assert result.callback_count == 2

    def test_callback_count_zero(self):
        result = compile_source("fn f(x: f32) -> f32 { x }")
        assert result.callback_count == 0


class TestCallback:
    def test_callback_fires(self):
        """callback should invoke Python function during execution."""
        result = compile_source("fn f(x: f32) -> f32 { callback(x); x + 1.0 }")
        sig = result.fn_table["f"]
        captured = []
        def cb(*args):
            captured.append([np.asarray(a) for a in args])
            return ()
        _, output = run_stablehlo(result.mlir_text, "f", sig, seed=42, host_callbacks=[cb])
        assert len(captured) == 1

    def test_callback_receives_correct_value(self):
        result = compile_source("fn f(x: f32) -> f32 { callback(x); x }")
        sig = result.fn_table["f"]
        captured = []
        def cb(*args):
            captured.append([np.asarray(a) for a in args])
            return ()
        inputs, _ = run_stablehlo(result.mlir_text, "f", sig, seed=42, host_callbacks=[cb])
        assert np.isclose(captured[0][0], inputs[0])

    def test_callback_zero_args(self):
        result = compile_source("fn f(x: f32) -> f32 { callback(); x }")
        sig = result.fn_table["f"]
        called = [False]
        def cb(*args):
            called[0] = True
            return ()
        run_stablehlo(result.mlir_text, "f", sig, seed=42, host_callbacks=[cb])
        assert called[0]

    def test_callback_multiple(self):
        result = compile_source("fn f(x: f32, y: f32) -> f32 { callback(x); callback(y); x + y }")
        sig = result.fn_table["f"]
        captured = []
        cbs = []
        for _ in range(result.callback_count):
            def make_cb():
                def cb(*args):
                    captured.append([np.asarray(a) for a in args])
                    return ()
                return cb
            cbs.append(make_cb())
        _, output = run_stablehlo(result.mlir_text, "f", sig, seed=42, host_callbacks=cbs)
        assert len(captured) == 2

    def test_callback_array_arg(self):
        result = compile_source("fn f(x: f32[4]) -> f32[4] { callback(x); x }")
        sig = result.fn_table["f"]
        captured = []
        def cb(*args):
            captured.append([np.asarray(a) for a in args])
            return ()
        inputs, _ = run_stablehlo(result.mlir_text, "f", sig, seed=42, host_callbacks=[cb])
        assert captured[0][0].shape == (4,)
        assert np.allclose(captured[0][0], inputs[0])

    def test_callback_multi_args(self):
        """callback with multiple args should pass all of them."""
        result = compile_source("fn f(x: f32, y: f32) -> f32 { callback(x, y); x + y }")
        sig = result.fn_table["f"]
        captured = []
        def cb(*args):
            captured.append([np.asarray(a) for a in args])
            return ()
        inputs, _ = run_stablehlo(result.mlir_text, "f", sig, seed=42, host_callbacks=[cb])
        assert len(captured[0]) == 2
        assert np.isclose(captured[0][0], inputs[0])
        assert np.isclose(captured[0][1], inputs[1])


class TestGradOfGradRunner:
    """Numerical verification of grad-of-grad through indexing, scan, and broadcast."""

    def test_grad_grad_index_numerical(self):
        """d/dx sum(d/dx(x[0]^2)) for x=[3,0,0] should give [2,0,0]."""
        source = """
fn f(x: f32[3]) -> f32[3] {
    grad(sum(grad(x[0] * x[0], x)), x)
}
"""
        result = compile_source(source)
        sig = result.fn_table["f"]
        x = np.array([3.0, 0.0, 0.0], dtype=np.float32)
        _, output = run_stablehlo(result.mlir_text, "f", sig, inputs=[x])
        # d/dx(x[0]^2) = [2*x[0], 0, 0], sum = 2*x[0]
        # d/dx(2*x[0]) = [2, 0, 0]
        expected = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(output, expected, atol=1e-5)

    def test_grad_grad_sum_x_squared(self):
        """d/dx sum(d/dx(sum(x*x))) = [2,2,2] (constant Hessian)."""
        source = """
fn f(x: f32[3]) -> f32[3] {
    grad(sum(grad(sum(x * x), x)), x)
}
"""
        result = compile_source(source)
        sig = result.fn_table["f"]
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        _, output = run_stablehlo(result.mlir_text, "f", sig, inputs=[x])
        # d/dx(sum(x*x)) = 2*x, sum(2*x) = 2*sum(x)
        # d/dx(2*sum(x)) = [2, 2, 2]
        expected = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        assert np.allclose(output, expected, atol=1e-5)

    def test_grad_grad_scan_init(self):
        """d/dinit(d/dinit(sum(cumsum(init, xs)))) for linear scan."""
        source = """
fn f(init: f32, x: f32[5]) -> f32 {
    grad(grad(sum(scan (carry, elem) in (init, x) { carry + elem }), init), init)
}
"""
        result = compile_source(source)
        sig = result.fn_table["f"]
        inputs = [np.float32(1.0), np.ones(5, dtype=np.float32)]
        _, output = run_stablehlo(result.mlir_text, "f", sig, inputs=inputs)
        # For carry + elem (linear), d(carry_t)/d(init) = 1 for all t
        # sum of carries: sum_t (init + sum(x[:t])) → d/dinit = 5
        # d^2/dinit^2 = 0 (linear in init)
        expected = np.float32(0.0)
        assert np.allclose(output, expected, atol=1e-5)
