"""Numerical gradient checking for Maomi's AD against JAX.

JAX is the ground truth — it has years of battle-testing. Every AD rule in
Maomi gets compared against JAX's AD with tight tolerances. If a test fails,
Maomi's AD has a real bug.

Add a test for every new differentiable operation.
"""

import numpy as np
import pytest

import maomi
from maomi.types import ScalarType, StructType

jax = pytest.importorskip("jax")
jnp = jax.numpy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(maomi_src, maomi_inputs, jax_fn, jax_inputs, *, atol=1e-6):
    """Compare Maomi grad output against JAX grad output."""
    mod = maomi.compile(maomi_src)
    maomi_grad = mod.grad_f(*maomi_inputs)

    jax_grad = jax_fn(*jax_inputs)

    if isinstance(maomi_grad, maomi.MaomiStruct):
        maomi_arrays = _flatten_struct(maomi_grad)
        if isinstance(jax_grad, tuple):
            jax_arrays = [np.asarray(g) for g in jax_grad]
        else:
            jax_arrays = [np.asarray(jax_grad)]
    else:
        maomi_arrays = [np.asarray(maomi_grad)]
        jax_arrays = [np.asarray(jax_grad)]

    assert len(maomi_arrays) == len(jax_arrays), (
        f"Output count mismatch: maomi {len(maomi_arrays)} vs jax {len(jax_arrays)}"
    )
    for i, (m, j) in enumerate(zip(maomi_arrays, jax_arrays)):
        np.testing.assert_allclose(m, j, atol=atol,
            err_msg=f"Gradient mismatch at output {i}")


def _flatten_struct(s):
    result = []
    for fname, _ in s._type.fields:
        v = getattr(s, fname)
        if isinstance(v, maomi.MaomiStruct):
            result.extend(_flatten_struct(v))
        else:
            result.append(np.asarray(v))
    return result


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

class TestArithmeticGrad:
    def test_add(self):
        x = np.array([1., 2., 3., 4.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(x + x) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(x + x)), [jnp.array(x)],
        )

    def test_sub(self):
        x = np.array([1., 2., 3., 4.], dtype=np.float32)
        y = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        _check(
            "fn f(x: f32[4], y: f32[4]) -> f32 { mean(x - y) }\n"
            "fn grad_f(x: f32[4], y: f32[4]) -> f32[4] { grad(f(x, y), x) }",
            [x, y],
            jax.grad(lambda x, y: jnp.mean(x - y)), [jnp.array(x), jnp.array(y)],
        )

    def test_mul(self):
        x = np.array([1., 2., 3., 4.], dtype=np.float32)
        y = np.array([2., 0.5, 1., 3.], dtype=np.float32)
        _check(
            "fn f(x: f32[4], y: f32[4]) -> f32 { mean(x * y) }\n"
            "fn grad_f(x: f32[4], y: f32[4]) -> f32[4] { grad(f(x, y), x) }",
            [x, y],
            jax.grad(lambda x, y: jnp.mean(x * y)), [jnp.array(x), jnp.array(y)],
        )

    def test_div(self):
        x = np.array([1., 2., 3., 4.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(x / x) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(x / x)), [jnp.array(x)],
        )

    def test_power(self):
        x = np.array([0.5, 1., 1.5, 2.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(x ** 3.0) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(x ** 3.0)), [jnp.array(x)],
        )

    def test_negate(self):
        x = np.array([1., 2., 3., 4.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(-x) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(-x)), [jnp.array(x)],
        )


# ---------------------------------------------------------------------------
# Elementwise builtins
# ---------------------------------------------------------------------------

class TestElementwiseGrad:
    def test_exp(self):
        x = np.array([0., 0.5, 1., -0.5], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(exp(x)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(jnp.exp(x))), [jnp.array(x)],
        )

    def test_log(self):
        x = np.array([0.5, 1., 2., 3.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(log(x)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(jnp.log(x))), [jnp.array(x)],
        )

    def test_tanh(self):
        x = np.array([0., 0.5, 1., -1.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(tanh(x)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(jnp.tanh(x))), [jnp.array(x)],
        )

    def test_sqrt(self):
        x = np.array([1., 4., 9., 16.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(sqrt(x)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(jnp.sqrt(x))), [jnp.array(x)],
        )

    def test_abs(self):
        x = np.array([-2., -1., 1., 2.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(abs(x)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(jnp.abs(x))), [jnp.array(x)],
        )

    def test_sigmoid(self):
        x = np.array([0., 0.5, 1., -0.5], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(sigmoid(x)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(jax.nn.sigmoid(x))), [jnp.array(x)],
        )

    def test_neg(self):
        x = np.array([1., -2., 3., -4.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(neg(x)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(-x)), [jnp.array(x)],
        )

    def test_log2(self):
        x = np.array([1., 2., 4., 8.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(log2(x)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(jnp.log2(x))), [jnp.array(x)],
        )

    def test_rsqrt(self):
        x = np.array([1., 4., 9., 16.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(rsqrt(x)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(jax.lax.rsqrt(x))), [jnp.array(x)],
        )

    def test_reciprocal(self):
        x = np.array([1., 2., 4., 8.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(reciprocal(x)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(1.0 / x)), [jnp.array(x)],
        )


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------

class TestMatmulGrad:
    def test_matmul_wrt_left(self):
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 4), dtype=np.float32) * 0.5
        _check(
            "fn f(a: f32[2, 3], b: f32[3, 4]) -> f32 { mean(a @ b) }\n"
            "fn grad_f(a: f32[2, 3], b: f32[3, 4]) -> f32[2, 3] { grad(f(a, b), a) }",
            [a, b],
            jax.grad(lambda a, b: jnp.mean(a @ b)), [jnp.array(a), jnp.array(b)],
        )

    def test_matmul_wrt_right(self):
        a = np.ones((2, 3), dtype=np.float32) * 0.5
        b = np.ones((3, 4), dtype=np.float32)
        _check(
            "fn f(a: f32[2, 3], b: f32[3, 4]) -> f32 { mean(a @ b) }\n"
            "fn grad_f(a: f32[2, 3], b: f32[3, 4]) -> f32[3, 4] { grad(f(a, b), b) }",
            [a, b],
            jax.grad(lambda a, b: jnp.mean(a @ b), argnums=1),
            [jnp.array(a), jnp.array(b)],
        )


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------

class TestBroadcastGrad:
    def test_add_bias(self):
        """f32[2,4] + f32[4] — bias gradient must reduce over batch dim."""
        b = np.array([1., 2., 3., 4.], dtype=np.float32)
        x = np.ones((2, 4), dtype=np.float32)
        _check(
            "fn f(b: f32[4], x: f32[2, 4]) -> f32 { mean(x + b) }\n"
            "fn grad_f(b: f32[4], x: f32[2, 4]) -> f32[4] { grad(f(b, x), b) }",
            [b, x],
            jax.grad(lambda b, x: jnp.mean(x + b)), [jnp.array(b), jnp.array(x)],
        )

    def test_mul_scalar_array(self):
        """f32 * f32[4] — scalar gradient must reduce all dims."""
        s = np.float32(2.0)
        x = np.array([1., 2., 3., 4.], dtype=np.float32)
        _check(
            "fn f(s: f32, x: f32[4]) -> f32 { mean(s * x) }\n"
            "fn grad_f(s: f32, x: f32[4]) -> f32 { grad(f(s, x), s) }",
            [s, x],
            jax.grad(lambda s, x: jnp.mean(s * x)), [jnp.float32(2.0), jnp.array(x)],
        )

    def test_div_by_scalar(self):
        """f32[4] / f32 — scalar denominator gradient must reduce."""
        x = np.array([2., 4., 6., 8.], dtype=np.float32)
        s = np.float32(2.0)
        _check(
            "fn f(x: f32[4], s: f32) -> f32 { mean(x / s) }\n"
            "fn grad_f(x: f32[4], s: f32) -> f32 { grad(f(x, s), s) }",
            [x, s],
            jax.grad(lambda x, s: jnp.mean(x / s), argnums=1),
            [jnp.array(x), jnp.float32(2.0)],
        )

    def test_matmul_plus_bias(self):
        """Full y = x @ w + b — bias gradient must reduce."""
        w = np.ones((3, 4), dtype=np.float32) * 0.5
        b = np.ones(4, dtype=np.float32) * 0.1
        x = np.ones((2, 3), dtype=np.float32)
        _check(
            "fn f(w: f32[3, 4], b: f32[4], x: f32[2, 3]) -> f32 { mean(x @ w + b) }\n"
            "fn grad_f(w: f32[3, 4], b: f32[4], x: f32[2, 3]) -> f32[4] { grad(f(w, b, x), b) }",
            [w, b, x],
            jax.grad(lambda w, b, x: jnp.mean(x @ w + b), argnums=1),
            [jnp.array(w), jnp.array(b), jnp.array(x)],
        )


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

class TestReductionGrad:
    def test_sum(self):
        x = np.array([1., 2., 3., 4.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { sum(x) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.sum(x)), [jnp.array(x)],
        )

    def test_mean(self):
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        _check(
            "fn f(x: f32[2, 3]) -> f32 { mean(x) }\n"
            "fn grad_f(x: f32[2, 3]) -> f32[2, 3] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(x)), [jnp.array(x)],
        )

    def test_sum_axis(self):
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        _check(
            "fn f(x: f32[2, 3]) -> f32 { mean(sum(x, 0)) }\n"
            "fn grad_f(x: f32[2, 3]) -> f32[2, 3] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(jnp.sum(x, axis=0))), [jnp.array(x)],
        )


# ---------------------------------------------------------------------------
# If/else (relu)
# ---------------------------------------------------------------------------

class TestIfElseGrad:
    def test_relu_scalar(self):
        mod = maomi.compile(
            "fn f(x: f32) -> f32 { if x > 0.0 { x * x } else { 0.0 } }\n"
            "fn grad_f(x: f32) -> f32 { grad(f(x), x) }"
        )
        # Positive: d/dx(x^2) = 2x = 6
        assert float(mod.grad_f(np.float32(3.0))) == pytest.approx(6.0, abs=1e-5)
        # Negative: d/dx(0) = 0
        assert float(mod.grad_f(np.float32(-1.0))) == pytest.approx(0.0, abs=1e-5)

    def test_relu_array(self):
        x = np.array([-1., 0.5, -0.5, 2.], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { mean(if x > 0.0 { x } else { 0.0 }) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(jnp.where(x > 0, x, 0.0))), [jnp.array(x)],
        )

    def test_relu_through_function_call(self):
        x = np.array([-0.5, 0.5, 1.5, 2.0], dtype=np.float32)
        _check(
            "fn relu(x: f32[4]) -> f32[4] { if x > 0.0 { x } else { 0.0 } }\n"
            "fn f(x: f32[4]) -> f32 { mean(relu(x * x - 1.0)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(jnp.where(x**2 - 1 > 0, x**2 - 1, 0.0))),
            [jnp.array(x)],
        )


# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------

class TestStructGrad:
    def test_struct_field_grad(self):
        """grad(x^2 + y^2, struct) = struct{2x, 2y}."""
        mod = maomi.compile(
            "struct P { x: f32, y: f32 }\n"
            "fn f(p: P) -> f32 { p.x * p.x + p.y * p.y }\n"
            "fn grad_f(p: P) -> P { grad(f(p), p) }"
        )
        stype = StructType("P", (("x", ScalarType("f32")), ("y", ScalarType("f32"))))
        p = maomi.MaomiStruct("P", stype, x=np.float32(3.0), y=np.float32(4.0))
        g = mod.grad_f(p)
        assert float(g.x) == pytest.approx(6.0, abs=1e-5)
        assert float(g.y) == pytest.approx(8.0, abs=1e-5)

    def test_mlp_layer_grad(self):
        """Linear layer: grad of w and b match JAX."""
        mod = maomi.compile(
            "struct P { w: f32[3, 4], b: f32[4] }\n"
            "fn f(p: P, x: f32[2, 3]) -> f32 { mean(x @ p.w + p.b) }\n"
            "fn grad_f(p: P, x: f32[2, 3]) -> P { grad(f(p, x), p) }"
        )
        rng = np.random.default_rng(0)
        w = rng.standard_normal((3, 4)).astype(np.float32) * 0.1
        b = np.zeros(4, dtype=np.float32)
        x = rng.standard_normal((2, 3)).astype(np.float32)
        p = mod.P(w=w, b=b)
        g = mod.grad_f(p, x)

        def jax_f(w, b, x): return jnp.mean(x @ w + b)
        jgw, jgb = jax.grad(jax_f, argnums=(0, 1))(
            jnp.array(w), jnp.array(b), jnp.array(x))
        np.testing.assert_allclose(g.w, np.array(jgw), atol=1e-6)
        np.testing.assert_allclose(g.b, np.array(jgb), atol=1e-6)


# ---------------------------------------------------------------------------
# Function calls inside grad
# ---------------------------------------------------------------------------

class TestFunctionCallGrad:
    def test_simple_call(self):
        x = np.array([1., 2., 3., 4.], dtype=np.float32)
        _check(
            "fn square(x: f32[4]) -> f32[4] { x * x }\n"
            "fn f(x: f32[4]) -> f32 { mean(square(x)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.mean(x * x)), [jnp.array(x)],
        )

    def test_call_with_struct(self):
        """Function call passing struct through grad."""
        mod = maomi.compile(
            "struct P { w: f32[3, 4], b: f32[4] }\n"
            "fn linear(p: P, x: f32[2, 3]) -> f32[2, 4] { x @ p.w + p.b }\n"
            "fn f(p: P, x: f32[2, 3]) -> f32 { mean(linear(p, x)) }\n"
            "fn grad_f(p: P, x: f32[2, 3]) -> P { grad(f(p, x), p) }"
        )
        rng = np.random.default_rng(1)
        w = rng.standard_normal((3, 4)).astype(np.float32) * 0.1
        b = np.zeros(4, dtype=np.float32)
        x = rng.standard_normal((2, 3)).astype(np.float32)

        g = mod.grad_f(mod.P(w=w, b=b), x)

        def jax_f(w, b, x): return jnp.mean(x @ w + b)
        jgw, jgb = jax.grad(jax_f, argnums=(0, 1))(
            jnp.array(w), jnp.array(b), jnp.array(x))
        np.testing.assert_allclose(g.w, np.array(jgw), atol=1e-6)
        np.testing.assert_allclose(g.b, np.array(jgb), atol=1e-6)

    def test_relu_matmul_chain(self):
        """relu(x @ w + b) through function call — the pattern that was broken."""
        mod = maomi.compile(
            "struct P { w: f32[3, 4], b: f32[4] }\n"
            "fn relu(x: f32[2, 4]) -> f32[2, 4] { if x > 0.0 { x } else { 0.0 } }\n"
            "fn f(p: P, x: f32[2, 3]) -> f32 { mean(relu(x @ p.w + p.b)) }\n"
            "fn grad_f(p: P, x: f32[2, 3]) -> P { grad(f(p, x), p) }"
        )
        rng = np.random.default_rng(2)
        w = rng.standard_normal((3, 4)).astype(np.float32) * 0.5
        b = np.zeros(4, dtype=np.float32)
        x = rng.standard_normal((2, 3)).astype(np.float32)
        g = mod.grad_f(mod.P(w=w, b=b), x)

        def jax_f(w, b, x):
            h = x @ w + b
            return jnp.mean(jnp.where(h > 0, h, 0.0))
        jgw, jgb = jax.grad(jax_f, argnums=(0, 1))(
            jnp.array(w), jnp.array(b), jnp.array(x))
        np.testing.assert_allclose(g.w, np.array(jgw), atol=1e-6)
        np.testing.assert_allclose(g.b, np.array(jgb), atol=1e-6)


# ---------------------------------------------------------------------------
# MLP end-to-end
# ---------------------------------------------------------------------------

class TestMLPGrad:
    def test_mlp_relu_grad_vs_jax(self):
        """Full MLP with relu — compare every gradient against JAX."""
        mod = maomi.compile("""
            struct Params { w1: f32[4, 8], b1: f32[8], w2: f32[8, 1], b2: f32[1] }
            fn relu(x: f32[32, 8]) -> f32[32, 8] { if x > 0.0 { x } else { 0.0 } }
            fn forward(p: Params, x: f32[32, 4]) -> f32[32, 1] {
                let h = relu(x @ p.w1 + p.b1);
                h @ p.w2 + p.b2
            }
            fn f(p: Params, x: f32[32, 4], y: f32[32, 1]) -> f32 {
                let pred = forward(p, x);
                let diff = pred - y;
                mean(diff * diff)
            }
            fn grad_f(p: Params, x: f32[32, 4], y: f32[32, 1]) -> Params {
                grad(f(p, x, y), p)
            }
        """)

        rng = np.random.default_rng(42)
        w1 = rng.standard_normal((4, 8)).astype(np.float32) * 0.1
        b1 = np.zeros(8, dtype=np.float32)
        w2 = rng.standard_normal((8, 1)).astype(np.float32) * 0.1
        b2 = np.zeros(1, dtype=np.float32)
        x = rng.standard_normal((32, 4)).astype(np.float32)
        y = x.sum(axis=1, keepdims=True).astype(np.float32)

        p = mod.Params(w1=w1, b1=b1, w2=w2, b2=b2)
        g = mod.grad_f(p, x, y)

        def jax_loss(w1, b1, w2, b2, x, y):
            h = jnp.where(x @ w1 + b1 > 0, x @ w1 + b1, 0.0)
            pred = h @ w2 + b2
            return jnp.mean((pred - y) ** 2)

        jgw1, jgb1, jgw2, jgb2 = jax.grad(jax_loss, argnums=(0,1,2,3))(
            jnp.array(w1), jnp.array(b1), jnp.array(w2), jnp.array(b2),
            jnp.array(x), jnp.array(y))

        np.testing.assert_allclose(g.w1, np.array(jgw1), atol=1e-6)
        np.testing.assert_allclose(g.b1, np.array(jgb1), atol=1e-6)
        np.testing.assert_allclose(g.w2, np.array(jgw2), atol=1e-6)
        np.testing.assert_allclose(g.b2, np.array(jgb2), atol=1e-6)

    def test_mlp_training_matches_jax(self):
        """Train for 50 steps, final params must match JAX."""
        mod = maomi.compile("""
            struct P { w1: f32[4, 8], b1: f32[8], w2: f32[8, 1], b2: f32[1] }
            fn relu(x: f32[16, 8]) -> f32[16, 8] { if x > 0.0 { x } else { 0.0 } }
            fn fwd(p: P, x: f32[16, 4]) -> f32[16, 1] {
                let h = relu(x @ p.w1 + p.b1); h @ p.w2 + p.b2
            }
            fn loss(p: P, x: f32[16, 4], y: f32[16, 1]) -> f32 {
                let d = fwd(p, x) - y; mean(d * d)
            }
            fn step(p: P, x: f32[16, 4], y: f32[16, 1]) -> P {
                let g = grad(loss(p, x, y), p);
                let lr = 0.01;
                p with { w1=p.w1-lr*g.w1, b1=p.b1-lr*g.b1, w2=p.w2-lr*g.w2, b2=p.b2-lr*g.b2 }
            }
        """)
        rng = np.random.default_rng(7)
        w1 = rng.standard_normal((4, 8)).astype(np.float32) * 0.1
        b1 = np.zeros(8, dtype=np.float32)
        w2 = rng.standard_normal((8, 1)).astype(np.float32) * 0.1
        b2 = np.zeros(1, dtype=np.float32)
        x = rng.standard_normal((16, 4)).astype(np.float32)
        y = x.sum(axis=1, keepdims=True).astype(np.float32)
        lr = 0.01

        # Maomi
        p = mod.P(w1=w1.copy(), b1=b1.copy(), w2=w2.copy(), b2=b2.copy())
        for _ in range(50):
            p = mod.step(p, x, y)

        # JAX
        jw1, jb1, jw2, jb2 = jnp.array(w1), jnp.array(b1), jnp.array(w2), jnp.array(b2)
        jx, jy = jnp.array(x), jnp.array(y)
        def jloss(w1, b1, w2, b2, x, y):
            h = jnp.where(x@w1+b1 > 0, x@w1+b1, 0.0)
            return jnp.mean((h@w2+b2 - y)**2)
        for _ in range(50):
            gw1, gb1, gw2, gb2 = jax.grad(jloss, argnums=(0,1,2,3))(jw1, jb1, jw2, jb2, jx, jy)
            jw1, jb1, jw2, jb2 = jw1-lr*gw1, jb1-lr*gb1, jw2-lr*gw2, jb2-lr*gb2

        np.testing.assert_allclose(p.w1, np.array(jw1), atol=1e-5)
        np.testing.assert_allclose(p.b1, np.array(jb1), atol=1e-5)
        np.testing.assert_allclose(p.w2, np.array(jw2), atol=1e-5)
        np.testing.assert_allclose(p.b2, np.array(jb2), atol=1e-5)


# ---------------------------------------------------------------------------
# Two-Arg Elementwise
# ---------------------------------------------------------------------------

class TestTwoArgElementwiseGrad:
    def test_maximum_wrt_x(self):
        x = np.array([1.0, 3.0, 2.0, 5.0], dtype=np.float32)
        y = np.array([2.0, 1.0, 4.0, 3.0], dtype=np.float32)
        _check(
            "fn f(x: f32[4], y: f32[4]) -> f32 { sum(maximum(x, y)) }\n"
            "fn grad_f(x: f32[4], y: f32[4]) -> f32[4] { grad(f(x, y), x) }",
            [x, y],
            jax.grad(lambda x, y: jnp.sum(jnp.maximum(x, y))),
            [jnp.array(x), jnp.array(y)],
        )

    def test_maximum_wrt_y(self):
        x = np.array([1.0, 3.0, 2.0, 5.0], dtype=np.float32)
        y = np.array([2.0, 1.0, 4.0, 3.0], dtype=np.float32)
        _check(
            "fn f(x: f32[4], y: f32[4]) -> f32 { sum(maximum(x, y)) }\n"
            "fn grad_f(x: f32[4], y: f32[4]) -> f32[4] { grad(f(x, y), y) }",
            [x, y],
            lambda x, y: jax.grad(lambda y, x: jnp.sum(jnp.maximum(x, y)))(y, x),
            [jnp.array(x), jnp.array(y)],
        )

    def test_minimum_wrt_x(self):
        x = np.array([1.0, 3.0, 2.0, 5.0], dtype=np.float32)
        y = np.array([2.0, 1.0, 4.0, 3.0], dtype=np.float32)
        _check(
            "fn f(x: f32[4], y: f32[4]) -> f32 { sum(minimum(x, y)) }\n"
            "fn grad_f(x: f32[4], y: f32[4]) -> f32[4] { grad(f(x, y), x) }",
            [x, y],
            jax.grad(lambda x, y: jnp.sum(jnp.minimum(x, y))),
            [jnp.array(x), jnp.array(y)],
        )

    def test_pow_wrt_x(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        _check(
            "fn f(x: f32[4], y: f32[4]) -> f32 { sum(pow(x, y)) }\n"
            "fn grad_f(x: f32[4], y: f32[4]) -> f32[4] { grad(f(x, y), x) }",
            [x, y],
            jax.grad(lambda x, y: jnp.sum(jnp.power(x, y))),
            [jnp.array(x), jnp.array(y)],
        )

    def test_pow_scalar_exponent(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        _check(
            "fn f(x: f32[4]) -> f32 { sum(pow(x, 2.0)) }\n"
            "fn grad_f(x: f32[4]) -> f32[4] { grad(f(x), x) }",
            [x],
            jax.grad(lambda x: jnp.sum(jnp.power(x, 2.0))),
            [jnp.array(x)],
        )

    def test_pow_wrt_y(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = np.array([2.0, 3.0, 1.0, 0.5], dtype=np.float32)
        _check(
            "fn f(x: f32[4], y: f32[4]) -> f32 { sum(pow(x, y)) }\n"
            "fn grad_f(x: f32[4], y: f32[4]) -> f32[4] { grad(f(x, y), y) }",
            [x, y],
            lambda x, y: jax.grad(lambda y, x: jnp.sum(jnp.power(x, y)))(y, x),
            [jnp.array(x), jnp.array(y)],
        )
