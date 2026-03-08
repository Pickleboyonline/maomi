"""Tests for einsum builtin: type checking, codegen, and AD."""

import pytest
import numpy as np
from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.codegen.stablehlo import StableHLOCodegen
from maomi.ad import transform_grad
from maomi.errors import MaomiError


def check(source: str) -> list[str]:
    """Returns list of error messages (empty = no errors)."""
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    errors = TypeChecker().check(program)
    return [e.message for e in errors]


def check_ok(source: str):
    """Assert that the source type-checks without errors."""
    errors = check(source)
    assert errors == [], f"Expected no errors, got: {errors}"


def check_err(source: str, *fragments: str):
    """Assert that the source has errors containing the given fragments."""
    errors = check(source)
    assert len(errors) > 0, "Expected errors, got none"
    for frag in fragments:
        assert any(frag in e for e in errors), f"Expected error containing {frag!r}, got: {errors}"


def codegen(source: str, *, run_ad: bool = False) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    if run_ad:
        program = transform_grad(program, checker.type_map)
    return StableHLOCodegen(program, checker.type_map).generate()


# ---------------------------------------------------------------------------
# Type Checking
# ---------------------------------------------------------------------------

class TestEinsumTypeChecking:
    def test_matmul(self):
        """einsum("ij,jk->ik", a, b) = matrix multiply"""
        check_ok(
            'fn f(a: f32[3, 4], b: f32[4, 5]) -> f32[3, 5] { einsum("ij,jk->ik", a, b) }'
        )

    def test_transpose(self):
        """einsum("ij->ji", a) = transpose"""
        check_ok(
            'fn f(a: f32[3, 4]) -> f32[4, 3] { einsum("ij->ji", a) }'
        )

    def test_sum_over_axis(self):
        """einsum("ij->i", a) = sum over j"""
        check_ok(
            'fn f(a: f32[3, 4]) -> f32[3] { einsum("ij->i", a) }'
        )

    def test_element_wise_multiply(self):
        """einsum("ij,ij->ij", a, b) = element-wise multiply"""
        check_ok(
            'fn f(a: f32[3, 4], b: f32[3, 4]) -> f32[3, 4] { einsum("ij,ij->ij", a, b) }'
        )

    def test_batched_matmul(self):
        """einsum("bij,bjk->bik", a, b) = batched matmul"""
        check_ok(
            'fn f(a: f32[2, 3, 4], b: f32[2, 4, 5]) -> f32[2, 3, 5] { einsum("bij,bjk->bik", a, b) }'
        )

    def test_outer_product(self):
        """einsum("i,j->ij", a, b) = outer product"""
        check_ok(
            'fn f(a: f32[3], b: f32[4]) -> f32[3, 4] { einsum("i,j->ij", a, b) }'
        )

    def test_dot_product(self):
        """einsum("i,i->", a, b) = dot product (scalar result)"""
        check_ok(
            'fn f(a: f32[3], b: f32[3]) -> f32 { einsum("i,i->", a, b) }'
        )

    def test_trace(self):
        """einsum("ii->", a) = trace (sum of diagonal)"""
        check_ok(
            'fn f(a: f32[3, 3]) -> f32 { einsum("ii->", a) }'
        )

    # Error cases

    def test_missing_arrow(self):
        check_err(
            'fn f(a: f32[3, 4]) -> f32[3] { einsum("ij", a) }',
            "must contain '->'",
        )

    def test_wrong_arg_count(self):
        check_err(
            'fn f(a: f32[3, 4], b: f32[4, 5]) -> f32[3, 5] { einsum("ij,jk,kl->il", a, b) }',
            "3 input(s) but got 2 array argument(s)",
        )

    def test_rank_mismatch(self):
        check_err(
            'fn f(a: f32[3, 4]) -> f32[3] { einsum("ijk->i", a) }',
            "subscript \"ijk\" has 3 indices but argument 1 has rank 2",
        )

    def test_dim_mismatch(self):
        check_err(
            'fn f(a: f32[3, 4], b: f32[5, 6]) -> f32[3, 6] { einsum("ij,jk->ik", a, b) }',
            "dimension mismatch for index 'j'",
        )

    def test_output_char_not_in_inputs(self):
        check_err(
            'fn f(a: f32[3, 4]) -> f32[3] { einsum("ij->k", a) }',
            "output index 'k' does not appear in any input subscript",
        )

    def test_spec_not_string(self):
        check_err(
            'fn f(a: f32[3, 4]) -> f32[3] { einsum(42, a) }',
            "string literal spec",
        )

    def test_too_few_args(self):
        check_err(
            'fn f() -> f32 { einsum("i->") }',
            "at least 2 arguments",
        )

    def test_base_type_mismatch(self):
        check_err(
            'fn f(a: f32[3, 4], b: f64[4, 5]) -> f32[3, 5] { einsum("ij,jk->ik", a, b) }',
            "same base type",
        )


# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------

class TestEinsumCodegen:
    def test_matmul_dot_general(self):
        """2-input einsum lowers to stablehlo.dot_general"""
        out = codegen(
            'fn f(a: f32[3, 4], b: f32[4, 5]) -> f32[3, 5] { einsum("ij,jk->ik", a, b) }'
        )
        assert "stablehlo.dot_general" in out
        assert "contracting_dims" in out

    def test_transpose_lowering(self):
        """1-input transpose einsum lowers to stablehlo.transpose"""
        out = codegen(
            'fn f(a: f32[3, 4]) -> f32[4, 3] { einsum("ij->ji", a) }'
        )
        assert "stablehlo.transpose" in out

    def test_sum_over_axis(self):
        """1-input sum einsum lowers to stablehlo.reduce"""
        out = codegen(
            'fn f(a: f32[3, 4]) -> f32[3] { einsum("ij->i", a) }'
        )
        assert "stablehlo.reduce" in out

    def test_batched_matmul(self):
        """Batched matmul uses batching_dims"""
        out = codegen(
            'fn f(a: f32[2, 3, 4], b: f32[2, 4, 5]) -> f32[2, 3, 5] { einsum("bij,bjk->bik", a, b) }'
        )
        assert "stablehlo.dot_general" in out
        assert "batching_dims" in out

    def test_element_wise(self):
        """Element-wise einsum (no contracting dims)"""
        out = codegen(
            'fn f(a: f32[3, 4], b: f32[3, 4]) -> f32[3, 4] { einsum("ij,ij->ij", a, b) }'
        )
        assert "stablehlo.dot_general" in out

    def test_outer_product(self):
        """Outer product: no contracting dims, no batch dims"""
        out = codegen(
            'fn f(a: f32[3], b: f32[4]) -> f32[3, 4] { einsum("i,j->ij", a, b) }'
        )
        assert "stablehlo.dot_general" in out


# ---------------------------------------------------------------------------
# AD (gradient)
# ---------------------------------------------------------------------------

class TestEinsumAD:
    def test_matmul_grad_compiles(self):
        """Gradient of einsum matmul compiles without error."""
        out = codegen(
            'fn f(a: f32[3, 4], b: f32[4, 5]) -> f32 { sum(einsum("ij,jk->ik", a, b)) }\n'
            'fn grad_a(a: f32[3, 4], b: f32[4, 5]) -> f32[3, 4] { grad(f(a, b), a) }\n'
            'fn grad_b(a: f32[3, 4], b: f32[4, 5]) -> f32[4, 5] { grad(f(a, b), b) }',
            run_ad=True,
        )
        assert "stablehlo.dot_general" in out

    def test_transpose_grad_compiles(self):
        """Gradient through transpose einsum compiles."""
        out = codegen(
            'fn f(a: f32[3, 4]) -> f32 { sum(einsum("ij->ji", a)) }\n'
            'fn grad_a(a: f32[3, 4]) -> f32[3, 4] { grad(f(a), a) }',
            run_ad=True,
        )
        assert "func.func @grad_a" in out

    def test_sum_axis_grad_compiles(self):
        """Gradient through sum-over-axis einsum compiles."""
        out = codegen(
            'fn f(a: f32[3, 4]) -> f32 { sum(einsum("ij->i", a)) }\n'
            'fn grad_a(a: f32[3, 4]) -> f32[3, 4] { grad(f(a), a) }',
            run_ad=True,
        )
        assert "func.func @grad_a" in out


# ---------------------------------------------------------------------------
# Numerical AD Verification (requires JAX)
# ---------------------------------------------------------------------------

jax = pytest.importorskip("jax")
jnp = jax.numpy


class TestEinsumNumericalAD:
    def _check(self, maomi_src, maomi_inputs, jax_fn, jax_inputs, *, atol=1e-5):
        import maomi as maomi_mod
        mod = maomi_mod.compile(maomi_src)
        maomi_grad = mod.grad_f(*maomi_inputs)
        jax_grad = jax_fn(*jax_inputs)
        np.testing.assert_allclose(
            np.asarray(maomi_grad), np.asarray(jax_grad), atol=atol,
            err_msg="Gradient mismatch"
        )

    def test_matmul_grad_a(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        self._check(
            'fn f(a: f32[3, 4], b: f32[4, 5]) -> f32 { sum(einsum("ij,jk->ik", a, b)) }\n'
            'fn grad_f(a: f32[3, 4], b: f32[4, 5]) -> f32[3, 4] { grad(f(a, b), a) }',
            [a, b],
            jax.grad(lambda a, b: jnp.sum(jnp.einsum("ij,jk->ik", a, b))),
            [jnp.array(a), jnp.array(b)],
        )

    def test_matmul_grad_b(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        self._check(
            'fn f(a: f32[3, 4], b: f32[4, 5]) -> f32 { sum(einsum("ij,jk->ik", a, b)) }\n'
            'fn grad_f(a: f32[3, 4], b: f32[4, 5]) -> f32[4, 5] { grad(f(a, b), b) }',
            [a, b],
            jax.grad(lambda a, b: jnp.sum(jnp.einsum("ij,jk->ik", a, b)), argnums=1),
            [jnp.array(a), jnp.array(b)],
        )

    def test_batched_matmul_grad(self):
        a = np.random.randn(2, 3, 4).astype(np.float32)
        b = np.random.randn(2, 4, 5).astype(np.float32)
        self._check(
            'fn f(a: f32[2, 3, 4], b: f32[2, 4, 5]) -> f32 { sum(einsum("bij,bjk->bik", a, b)) }\n'
            'fn grad_f(a: f32[2, 3, 4], b: f32[2, 4, 5]) -> f32[2, 3, 4] { grad(f(a, b), a) }',
            [a, b],
            jax.grad(lambda a, b: jnp.sum(jnp.einsum("bij,bjk->bik", a, b))),
            [jnp.array(a), jnp.array(b)],
        )

    def test_transpose_grad(self):
        a = np.random.randn(3, 4).astype(np.float32)
        self._check(
            'fn f(a: f32[3, 4]) -> f32 { sum(einsum("ij->ji", a)) }\n'
            'fn grad_f(a: f32[3, 4]) -> f32[3, 4] { grad(f(a), a) }',
            [a],
            jax.grad(lambda a: jnp.sum(jnp.einsum("ij->ji", a))),
            [jnp.array(a)],
        )

    def test_sum_axis_grad(self):
        a = np.random.randn(3, 4).astype(np.float32)
        self._check(
            'fn f(a: f32[3, 4]) -> f32 { sum(einsum("ij->i", a)) }\n'
            'fn grad_f(a: f32[3, 4]) -> f32[3, 4] { grad(f(a), a) }',
            [a],
            jax.grad(lambda a: jnp.sum(jnp.einsum("ij->i", a))),
            [jnp.array(a)],
        )

    def test_outer_product_grad(self):
        a = np.random.randn(3).astype(np.float32)
        b = np.random.randn(4).astype(np.float32)
        self._check(
            'fn f(a: f32[3], b: f32[4]) -> f32 { sum(einsum("i,j->ij", a, b)) }\n'
            'fn grad_f(a: f32[3], b: f32[4]) -> f32[3] { grad(f(a, b), a) }',
            [a, b],
            jax.grad(lambda a, b: jnp.sum(jnp.einsum("i,j->ij", a, b))),
            [jnp.array(a), jnp.array(b)],
        )
