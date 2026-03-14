"""Tests for new reduction builtins: prod, all, any."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.codegen.stablehlo import StableHLOCodegen
from maomi.ad import transform_grad
from maomi.cli import compile_source
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
    from maomi.resolver import resolve
    program = resolve(program, "/tmp/test.mao")
    tc = TypeChecker()
    errors = tc.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    program = transform_grad(program, tc.type_map)
    return StableHLOCodegen(program, tc.type_map).generate()


def compile_ok(source: str) -> str:
    result = compile_source(source, filename="/tmp/test.mao")
    return result.mlir_text


def type_check_fails(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert len(errors) > 0, "Expected type error"
    return errors[0].message


# ============================================================
# prod — Product reduction
# ============================================================

class TestProd:
    def test_prod_full_reduction(self):
        """prod(x) reduces all elements to a scalar."""
        out = codegen("fn f(x: f32[4]) -> f32 { prod(x) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.multiply" in out
        # Init value should be 1.0
        assert "dense<1.000000e+00>" in out

    def test_prod_axis_reduction(self):
        """prod(x, axis=1) reduces along axis 1."""
        out = codegen("fn f(x: f32[3, 4]) -> f32[3] { prod(x, axis=1) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.multiply" in out
        assert "across dimensions = [1]" in out

    def test_prod_keepdims(self):
        """prod(x, axis=1, keepdims=true) keeps reduced dim as size 1."""
        out = codegen("fn f(x: f32[3, 4]) -> f32[3, 1] { prod(x, axis=1, keepdims=true) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.multiply" in out
        assert "stablehlo.reshape" in out

    def test_prod_2d_full(self):
        """prod over all elements of a 2D array."""
        out = codegen("fn f(x: f32[3, 4]) -> f32 { prod(x) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.multiply" in out

    def test_prod_axis0(self):
        """prod along axis 0."""
        out = codegen("fn f(x: f32[3, 4]) -> f32[4] { prod(x, axis=0) }")
        assert "stablehlo.reduce" in out
        assert "across dimensions = [0]" in out

    def test_prod_compile_ok(self):
        """Full compilation pipeline for prod."""
        out = compile_ok("fn f(x: f32[4]) -> f32 { prod(x) }")
        assert "stablehlo.reduce" in out

    def test_prod_named_args(self):
        """prod with named arguments."""
        out = codegen("fn f(x: f32[3, 4]) -> f32[3] { prod(x, axis=1) }")
        assert "stablehlo.reduce" in out


class TestProdGrad:
    def test_prod_grad_full(self):
        """Gradient of prod(x) w.r.t. x: adj * prod(x) / x_i."""
        src = """
        fn loss(x: f32[4]) -> f32 { prod(x) }
        fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """
        out = codegen_ad(src)
        # Should contain multiply (for prod forward) and divide (for grad)
        assert "stablehlo.multiply" in out
        assert "stablehlo.divide" in out

    def test_prod_grad_axis(self):
        """Gradient of prod(x, axis=1) w.r.t. x."""
        src = """
        fn loss(x: f32[3, 4]) -> f32 { sum(prod(x, axis=1)) }
        fn main(x: f32[3, 4]) -> f32[3, 4] { grad(loss(x), x) }
        """
        out = codegen_ad(src)
        assert "stablehlo.multiply" in out
        assert "stablehlo.divide" in out


# ============================================================
# all — Logical AND reduction
# ============================================================

class TestAll:
    def test_all_full_reduction(self):
        """all(x) reduces all elements to a bool scalar."""
        out = codegen("fn f(x: bool[4]) -> bool { all(x) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.and" in out
        # Init value should be true (dense<1> : tensor<i1>)
        assert "dense<1> : tensor<i1>" in out

    def test_all_axis_reduction(self):
        """all(x, axis=1) reduces along axis 1."""
        out = codegen("fn f(x: bool[3, 4]) -> bool[3] { all(x, axis=1) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.and" in out
        assert "across dimensions = [1]" in out

    def test_all_axis0(self):
        """all along axis 0."""
        out = codegen("fn f(x: bool[3, 4]) -> bool[4] { all(x, axis=0) }")
        assert "stablehlo.reduce" in out
        assert "across dimensions = [0]" in out

    def test_all_full_2d(self):
        """all over all elements of a 2D bool array."""
        out = codegen("fn f(x: bool[3, 4]) -> bool { all(x) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.and" in out

    def test_all_compile_ok(self):
        """Full compilation pipeline for all."""
        out = compile_ok("fn f(x: bool[4]) -> bool { all(x) }")
        assert "stablehlo.reduce" in out

    def test_all_type_error_float_input(self):
        """all requires bool input, not float."""
        msg = type_check_fails("fn f(x: f32[4]) -> bool { all(x) }")
        assert "bool" in msg.lower()

    def test_all_type_error_int_input(self):
        """all requires bool input, not int."""
        msg = type_check_fails("fn f(x: i32[4]) -> bool { all(x) }")
        assert "bool" in msg.lower()


# ============================================================
# any — Logical OR reduction
# ============================================================

class TestAny:
    def test_any_full_reduction(self):
        """any(x) reduces all elements to a bool scalar."""
        out = codegen("fn f(x: bool[4]) -> bool { any(x) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.or" in out
        # Init value should be false (dense<0> : tensor<i1>)
        assert "dense<0> : tensor<i1>" in out

    def test_any_axis_reduction(self):
        """any(x, axis=1) reduces along axis 1."""
        out = codegen("fn f(x: bool[3, 4]) -> bool[3] { any(x, axis=1) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.or" in out
        assert "across dimensions = [1]" in out

    def test_any_axis0(self):
        """any along axis 0."""
        out = codegen("fn f(x: bool[3, 4]) -> bool[4] { any(x, axis=0) }")
        assert "stablehlo.reduce" in out
        assert "across dimensions = [0]" in out

    def test_any_full_2d(self):
        """any over all elements of a 2D bool array."""
        out = codegen("fn f(x: bool[3, 4]) -> bool { any(x) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.or" in out

    def test_any_compile_ok(self):
        """Full compilation pipeline for any."""
        out = compile_ok("fn f(x: bool[4]) -> bool { any(x) }")
        assert "stablehlo.reduce" in out

    def test_any_type_error_float_input(self):
        """any requires bool input, not float."""
        msg = type_check_fails("fn f(x: f32[4]) -> bool { any(x) }")
        assert "bool" in msg.lower()
