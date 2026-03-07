"""Tests for the TVM Relax code generation backend."""

import pytest

tvm = pytest.importorskip("tvm")

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.codegen_relax import RelaxCodegen


def codegen(source: str):
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    return RelaxCodegen(program, checker.type_map).generate()


def ir_text(source: str) -> str:
    """Generate Relax IR and return its string representation."""
    return str(codegen(source))


class TestScalarArithmetic:
    def test_scalar_add(self):
        out = ir_text("fn f(a: f32, b: f32) -> f32 { a + b }")
        assert "f" in out
        assert "add" in out

    def test_scalar_subtract(self):
        out = ir_text("fn f(a: f32, b: f32) -> f32 { a - b }")
        assert "subtract" in out

    def test_scalar_multiply(self):
        out = ir_text("fn f(a: f32, b: f32) -> f32 { a * b }")
        assert "multiply" in out

    def test_scalar_divide(self):
        out = ir_text("fn f(a: f32, b: f32) -> f32 { a / b }")
        assert "divide" in out

    def test_scalar_power(self):
        out = ir_text("fn f(a: f32, b: f32) -> f32 { a ** b }")
        assert "power" in out

    def test_negate(self):
        out = ir_text("fn f(x: f32) -> f32 { -x }")
        assert "negative" in out


class TestLiterals:
    def test_float_literal(self):
        mod = codegen("fn f() -> f32 { 3.14 }")
        # Should produce a valid IRModule
        assert mod is not None

    def test_int_literal(self):
        mod = codegen("fn f() -> i32 { 42 }")
        assert mod is not None

    def test_bool_literal(self):
        mod = codegen("fn f() -> bool { true }")
        assert mod is not None


class TestMatmul:
    def test_2d_matmul(self):
        out = ir_text("fn f(a: f32[32, 128], b: f32[128, 64]) -> f32[32, 64] { a @ b }")
        assert "matmul" in out

    def test_matmul_small(self):
        out = ir_text("fn f(a: f32[4, 8], b: f32[8, 3]) -> f32[4, 3] { a @ b }")
        assert "matmul" in out


class TestLetBindings:
    def test_let_threading(self):
        out = ir_text("fn f(a: f32, b: f32) -> f32 { let x = a + b; x * x }")
        assert "add" in out
        assert "multiply" in out


class TestMixedExpressions:
    def test_linear(self):
        out = ir_text("""
            fn linear(x: f32[4, 8], w: f32[8, 3], b: f32[3]) -> f32[4, 3] {
                x @ w + b
            }
        """)
        assert "matmul" in out
        assert "add" in out

    def test_nested_arithmetic(self):
        out = ir_text("fn f(a: f32, b: f32, c: f32) -> f32 { (a + b) * c }")
        assert "add" in out
        assert "multiply" in out


class TestIRModuleStructure:
    def test_returns_irmodule(self):
        mod = codegen("fn f(a: f32, b: f32) -> f32 { a + b }")
        assert isinstance(mod, tvm.IRModule)

    def test_function_exists_in_module(self):
        mod = codegen("fn myfn(a: f32, b: f32) -> f32 { a + b }")
        # The function should be accessible
        text = str(mod)
        assert "myfn" in text

    def test_multiple_functions(self):
        mod = codegen("""
            fn add(a: f32, b: f32) -> f32 { a + b }
            fn mul(a: f32, b: f32) -> f32 { a * b }
        """)
        text = str(mod)
        assert "add" in text
        assert "mul" in text
