"""Tests for the TVM Relax code generation backend."""

import pytest

tvm = pytest.importorskip("tvm")

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.codegen.relax import RelaxCodegen


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


# ---- Phase 2 tests ----


class TestComparisons:
    def test_greater_than(self):
        out = ir_text("fn f(a: f32, b: f32) -> bool { a > b }")
        assert "greater" in out

    def test_less_than(self):
        out = ir_text("fn f(a: f32, b: f32) -> bool { a < b }")
        assert "less" in out

    def test_equal(self):
        out = ir_text("fn f(a: f32, b: f32) -> bool { a == b }")
        assert "equal" in out

    def test_not_equal(self):
        out = ir_text("fn f(a: f32, b: f32) -> bool { a != b }")
        assert "not_equal" in out


class TestIfExpr:
    def test_select(self):
        out = ir_text("fn f(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }")
        assert "where" in out

    def test_relu(self):
        mod = codegen("fn f(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }")
        assert mod is not None


class TestElementwiseBuiltins:
    def test_exp(self):
        out = ir_text("fn f(x: f32) -> f32 { exp(x) }")
        assert "exp" in out

    def test_log(self):
        out = ir_text("fn f(x: f32) -> f32 { log(x) }")
        assert "log" in out

    def test_tanh(self):
        out = ir_text("fn f(x: f32) -> f32 { tanh(x) }")
        assert "tanh" in out

    def test_sqrt(self):
        out = ir_text("fn f(x: f32) -> f32 { sqrt(x) }")
        assert "sqrt" in out

    def test_abs(self):
        out = ir_text("fn f(x: f32) -> f32 { abs(x) }")
        assert "abs" in out


class TestReductions:
    def test_sum(self):
        out = ir_text("fn f(x: f32[10]) -> f32 { sum(x) }")
        assert "sum" in out

    def test_mean(self):
        out = ir_text("fn f(x: f32[10]) -> f32 { mean(x) }")
        assert "mean" in out


class TestTransposeReshape:
    def test_transpose(self):
        mod = codegen("fn f(x: f32[3, 4]) -> f32[4, 3] { transpose(x) }")
        out = str(mod)
        assert "permute_dims" in out

    def test_reshape(self):
        mod = codegen("fn f(x: f32[3, 4]) -> f32[12] { reshape(x, 12) }")
        out = str(mod)
        assert "reshape" in out


class TestConcat:
    def test_concat_default_axis(self):
        out = ir_text("fn f(a: f32[3, 4], b: f32[3, 4]) -> f32[6, 4] { concat(a, b) }")
        assert "concat" in out.lower() or "concatenate" in out.lower()

    def test_concat_with_axis(self):
        out = ir_text("fn f(a: f32[3, 4], b: f32[3, 4]) -> f32[3, 8] { concat(a, b, 1) }")
        assert "concat" in out.lower() or "concatenate" in out.lower()


class TestStructCodegen:
    def test_struct_literal(self):
        mod = codegen("""
            struct Point { x: f32, y: f32 }
            fn f() -> Point { Point { x: 1.0, y: 2.0 } }
        """)
        assert mod is not None

    def test_field_access(self):
        out = ir_text("""
            struct Point { x: f32, y: f32 }
            fn f(p: Point) -> f32 { p.x }
        """)
        # TVM renders TupleGetItem as p[0] in TVMScript
        assert "p[0]" in out

    def test_with_update(self):
        mod = codegen("""
            struct Point { x: f32, y: f32 }
            fn f(p: Point) -> Point { p with { x = 1.0 } }
        """)
        assert mod is not None


class TestUserFnCall:
    def test_user_fn_call(self):
        mod = codegen("""
            fn add(a: f32, b: f32) -> f32 { a + b }
            fn main(x: f32, y: f32) -> f32 { add(x, y) }
        """)
        text = str(mod)
        assert "add" in text
        assert "main" in text
