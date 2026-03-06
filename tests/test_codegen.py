from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.codegen_stablehlo import StableHLOCodegen
from maomi.errors import MaomiError
import pytest


def codegen(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    return StableHLOCodegen(program, checker.type_map).generate()


class TestBasicCodegen:
    def test_scalar_add(self):
        out = codegen("fn f(a: f32, b: f32) -> f32 { a + b }")
        assert "stablehlo.add" in out
        assert "tensor<f32>" in out
        assert "module {" in out
        assert "func.func @f" in out

    def test_scalar_subtract(self):
        out = codegen("fn f(a: f32, b: f32) -> f32 { a - b }")
        assert "stablehlo.subtract" in out

    def test_scalar_multiply(self):
        out = codegen("fn f(a: f32, b: f32) -> f32 { a * b }")
        assert "stablehlo.multiply" in out

    def test_scalar_divide(self):
        out = codegen("fn f(a: f32, b: f32) -> f32 { a / b }")
        assert "stablehlo.divide" in out

    def test_scalar_power(self):
        out = codegen("fn f(a: f32, b: f32) -> f32 { a ** b }")
        assert "stablehlo.power" in out

    def test_negate(self):
        out = codegen("fn f(x: f32) -> f32 { -x }")
        assert "stablehlo.negate" in out

    def test_literal_constant(self):
        out = codegen("fn f() -> f32 { 3.14 }")
        assert "stablehlo.constant" in out

    def test_int_literal(self):
        out = codegen("fn f() -> i32 { 42 }")
        assert "stablehlo.constant dense<42>" in out
        assert "tensor<i32>" in out


class TestMatmul:
    def test_2d_matmul(self):
        out = codegen("fn f(a: f32[32, 128], b: f32[128, 64]) -> f32[32, 64] { a @ b }")
        assert "stablehlo.dot_general" in out
        assert "contracting_dims" in out
        assert "tensor<32x128xf32>" in out
        assert "tensor<32x64xf32>" in out


class TestComparisons:
    def test_greater_than(self):
        out = codegen("fn f(a: f32, b: f32) -> bool { a > b }")
        assert "stablehlo.compare" in out
        assert "GT" in out


class TestIfExpr:
    def test_select(self):
        out = codegen("fn f(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }")
        assert "stablehlo.select" in out
        assert "stablehlo.compare" in out


class TestLetBindings:
    def test_let_threading(self):
        out = codegen("fn f(a: f32, b: f32) -> f32 { let x = a + b; x * x }")
        assert "stablehlo.add" in out
        assert "stablehlo.multiply" in out


class TestFunctionCalls:
    def test_user_fn_call(self):
        out = codegen("""
            fn add(a: f32, b: f32) -> f32 { a + b }
            fn main(x: f32, y: f32) -> f32 { add(x, y) }
        """)
        assert "func.call @add" in out

    def test_builtin_exp(self):
        out = codegen("fn f(x: f32) -> f32 { exp(x) }")
        assert "stablehlo.exponential" in out

    def test_builtin_tanh(self):
        out = codegen("fn f(x: f32) -> f32 { tanh(x) }")
        assert "stablehlo.tanh" in out


class TestReductions:
    def test_mean(self):
        out = codegen("fn f(x: f32[10]) -> f32 { mean(x) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.divide" in out

    def test_sum(self):
        out = codegen("fn f(x: f32[10]) -> f32 { sum(x) }")
        assert "stablehlo.reduce" in out


class TestBroadcasting:
    def test_scalar_array_add(self):
        out = codegen("fn f(x: f32[32, 64], b: f32) -> f32[32, 64] { x + b }")
        assert "stablehlo.broadcast_in_dim" in out
        assert "stablehlo.add" in out

    def test_bias_add(self):
        out = codegen("""
            fn f(x: f32[32, 128], w: f32[128, 64], b: f32[64]) -> f32[32, 64] {
                x @ w + b
            }
        """)
        assert "stablehlo.dot_general" in out
        assert "stablehlo.broadcast_in_dim" in out
        assert "stablehlo.add" in out


class TestMap:
    def test_map_relu(self):
        out = codegen("""
            fn f(xs: f32[32, 64]) -> f32[32, 64] {
                map x in xs {
                    if x > 0.0 { x } else { 0.0 }
                }
            }
        """)
        assert "stablehlo.compare" in out
        assert "stablehlo.select" in out


class TestScan:
    def test_scan_accumulate(self):
        out = codegen("""
            fn f(xs: f32[5], init: f32) -> f32[5] {
                scan (acc, x) in (init, xs) {
                    acc + x
                }
            }
        """)
        assert "stablehlo.while" in out
        assert "stablehlo.add" in out

    def test_multi_sequence_scan(self):
        out = codegen("""
            fn f(xs: f32[5], ys: f32[5]) -> f32[5] {
                scan (acc, (x, y)) in (0.0, (xs, ys)) {
                    acc + x * y
                }
            }
        """)
        assert "stablehlo.while" in out
        assert "stablehlo.multiply" in out
        assert "stablehlo.add" in out


class TestSymbolicDimError:
    def test_symbolic_dim_rejected(self):
        with pytest.raises(MaomiError, match="unresolved symbolic dimension"):
            codegen("fn f(x: f32[N]) -> f32 { mean(x) }")


class TestCallbackCodegen:
    def test_callback_compiles(self):
        """callback should compile as a no-op."""
        out = codegen("""
            fn f(x: f32, y: f32) -> f32 {
                callback(x, y);
                x + y
            }
        """)
        assert "callback" in out  # comment
        assert "func.func @f" in out

    def test_callback_no_args_compiles(self):
        out = codegen("""
            fn f(x: f32) -> f32 {
                callback();
                x
            }
        """)
        assert "func.func @f" in out
