from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.ad import transform_grad
from maomi.codegen_stablehlo import StableHLOCodegen
from maomi.ast_nodes import *
from maomi.errors import MaomiError
import pytest


def ad_transform(source: str) -> Program:
    """Parse, type check, and AD-transform a program."""
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    return transform_grad(program, checker.type_map)


def ad_codegen(source: str) -> str:
    """Parse, type check, AD-transform, and codegen a program."""
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    program = transform_grad(program, checker.type_map)
    return StableHLOCodegen(program, checker.type_map).generate()


def get_body_expr(program: Program) -> Expr:
    """Get the trailing expression of the first function's body."""
    fn = program.functions[0]
    return fn.body.expr


class TestADBasic:
    def test_grad_removed(self):
        """After AD, no GradExpr nodes should remain."""
        prog = ad_transform("fn f(x: f32) -> f32 { grad(x * x, x) }")
        expr = get_body_expr(prog)
        assert not isinstance(expr, GradExpr)

    def test_grad_x_squared(self):
        """d/dx(x*x) = 2*x — result should be an expression involving x."""
        prog = ad_transform("fn f(x: f32) -> f32 { grad(x * x, x) }")
        expr = get_body_expr(prog)
        # Should be some form of x + x or 2*x (both are valid)
        assert isinstance(expr, BinOp)

    def test_grad_unused_var(self):
        """d/dy(x*x) = 0 when y is not in the expression."""
        prog = ad_transform("fn f(x: f32, y: f32) -> f32 { grad(x * x, y) }")
        expr = get_body_expr(prog)
        assert isinstance(expr, FloatLiteral)
        assert expr.value == 0.0

    def test_grad_addition(self):
        """d/dx(x + x) = 2 (adjoint of x is 1+1)."""
        prog = ad_transform("fn f(x: f32) -> f32 { grad(x + x, x) }")
        expr = get_body_expr(prog)
        # Should accumulate two copies of adjoint 1.0
        assert isinstance(expr, BinOp) and expr.op == "+"


class TestADArithmetic:
    def test_subtraction(self):
        """d/dx(x - y) w.r.t. x = 1, w.r.t. y = -1."""
        prog = ad_transform("fn f(x: f32, y: f32) -> f32 { grad(x - y, x) }")
        expr = get_body_expr(prog)
        # adjoint of x in (x - y) is dz = 1.0
        assert isinstance(expr, FloatLiteral) and expr.value == 1.0

    def test_multiplication(self):
        """d/dx(x * y) = y."""
        prog = ad_transform("fn f(x: f32, y: f32) -> f32 { grad(x * y, x) }")
        expr = get_body_expr(prog)
        # Should reference y
        assert isinstance(expr, BinOp) and expr.op == "*"

    def test_power(self):
        """d/dx(x ** 3.0) = 3 * x^2."""
        prog = ad_transform("fn f(x: f32) -> f32 { grad(x ** 3.0, x) }")
        expr = get_body_expr(prog)
        assert isinstance(expr, BinOp)


class TestADBuiltins:
    def test_exp(self):
        """d/dx(exp(x)) = exp(x)."""
        prog = ad_transform("fn f(x: f32) -> f32 { grad(exp(x), x) }")
        expr = get_body_expr(prog)
        assert isinstance(expr, BinOp)

    def test_log(self):
        """d/dx(log(x)) = 1/x."""
        prog = ad_transform("fn f(x: f32) -> f32 { grad(log(x), x) }")
        expr = get_body_expr(prog)
        assert isinstance(expr, BinOp)

    def test_tanh(self):
        """d/dx(tanh(x)) = 1 - tanh(x)^2."""
        prog = ad_transform("fn f(x: f32) -> f32 { grad(tanh(x), x) }")
        expr = get_body_expr(prog)
        assert isinstance(expr, BinOp)


class TestADReductions:
    def test_mean(self):
        """d/dx(mean(x)) = 1/numel for each element."""
        prog = ad_transform("fn f(x: f32[4]) -> f32[4] { grad(mean(x), x) }")
        expr = get_body_expr(prog)
        # Should be 1.0 / 4.0 (scalar that gets broadcast)
        assert isinstance(expr, BinOp) and expr.op == "/"


class TestADCodegen:
    def test_grad_scalar_mul_codegen(self):
        """Full pipeline: grad of scalar multiplication generates valid StableHLO."""
        out = ad_codegen("fn f(x: f32) -> f32 { grad(x * x, x) }")
        assert "module {" in out
        assert "func.func @f" in out
        assert "stablehlo" in out

    def test_grad_mean_codegen(self):
        """Full pipeline: grad of mean generates valid StableHLO."""
        out = ad_codegen("fn f(x: f32[4]) -> f32[4] { grad(mean(x), x) }")
        assert "module {" in out

    def test_no_grad_passthrough(self):
        """Functions without grad are unchanged."""
        out = ad_codegen("fn f(a: f32, b: f32) -> f32 { a + b }")
        assert "stablehlo.add" in out


class TestADLetBindings:
    def test_grad_through_let(self):
        """grad should see through let bindings."""
        prog = ad_transform("""
            fn f(x: f32[4], w: f32[4]) -> f32[4] {
                let loss = mean(x * w);
                grad(loss, w)
            }
        """)
        expr = get_body_expr(prog)
        # Should NOT be zero — should compute x / 4
        assert not (isinstance(expr, FloatLiteral) and expr.value == 0.0)

    def test_grad_through_let_codegen(self):
        """Full pipeline with let binding produces non-trivial gradient."""
        out = ad_codegen("""
            fn f(x: f32[4], w: f32[4]) -> f32[4] {
                let loss = mean(x * w);
                grad(loss, w)
            }
        """)
        assert "stablehlo.multiply" in out
        assert "stablehlo.divide" in out


class TestADErrors:
    def test_if_inside_grad(self):
        with pytest.raises(MaomiError, match="unsupported"):
            ad_transform("fn f(x: f32) -> f32 { grad(if x > 0.0 { x } else { 0.0 }, x) }")

    def test_user_fn_inside_grad(self):
        with pytest.raises(MaomiError, match="unsupported"):
            ad_transform("""
                fn g(x: f32) -> f32 { x }
                fn f(x: f32) -> f32 { grad(g(x), x) }
            """)
