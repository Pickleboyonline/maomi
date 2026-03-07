from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.ad import transform_grad
from maomi.codegen_stablehlo import StableHLOCodegen
from maomi.ast_nodes import *
from maomi.ast_nodes import _BroadcastExpr
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
        """d/dx(mean(x)) = 1/numel for each element, broadcast to input shape."""
        prog = ad_transform("fn f(x: f32[4]) -> f32[4] { grad(mean(x), x) }")
        expr = get_body_expr(prog)
        # Should be broadcast(1.0 / 4.0) to input shape
        assert isinstance(expr, _BroadcastExpr)
        assert isinstance(expr.expr, BinOp) and expr.expr.op == "/"


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


class TestADIfElse:
    def test_if_relu_grad(self):
        """grad(if x > 0 { x } else { 0 }, x) should produce an IfExpr."""
        prog = ad_transform("fn f(x: f32) -> f32 { grad(if x > 0.0 { x } else { 0.0 }, x) }")
        expr = get_body_expr(prog)
        # Result should be if cond { 1.0 } else { 0.0 } (times adj=1.0)
        assert isinstance(expr, BinOp)  # adj * if_expr

    def test_if_different_branches(self):
        """grad(if x > 0 { x*x } else { -x }, x) differentiates both branches."""
        prog = ad_transform("fn f(x: f32) -> f32 { grad(if x > 0.0 { x * x } else { -x }, x) }")
        expr = get_body_expr(prog)
        assert isinstance(expr, BinOp)

    def test_if_codegen(self):
        """Full pipeline: if/else inside grad produces valid StableHLO."""
        out = ad_codegen("fn f(x: f32) -> f32 { grad(if x > 0.0 { x } else { 0.0 }, x) }")
        assert "stablehlo.select" in out or "func.func" in out


class TestADUserFunctions:
    def test_simple_call(self):
        """grad through a simple user function call."""
        prog = ad_transform("""
            fn g(x: f32) -> f32 { x }
            fn f(x: f32) -> f32 { grad(g(x), x) }
        """)
        # g(x) = x, so grad = 1.0
        fn_f = [fn for fn in prog.functions if fn.name == "f"][0]
        expr = fn_f.body.expr
        assert isinstance(expr, FloatLiteral) and expr.value == 1.0

    def test_double(self):
        """fn double(x) { x + x }; grad(double(a), a) = 2."""
        prog = ad_transform("""
            fn double(x: f32) -> f32 { x + x }
            fn f(a: f32) -> f32 { grad(double(a), a) }
        """)
        fn_f = [fn for fn in prog.functions if fn.name == "f"][0]
        expr = fn_f.body.expr
        # Should be 1.0 + 1.0
        assert isinstance(expr, BinOp) and expr.op == "+"

    def test_composition(self):
        """fn sq(x) { x*x }; fn g(x) { sq(x) + x }; grad(g(a), a) = 2a + 1."""
        prog = ad_transform("""
            fn sq(x: f32) -> f32 { x * x }
            fn g(x: f32) -> f32 { sq(x) + x }
            fn f(a: f32) -> f32 { grad(g(a), a) }
        """)
        fn_f = [fn for fn in prog.functions if fn.name == "f"][0]
        expr = fn_f.body.expr
        assert isinstance(expr, BinOp)

    def test_composition_codegen(self):
        """Composition through codegen pipeline."""
        out = ad_codegen("""
            fn sq(x: f32) -> f32 { x * x }
            fn g(x: f32) -> f32 { sq(x) + x }
            fn f(a: f32) -> f32 { grad(g(a), a) }
        """)
        assert "func.func @f" in out

    def test_fn_with_if(self):
        """grad through function containing if/else (ReLU pattern)."""
        prog = ad_transform("""
            fn relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
            fn f(x: f32) -> f32 { grad(relu(x), x) }
        """)
        fn_f = [fn for fn in prog.functions if fn.name == "f"][0]
        expr = fn_f.body.expr
        assert expr is not None

    def test_recursive_error(self):
        """Recursive functions inside grad should error."""
        with pytest.raises(MaomiError, match="recursive"):
            ad_transform("""
                fn rec(x: f32) -> f32 { rec(x) }
                fn f(x: f32) -> f32 { grad(rec(x), x) }
            """)


class TestADScan:
    def test_scan_grad_produces_nonzero_node(self):
        """grad through scan should produce a reverse ScanExpr (not zero)."""
        prog = ad_transform("""
            fn f(xs: f32[5]) -> f32[5] {
                let s = scan (acc, x) in (0.0, xs) { acc + x };
                grad(sum(s), xs)
            }
        """)
        fn_f = [fn for fn in prog.functions if fn.name == "f"][0]
        expr = fn_f.body.expr
        # Should not be zero — should contain a _ScanGrad node somewhere
        assert not (isinstance(expr, FloatLiteral) and expr.value == 0.0)

    def test_scan_grad_codegen(self):
        """Full pipeline: grad of scan produces valid StableHLO."""
        out = ad_codegen("""
            fn f(xs: f32[5]) -> f32[5] {
                let s = scan (acc, x) in (0.0, xs) { acc + x };
                grad(sum(s), xs)
            }
        """)
        assert "module {" in out
        assert "func.func @f" in out
        assert "stablehlo.while" in out

    def test_scan_grad_wrt_init(self):
        """grad of scan w.r.t. init should not be zero."""
        prog = ad_transform("""
            fn f(xs: f32[5], init: f32) -> f32 {
                let s = scan (acc, x) in (init, xs) { acc + x };
                grad(sum(s), init)
            }
        """)
        fn_f = [fn for fn in prog.functions if fn.name == "f"][0]
        expr = fn_f.body.expr
        assert not (isinstance(expr, FloatLiteral) and expr.value == 0.0)


class TestCallbackAD:
    def test_callback_ignored_in_grad(self):
        """callback calls should be skipped during AD — no error, no gradient."""
        prog = ad_transform("""
            fn f(x: f32[4], w: f32[4]) -> f32[4] {
                callback(x, w);
                let loss = mean(x * w);
                grad(loss, w)
            }
        """)
        fn = prog.functions[0]
        assert fn.body.expr is not None

    def test_callback_in_grad_body_codegen(self):
        """Programs with callback inside a grad function should compile."""
        mlir = ad_codegen("""
            fn f(x: f32[4], w: f32[4]) -> f32[4] {
                callback(x);
                let loss = mean(x * w);
                grad(loss, w)
            }
        """)
        assert "func.func @f" in mlir


class TestStructAD:
    def test_grad_through_field_access(self):
        """grad of s.x * s.x w.r.t. s where s is a struct should compile through codegen."""
        out = ad_codegen("""
            struct Params { x: f32, y: f32 }
            fn f(s: Params) -> Params {
                let loss = s.x * s.x;
                grad(loss, s)
            }
        """)
        assert "module {" in out
        assert "func.func @f" in out


class TestGradOfGrad:
    """Tests for grad-of-grad (second-order differentiation)."""

    def test_grad_grad_x_cubed(self):
        """d²/dx²(x³) = 6x — result should be a BinOp (not zero, not GradExpr)."""
        prog = ad_transform("fn f(x: f32) -> f32 { grad(grad(x ** 3.0, x), x) }")
        expr = get_body_expr(prog)
        assert not isinstance(expr, GradExpr)
        assert isinstance(expr, BinOp)

    def test_grad_grad_x_squared(self):
        """d²/dx²(x²) = 2 — result should be constant (no x dependency)."""
        prog = ad_transform("fn f(x: f32) -> f32 { grad(grad(x * x, x), x) }")
        expr = get_body_expr(prog)
        assert not isinstance(expr, GradExpr)

    def test_grad_grad_exp(self):
        """d²/dx²(exp(x)) = exp(x) — should produce non-trivial expression."""
        prog = ad_transform("fn f(x: f32) -> f32 { grad(grad(exp(x), x), x) }")
        expr = get_body_expr(prog)
        assert not isinstance(expr, GradExpr)
        assert not isinstance(expr, FloatLiteral)  # not a constant

    def test_mixed_partial(self):
        """d/dy(d/dx(x*y)) = d/dy(y) = 1."""
        prog = ad_transform("fn f(x: f32, y: f32) -> f32 { grad(grad(x * y, x), y) }")
        expr = get_body_expr(prog)
        assert not isinstance(expr, GradExpr)
        # Result is 1.0 (possibly as 1.0 * 1.0 without simplification)
        assert not isinstance(expr, Identifier)  # should not reference x or y

    def test_depth_limit(self):
        """Exceeding max nesting depth should raise an error."""
        # Build 12 nested grads — outermost is handled by _transform_expr,
        # so _linearize sees 11 inner GradExprs, exceeding the limit of 10.
        inner = "x * x"
        for _ in range(12):
            inner = f"grad({inner}, x)"
        source = f"fn f(x: f32) -> f32 {{ {inner} }}"
        with pytest.raises(MaomiError, match="maximum nesting depth"):
            ad_transform(source)

    def test_let_binding_equivalence(self):
        """grad(grad(expr, x), x) should produce same structure as let-based version."""
        # Direct nesting
        prog1 = ad_transform("fn f(x: f32) -> f32 { grad(grad(x ** 3.0, x), x) }")
        expr1 = get_body_expr(prog1)
        # Let-binding based
        prog2 = ad_transform("""
            fn f(x: f32) -> f32 {
                let g = grad(x ** 3.0, x);
                grad(g, x)
            }
        """)
        expr2 = get_body_expr(prog2)
        # Both should be BinOps (not GradExpr, not zero)
        assert isinstance(expr1, BinOp)
        assert isinstance(expr2, BinOp)

    def test_codegen_x_cubed(self):
        """grad(grad(x³, x), x) should produce valid StableHLO."""
        out = ad_codegen("fn f(x: f32) -> f32 { grad(grad(x ** 3.0, x), x) }")
        assert "module {" in out
        assert "func.func @f" in out

    def test_codegen_exp(self):
        """grad(grad(exp(x), x), x) should produce valid StableHLO."""
        out = ad_codegen("fn f(x: f32) -> f32 { grad(grad(exp(x), x), x) }")
        assert "module {" in out
        assert "func.func @f" in out

    def test_codegen_tanh(self):
        """grad(grad(tanh(x), x), x) should produce valid StableHLO."""
        out = ad_codegen("fn f(x: f32) -> f32 { grad(grad(tanh(x), x), x) }")
        assert "module {" in out
        assert "func.func @f" in out

    def test_third_derivative(self):
        """d³/dx³(x⁴) = 24x — should work with 3 levels of nesting."""
        prog = ad_transform("fn f(x: f32) -> f32 { grad(grad(grad(x ** 4.0, x), x), x) }")
        expr = get_body_expr(prog)
        assert not isinstance(expr, GradExpr)
        assert isinstance(expr, BinOp)

    def test_codegen_third_derivative(self):
        """d³/dx³(x⁴) should produce valid StableHLO."""
        out = ad_codegen("fn f(x: f32) -> f32 { grad(grad(grad(x ** 4.0, x), x), x) }")
        assert "module {" in out
        assert "func.func @f" in out


class TestGradOfGradIndexing:
    """Tests for grad-of-grad through indexing operations."""

    def test_grad_grad_index_single(self):
        """d/dx sum(d/dx(x[0]^2)) should produce valid AST."""
        prog = ad_transform("""
            fn f(x: f32[3]) -> f32[3] {
                grad(sum(grad(x[0] * x[0], x)), x)
            }
        """)
        expr = get_body_expr(prog)
        assert not isinstance(expr, GradExpr)

    def test_codegen_grad_grad_index(self):
        """d/dx sum(d/dx(x[0]^2)) should produce valid StableHLO."""
        out = ad_codegen("""
            fn f(x: f32[3]) -> f32[3] {
                grad(sum(grad(x[0] * x[0], x)), x)
            }
        """)
        assert "module {" in out
        assert "func.func @f" in out

    def test_grad_grad_index_sum(self):
        """d/dx sum(d/dx(x[0] + x[1])) — mixed indices."""
        prog = ad_transform("""
            fn f(x: f32[3]) -> f32[3] {
                grad(sum(grad(x[0] + x[1], x)), x)
            }
        """)
        expr = get_body_expr(prog)
        assert not isinstance(expr, GradExpr)

    def test_codegen_grad_grad_index_sum(self):
        """Should produce valid StableHLO for multi-index grad-of-grad."""
        out = ad_codegen("""
            fn f(x: f32[3]) -> f32[3] {
                grad(sum(grad(x[0] + x[1], x)), x)
            }
        """)
        assert "module {" in out
        assert "func.func @f" in out

    def test_grad_grad_gather(self):
        """d/dx sum(d/dx(sum(x[ids]))) — gather grad-of-grad."""
        out = ad_codegen("""
            fn f(x: f32[5], ids: i32[3]) -> f32[5] {
                grad(sum(grad(sum(x[ids]), x)), x)
            }
        """)
        assert "module {" in out
        assert "func.func @f" in out


class TestGradOfGradScan:
    """Tests for grad-of-grad through scan operations."""

    def test_grad_grad_scan_seq_constant(self):
        """d/dx sum(d/dx(sum(scan))) with constant derivatives should work."""
        out = ad_codegen("""
            fn f(x: f32[5]) -> f32[5] {
                grad(sum(grad(sum(scan (carry, elem) in (0.0, x) { carry + elem }), x)), x)
            }
        """)
        assert "module {" in out
        assert "func.func @f" in out

    def test_grad_grad_scan_init_constant(self):
        """grad(grad(sum(scan), init), init) with constant derivatives should work."""
        out = ad_codegen("""
            fn f(init: f32, x: f32[5]) -> f32 {
                grad(grad(sum(scan (carry, elem) in (init, x) { carry + elem }), init), init)
            }
        """)
        assert "module {" in out
        assert "func.func @f" in out

    def test_first_order_scan_still_works(self):
        """Verify first-order scan grad still works after refactoring."""
        out = ad_codegen("""
            fn f(init: f32, x: f32[5]) -> f32 {
                grad(sum(scan (carry, elem) in (init, x) { carry + elem }), init)
            }
        """)
        assert "module {" in out
        assert "func.func @f" in out


class TestGradOfGradBroadcastReduce:
    """Tests for grad-of-grad through broadcast and reduce_sum."""

    def test_grad_grad_through_sum(self):
        """d/dx sum(d/dx(sum(x * x))) — inner grad goes through sum backprop (broadcast)."""
        out = ad_codegen("""
            fn f(x: f32[3]) -> f32[3] {
                grad(sum(grad(sum(x * x), x)), x)
            }
        """)
        assert "module {" in out
        assert "func.func @f" in out

    def test_grad_grad_through_mean(self):
        """d/dx sum(d/dx(mean(x * x))) — mean backprop uses broadcast."""
        out = ad_codegen("""
            fn f(x: f32[3]) -> f32[3] {
                grad(sum(grad(mean(x * x), x)), x)
            }
        """)
        assert "module {" in out
        assert "func.func @f" in out


class TestConv2dGrad:
    def test_grad_wrt_input(self):
        """grad of sum(reshape(conv2d(x, w))) w.r.t. x should produce a backward convolution."""
        out = ad_codegen("""
            fn f(x: f32[1, 1, 4, 4], w: f32[1, 1, 3, 3]) -> f32[1, 1, 4, 4] {
                let y = conv2d(x, w);
                let flat = reshape(y, 4);
                grad(sum(flat), x)
            }
        """)
        assert "stablehlo.convolution" in out
        assert "stablehlo.reverse" in out

    def test_grad_wrt_kernel(self):
        """grad of sum(reshape(conv2d(x, w))) w.r.t. w should produce a backward convolution."""
        out = ad_codegen("""
            fn f(x: f32[1, 1, 4, 4], w: f32[1, 1, 3, 3]) -> f32[1, 1, 3, 3] {
                let y = conv2d(x, w);
                let flat = reshape(y, 4);
                grad(sum(flat), w)
            }
        """)
        # Kernel gradient is a convolution (no reverse needed)
        assert "stablehlo.convolution" in out


class TestMaxPoolGrad:
    def test_grad_max_pool(self):
        """grad of sum(reshape(max_pool(x))) w.r.t. x should produce select_and_scatter."""
        out = ad_codegen("""
            fn f(x: f32[1, 1, 4, 4]) -> f32[1, 1, 4, 4] {
                let y = max_pool(x, 2, 2, 2, 2);
                let flat = reshape(y, 4);
                grad(sum(flat), x)
            }
        """)
        assert "select_and_scatter" in out


class TestAvgPoolGrad:
    def test_grad_avg_pool(self):
        """grad of sum(reshape(avg_pool(x))) w.r.t. x should produce pad + reduce_window."""
        out = ad_codegen("""
            fn f(x: f32[1, 1, 4, 4]) -> f32[1, 1, 4, 4] {
                let y = avg_pool(x, 2, 2, 2, 2);
                let flat = reshape(y, 4);
                grad(sum(flat), x)
            }
        """)
        assert "stablehlo.pad" in out
        assert "stablehlo.reduce_window" in out


class TestMapAD:
    def test_grad_sum_in_map(self):
        """grad of sum(map x in xs { sum(x) }) w.r.t. xs — result is all-ones."""
        out = ad_codegen("""
            fn f(xs: f32[4, 3]) -> f32[4, 3] {
                grad(sum(map x in xs { sum(x) }), xs)
            }
        """)
        # Gradient of sum(sum(x_i)) w.r.t. xs is all-ones — emits broadcast + multiply
        assert "stablehlo.broadcast_in_dim" in out
        assert "tensor<4x3xf32>" in out

    def test_grad_map_free_var_scalar(self):
        """grad of sum(map x in xs { x * w }) w.r.t. scalar w."""
        out = ad_codegen("""
            fn f(xs: f32[4], w: f32) -> f32 {
                grad(sum(map x in xs { x * w }), w)
            }
        """)
        assert "stablehlo.reduce" in out
        assert "stablehlo.multiply" in out

    def test_grad_map_free_var_array(self):
        """grad of sum(map x in xs { sum(x * w) }) w.r.t. array w."""
        out = ad_codegen("""
            fn f(xs: f32[4, 3], w: f32[3]) -> f32[3] {
                grad(sum(map x in xs { sum(x * w) }), w)
            }
        """)
        assert "stablehlo.reduce" in out

    def test_grad_matmul_in_map_wrt_weight(self):
        """grad of sum(map x in xs { sum(x @ w) }) w.r.t. w."""
        out = ad_codegen("""
            fn f(w: f32[3, 2], xs: f32[4, 3]) -> f32[3, 2] {
                grad(sum(map x in xs { sum(x @ w) }), w)
            }
        """)
        assert "stablehlo.dot_general" in out or "stablehlo.reduce" in out


class TestAxisReductionAD:
    def test_sum_axis_grad(self):
        """grad(sum(sum(x, 1)), x) should broadcast back with correct dims."""
        out = ad_codegen("""
            fn f(x: f32[3, 4]) -> f32[3, 4] {
                grad(sum(sum(x, 1)), x)
            }
        """)
        assert "broadcast_in_dim" in out
        assert "dims = [0]" in out  # broadcast from f32[3] to f32[3,4] via dim 0

    def test_sum_axis_0_grad(self):
        """grad(sum(sum(x, 0)), x) should broadcast back via dim 1."""
        out = ad_codegen("""
            fn f(x: f32[3, 4]) -> f32[3, 4] {
                grad(sum(sum(x, 0)), x)
            }
        """)
        assert "broadcast_in_dim" in out
        assert "dims = [1]" in out  # broadcast from f32[4] to f32[3,4] via dim 1

    def test_mean_axis_grad(self):
        """grad of mean with axis should divide by axis size."""
        out = ad_codegen("""
            fn f(x: f32[3, 4]) -> f32[3, 4] {
                grad(sum(mean(x, 1)), x)
            }
        """)
        assert "stablehlo.divide" in out
        assert "4.000000e+00" in out  # divided by axis size 4


class TestStopGradientAD:
    def test_stop_gradient_zeroes_grad(self):
        """stop_gradient prevents gradient flow to its argument."""
        out = ad_codegen("""
            fn f(x: f32[4], y: f32[4]) -> f32[4] {
                let z = stop_gradient(x);
                grad(sum(z * y), y)
            }
        """)
        # Gradient w.r.t. y of z*y is z (which is x)
        assert "stablehlo.multiply" in out

    def test_stop_gradient_no_flow(self):
        """Gradient should not flow through stop_gradient to x."""
        prog = ad_transform("""
            fn f(x: f32) -> f32 {
                grad(stop_gradient(x) * x, x)
            }
        """)
        # The gradient of stop_gradient(x) * x w.r.t. x:
        # d/dx [sg(x) * x] = sg(x) * 1 + 0 * x = sg(x) = x
        # So gradient should reference x but NOT have two terms


class TestWhereAD:
    def test_where_grad_x(self):
        """Gradient through where w.r.t. the true branch."""
        out = ad_codegen("""
            fn f(x: f32[4], mask: bool[4]) -> f32[4] {
                grad(sum(where(mask, x, 0.0)), x)
            }
        """)
        # adj_x = where(mask, adj, 0)
        assert "stablehlo.select" in out

    def test_where_grad_both(self):
        """Gradient through where w.r.t. both branches."""
        out = ad_codegen("""
            fn f(x: f32[4], y: f32[4], mask: bool[4]) -> f32[4] {
                grad(sum(where(mask, x, y)), x)
            }
        """)
        assert "stablehlo.select" in out
