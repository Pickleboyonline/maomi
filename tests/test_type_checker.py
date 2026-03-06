from pathlib import Path
from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
import pytest


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


class TestScalarArithmetic:
    def test_f32_add(self):
        check_ok("fn f(a: f32, b: f32) -> f32 { a + b }")

    def test_i32_mul(self):
        check_ok("fn f(a: i32, b: i32) -> i32 { a * b }")

    def test_type_mismatch(self):
        check_err(
            "fn f(a: f32, b: i32) -> f32 { a + b }",
            "mismatched types",
        )

    def test_unary_minus(self):
        check_ok("fn f(x: f32) -> f32 { -x }")


class TestLiterals:
    def test_int_literal_is_i32(self):
        check_ok("fn f() -> i32 { 42 }")

    def test_float_literal_is_f32(self):
        check_ok("fn f() -> f32 { 3.14 }")

    def test_bool_literal(self):
        check_ok("fn f() -> bool { true }")


class TestMatmul:
    def test_basic_2d(self):
        check_ok("fn f(a: f32[M, K], b: f32[K, N]) -> f32[M, N] { a @ b }")

    def test_concrete_dims(self):
        check_ok("fn f(a: f32[32, 128], b: f32[128, 64]) -> f32[32, 64] { a @ b }")

    def test_dimension_mismatch(self):
        check_err(
            "fn f(a: f32[32, 128], b: f32[64, 64]) -> f32[32, 64] { a @ b }",
            "dimension mismatch",
        )

    def test_non_array_operand(self):
        check_err(
            "fn f(a: f32, b: f32) -> f32 { a @ b }",
            "must be arrays",
        )

    def test_base_type_mismatch(self):
        check_err(
            "fn f(a: f32[2, 3], b: f64[3, 4]) -> f32[2, 4] { a @ b }",
            "base type mismatch",
        )


class TestComparisons:
    def test_f32_comparison(self):
        check_ok("fn f(a: f32, b: f32) -> bool { a > b }")

    def test_comparison_result_is_bool(self):
        check_err(
            "fn f(a: f32, b: f32) -> f32 { a > b }",
            "return type mismatch",
        )


class TestIfExpr:
    def test_basic_if(self):
        check_ok("fn f(x: f32) -> f32 { if true { x } else { 0.0 } }")

    def test_branch_type_mismatch(self):
        check_err(
            "fn f(x: f32) -> f32 { if true { x } else { 42 } }",
            "different types",
        )


class TestLetBindings:
    def test_basic_let(self):
        check_ok("fn f(a: f32, b: f32) -> f32 { let x = a + b; x }")

    def test_let_with_matching_annotation(self):
        check_ok("fn f(a: f32) -> f32 { let x: f32 = a; x }")

    def test_let_with_wrong_annotation(self):
        check_err(
            "fn f(a: f32) -> f32 { let x: i32 = a; x }",
            "type mismatch in let binding",
        )


class TestFunctionCalls:
    def test_simple_call(self):
        check_ok("""
            fn add(a: f32, b: f32) -> f32 { a + b }
            fn main() -> f32 { add(1.0, 2.0) }
        """)

    def test_wrong_arity(self):
        check_err(
            """
            fn add(a: f32, b: f32) -> f32 { a + b }
            fn main() -> f32 { add(1.0) }
            """,
            "expects 2 arguments, got 1",
        )

    def test_undefined_function(self):
        check_err(
            "fn f() -> f32 { unknown() }",
            "undefined function",
        )

    def test_undefined_variable(self):
        check_err(
            "fn f() -> f32 { x }",
            "undefined variable",
        )


class TestSymbolicDimUnification:
    def test_bind_symbolic(self):
        check_ok("""
            fn identity(x: f32[N]) -> f32[N] { x }
            fn main(v: f32[64]) -> f32[64] { identity(v) }
        """)

    def test_matmul_with_symbolic(self):
        check_ok("""
            fn linear(x: f32[B, N], w: f32[N, M]) -> f32[B, M] { x @ w }
            fn main(x: f32[32, 128], w: f32[128, 64]) -> f32[32, 64] { linear(x, w) }
        """)

    def test_inconsistent_symbolic(self):
        check_err(
            """
            fn f(a: f32[N], b: f32[N]) -> f32[N] { a }
            fn main(x: f32[3], y: f32[5]) -> f32[3] { f(x, y) }
            """,
            "dimension",
        )


class TestReturnType:
    def test_correct_return(self):
        check_ok("fn f() -> f32 { 1.0 }")

    def test_wrong_return(self):
        check_err(
            "fn f() -> i32 { 1.0 }",
            "return type mismatch",
        )


class TestBuiltins:
    def test_mean(self):
        check_ok("fn f(x: f32[N]) -> f32 { mean(x) }")

    def test_exp(self):
        check_ok("fn f(x: f32) -> f32 { exp(x) }")


class TestScan:
    def test_basic_scan(self):
        check_ok("""
            fn f(xs: f32[10, 4], h0: f32[8], w: f32[4, 8]) -> f32[10, 8] {
                scan (h, x) in (h0, xs) {
                    tanh(h + x @ w)
                }
            }
        """)

    def test_scan_scalar_carry(self):
        check_ok("""
            fn f(xs: f32[5], init: f32) -> f32[5] {
                scan (acc, x) in (init, xs) {
                    acc + x
                }
            }
        """)

    def test_scan_sequence_not_array(self):
        check_err(
            "fn f(x: f32, init: f32) -> f32 { scan (acc, el) in (init, x) { acc } }",
            "must be an array",
        )

    def test_scan_body_type_mismatch(self):
        check_err("""
            fn f(xs: f32[5], init: f32) -> f32[5] {
                scan (acc, x) in (init, xs) {
                    42
                }
            }
        """, "scan body returns")

    def test_multi_sequence_scan(self):
        check_ok("""
            fn f(xs: f32[5], ys: f32[5]) -> f32[5] {
                scan (acc, (x, y)) in (0.0, (xs, ys)) {
                    acc + x * y
                }
            }
        """)

    def test_multi_sequence_dim_mismatch(self):
        check_err("""
            fn f(xs: f32[5], ys: f32[3]) -> f32[5] {
                scan (acc, (x, y)) in (0.0, (xs, ys)) {
                    acc + x * y
                }
            }
        """, "same first dimension")


class TestMap:
    def test_basic_map(self):
        check_ok("""
            fn f(xs: f32[32, 64]) -> f32[32, 64] {
                map x in xs {
                    if x > 0.0 { x } else { 0.0 }
                }
            }
        """)

    def test_map_1d(self):
        check_ok("""
            fn f(xs: f32[10]) -> f32[10] {
                map x in xs { x * 2.0 }
            }
        """)

    def test_map_sequence_not_array(self):
        check_err(
            "fn f(x: f32) -> f32 { map el in x { el } }",
            "must be an array",
        )


class TestGrad:
    def test_basic_grad(self):
        check_ok("""
            fn f(x: f32, w: f32) -> f32 {
                grad(x * w, w)
            }
        """)

    def test_grad_returns_wrt_type(self):
        check_ok("""
            fn f(x: f32[4], w: f32[4, 2]) -> f32[4, 2] {
                let loss = mean((x @ w) ** 2.0);
                grad(loss, w)
            }
        """)

    def test_grad_non_scalar_expr(self):
        check_err("""
            fn f(x: f32[4], w: f32[4]) -> f32[4] {
                grad(x + w, w)
            }
        """, "must be scalar")

    def test_grad_undefined_var(self):
        check_err(
            "fn f(x: f32) -> f32 { grad(x, y) }",
            "undefined variable",
        )


class TestCallback:
    def test_callback_any_args(self):
        check_ok("""
            fn f(x: f32, y: f32[4]) -> f32 {
                callback(x, y);
                x
            }
        """)

    def test_callback_no_args(self):
        check_ok("""
            fn f(x: f32) -> f32 {
                callback();
                x
            }
        """)

class TestFixtures:
    fixtures_dir = Path(__file__).parent / "fixtures"

    def test_linear(self):
        source = (self.fixtures_dir / "linear.mao").read_text()
        errors = check(source)
        assert errors == []

    def test_relu(self):
        source = (self.fixtures_dir / "relu.mao").read_text()
        errors = check(source)
        assert errors == []

    def test_mlp(self):
        source = (self.fixtures_dir / "mlp.mao").read_text()
        errors = check(source)
        assert errors == []
