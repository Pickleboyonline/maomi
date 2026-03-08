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

    def test_sigmoid(self):
        check_ok("fn f(x: f32) -> f32 { sigmoid(x) }")

    def test_sigmoid_array(self):
        check_ok("fn f(x: f32[4]) -> f32[4] { sigmoid(x) }")

    def test_neg(self):
        check_ok("fn f(x: f32) -> f32 { neg(x) }")

    def test_rsqrt(self):
        check_ok("fn f(x: f32) -> f32 { rsqrt(x) }")

    def test_log2(self):
        check_ok("fn f(x: f32) -> f32 { log2(x) }")

    def test_floor(self):
        check_ok("fn f(x: f32) -> f32 { floor(x) }")

    def test_ceil(self):
        check_ok("fn f(x: f32) -> f32 { ceil(x) }")

    def test_sign(self):
        check_ok("fn f(x: f32) -> f32 { sign(x) }")

    def test_reciprocal(self):
        check_ok("fn f(x: f32) -> f32 { reciprocal(x) }")

    def test_log1p(self):
        check_ok("fn f(x: f32) -> f32 { log1p(x) }")

    def test_log1p_array(self):
        check_ok("fn f(x: f32[3]) -> f32[3] { log1p(x) }")

    def test_expm1(self):
        check_ok("fn f(x: f32) -> f32 { expm1(x) }")

    def test_expm1_array(self):
        check_ok("fn f(x: f32[3]) -> f32[3] { expm1(x) }")
    def test_square(self):
        check_ok("fn f(x: f32) -> f32 { square(x) }")

    def test_square_array(self):
        check_ok("fn f(x: f32[3]) -> f32[3] { square(x) }")

    def test_softplus(self):
        check_ok("fn f(x: f32) -> f32 { softplus(x) }")

    def test_softplus_array(self):
        check_ok("fn f(x: f32[4]) -> f32[4] { softplus(x) }")

    def test_relu(self):
        check_ok("fn f(x: f32) -> f32 { relu(x) }")

    def test_relu_array(self):
        check_ok("fn f(x: f32[4]) -> f32[4] { relu(x) }")

    def test_silu(self):
        check_ok("fn f(x: f32) -> f32 { silu(x) }")

    def test_silu_array(self):
        check_ok("fn f(x: f32[4]) -> f32[4] { silu(x) }")

    def test_gelu(self):
        check_ok("fn f(x: f32) -> f32 { gelu(x) }")

    def test_gelu_array(self):
        check_ok("fn f(x: f32[4]) -> f32[4] { gelu(x) }")


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

class TestStructTypes:
    def test_struct_field_access(self):
        check_ok("""
            struct Point { x: f32, y: f32 }
            fn f(p: Point) -> f32 { p.x }
        """)

    def test_struct_literal_valid(self):
        check_ok("""
            struct Point { x: f32, y: f32 }
            fn f() -> Point { Point { x: 1.0, y: 2.0 } }
        """)

    def test_unknown_struct(self):
        check_err(
            "fn f(p: Unknown) -> f32 { 0.0 }",
            "unknown type",
        )

    def test_wrong_field_name(self):
        check_err("""
            struct Point { x: f32, y: f32 }
            fn f() -> Point { Point { x: 1.0, z: 2.0 } }
        """, "expected field")

    def test_wrong_field_type(self):
        check_err("""
            struct Point { x: f32, y: f32 }
            fn f() -> Point { Point { x: 1, y: 2.0 } }
        """, "expected")

    def test_field_access_nonexistent(self):
        check_err("""
            struct Point { x: f32 }
            fn f(p: Point) -> f32 { p.z }
        """, "no field")

    def test_with_valid(self):
        check_ok("""
            struct Point { x: f32, y: f32 }
            fn f(p: Point) -> Point { p with { x = 1.0 } }
        """)

    def test_nested_struct(self):
        check_ok("""
            struct Inner { w: f32 }
            struct Outer { inner: Inner, b: f32 }
            fn f(o: Outer) -> f32 { o.inner.w }
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


class TestConv2d:
    def test_basic_conv2d(self):
        check_ok("fn f(x: f32[1, 3, 8, 8], w: f32[16, 3, 3, 3]) -> f32[1, 16, 6, 6] { conv2d(x, w) }")

    def test_conv2d_with_stride_pad(self):
        # stride=2, pad=1: (8 + 2*1 - 3) / 2 + 1 = 4
        check_ok("fn f(x: f32[1, 3, 8, 8], w: f32[16, 3, 3, 3]) -> f32[1, 16, 4, 4] { conv2d(x, w, 2, 1) }")

    def test_conv2d_6_args(self):
        # stride_h=1, stride_w=2, pad_h=1, pad_w=0: OH=(8+2-3)/1+1=8, OW=(8+0-3)/2+1=3
        check_ok("fn f(x: f32[1, 3, 8, 8], w: f32[16, 3, 3, 3]) -> f32[1, 16, 8, 3] { conv2d(x, w, 1, 2, 1, 0) }")

    def test_conv2d_channel_mismatch(self):
        check_err(
            "fn f(x: f32[1, 3, 8, 8], w: f32[16, 5, 3, 3]) -> f32[1, 16, 6, 6] { conv2d(x, w) }",
            "input channels",
        )

    def test_conv2d_wrong_rank(self):
        check_err(
            "fn f(x: f32[3, 8, 8], w: f32[16, 3, 3, 3]) -> f32[1, 16, 6, 6] { conv2d(x, w) }",
            "4D",
        )

    def test_conv2d_wrong_arg_count(self):
        check_err(
            "fn f(x: f32[1, 3, 8, 8], w: f32[16, 3, 3, 3]) -> f32[1, 16, 6, 6] { conv2d(x, w, 1) }",
            "2, 4, or 6",
        )


class TestPool:
    def test_max_pool(self):
        # (8 - 2) / 2 + 1 = 4
        check_ok("fn f(x: f32[1, 16, 8, 8]) -> f32[1, 16, 4, 4] { max_pool(x, 2, 2, 2, 2) }")

    def test_avg_pool(self):
        check_ok("fn f(x: f32[1, 16, 8, 8]) -> f32[1, 16, 4, 4] { avg_pool(x, 2, 2, 2, 2) }")

    def test_pool_wrong_rank(self):
        check_err(
            "fn f(x: f32[16, 8, 8]) -> f32[16, 4, 4] { max_pool(x, 2, 2, 2, 2) }",
            "4D",
        )

    def test_pool_wrong_arg_count(self):
        check_err(
            "fn f(x: f32[1, 16, 8, 8]) -> f32[1, 16, 4, 4] { max_pool(x, 2, 2) }",
            "5 arguments",
        )


class TestAxisReduction:
    def test_sum_axis_1(self):
        check_ok("fn f(x: f32[3, 4]) -> f32[3] { sum(x, 1) }")

    def test_sum_axis_0(self):
        check_ok("fn f(x: f32[3, 4]) -> f32[4] { sum(x, 0) }")

    def test_mean_axis(self):
        check_ok("fn f(x: f32[3, 4]) -> f32[4] { mean(x, 0) }")

    def test_sum_no_axis(self):
        check_ok("fn f(x: f32[3, 4]) -> f32 { sum(x) }")

    def test_mean_no_axis(self):
        check_ok("fn f(x: f32[3, 4]) -> f32 { mean(x) }")

    def test_sum_3d(self):
        check_ok("fn f(x: f32[2, 3, 4]) -> f32[2, 4] { sum(x, 1) }")

    def test_sum_axis_out_of_range(self):
        check_err("fn f(x: f32[3, 4]) -> f32[3] { sum(x, 2) }", "out of range")

    def test_sum_1d_axis_0_becomes_scalar(self):
        check_ok("fn f(x: f32[4]) -> f32 { sum(x, 0) }")

    def test_sum_scalar_no_axis(self):
        check_ok("fn f(x: f32) -> f32 { sum(x) }")

    def test_sum_scalar_with_axis_error(self):
        check_err("fn f(x: f32) -> f32 { sum(x, 0) }", "array argument")


class TestSizeOneBroadcast:
    def test_broadcast_trailing(self):
        check_ok("fn f(x: f32[3, 1], y: f32[3, 4]) -> f32[3, 4] { x * y }")

    def test_broadcast_leading(self):
        check_ok("fn f(x: f32[1, 4], y: f32[3, 4]) -> f32[3, 4] { x + y }")

    def test_broadcast_both(self):
        check_ok("fn f(x: f32[3, 1], y: f32[1, 4]) -> f32[3, 4] { x - y }")

    def test_broadcast_no_match(self):
        check_err("fn f(x: f32[3, 2], y: f32[3, 4]) -> f32[3, 4] { x * y }", "mismatched")


class TestStopGradient:
    def test_stop_gradient_scalar(self):
        check_ok("fn f(x: f32) -> f32 { stop_gradient(x) }")

    def test_stop_gradient_array(self):
        check_ok("fn f(x: f32[4]) -> f32[4] { stop_gradient(x) }")

    def test_stop_gradient_wrong_args(self):
        check_err("fn f(x: f32) -> f32 { stop_gradient(x, x) }", "1 argument")


class TestWhere:
    def test_where_basic(self):
        check_ok("fn f(m: bool[4], x: f32[4], y: f32[4]) -> f32[4] { where(m, x, y) }")

    def test_where_scalar_cond(self):
        check_ok("fn f(m: bool, x: f32[4], y: f32[4]) -> f32[4] { where(m, x, y) }")

    def test_where_scalar_branch(self):
        check_ok("fn f(m: bool[4], x: f32[4]) -> f32[4] { where(m, x, 0.0) }")

    def test_where_wrong_cond_type(self):
        check_err("fn f(m: f32[4], x: f32[4], y: f32[4]) -> f32[4] { where(m, x, y) }", "bool")

    def test_where_wrong_arg_count(self):
        check_err("fn f(m: bool[4], x: f32[4]) -> f32[4] { where(m, x) }", "3 arguments")


class TestStringLiteral:
    def test_callback_with_string_label(self):
        check_ok("""
            fn f(x: f32) -> f32 {
                callback("loss", x);
                x
            }
        """)

    def test_callback_string_only(self):
        check_ok("""
            fn f(x: f32) -> f32 {
                callback("hello");
                x
            }
        """)

    def test_callback_multiple_strings(self):
        check_ok("""
            fn f(x: f32) -> f32 {
                callback("epoch", "loss", x);
                x
            }
        """)

    def test_string_let_binding_rejected(self):
        check_err(
            'fn f() -> f32 { let s = "hello"; 0.0 }',
            "cannot bind string",
        )

    def test_string_in_user_function_rejected(self):
        check_err(
            'fn g(x: f32) -> f32 { x }\nfn f() -> f32 { g("hello") }',
            "string literals can only be used as callback arguments",
        )


class TestUtilityBuiltins:
    def test_isfinite_scalar(self):
        check_ok("fn f(x: f32) -> bool { isfinite(x) }")

    def test_isfinite_array(self):
        check_ok("fn f(x: f32[3, 4]) -> bool[3, 4] { isfinite(x) }")

    def test_isfinite_error_int(self):
        check_err("fn f(x: i32) -> bool { isfinite(x) }", "float")

    def test_isfinite_error_no_args(self):
        check_err("fn f(x: f32) -> bool { isfinite() }", "1 argument")

    def test_zeros_like_scalar(self):
        check_ok("fn f(x: f32) -> f32 { zeros_like(x) }")

    def test_zeros_like_array(self):
        check_ok("fn f(x: f32[3, 4]) -> f32[3, 4] { zeros_like(x) }")

    def test_zeros_like_error_int(self):
        check_err("fn f(x: i32) -> i32 { zeros_like(x) }", "float")

    def test_ones_like_scalar(self):
        check_ok("fn f(x: f32) -> f32 { ones_like(x) }")

    def test_ones_like_array(self):
        check_ok("fn f(x: f32[4]) -> f32[4] { ones_like(x) }")

    def test_ones_like_error_no_args(self):
        check_err("fn f(x: f32) -> f32 { ones_like() }", "1 argument")


class TestTwoArgElementwise:
    def test_maximum_scalars(self):
        check_ok("fn f(x: f32, y: f32) -> f32 { maximum(x, y) }")

    def test_maximum_arrays(self):
        check_ok("fn f(x: f32[4], y: f32[4]) -> f32[4] { maximum(x, y) }")

    def test_maximum_broadcast(self):
        check_ok("fn f(x: f32[4], y: f32) -> f32[4] { maximum(x, y) }")

    def test_minimum_scalars(self):
        check_ok("fn f(x: f32, y: f32) -> f32 { minimum(x, y) }")

    def test_minimum_arrays(self):
        check_ok("fn f(x: f32[4], y: f32[4]) -> f32[4] { minimum(x, y) }")

    def test_pow_scalars(self):
        check_ok("fn f(x: f32, y: f32) -> f32 { pow(x, y) }")

    def test_pow_arrays(self):
        check_ok("fn f(x: f32[4], y: f32[4]) -> f32[4] { pow(x, y) }")

    def test_pow_broadcast(self):
        check_ok("fn f(x: f32[4], y: f32) -> f32[4] { pow(x, y) }")

    def test_maximum_wrong_arity(self):
        check_err("fn f(x: f32) -> f32 { maximum(x) }", "expects")

    def test_minimum_wrong_arity(self):
        check_err("fn f(x: f32) -> f32 { minimum(x, x, x) }", "expects")

    def test_pow_int_rejected(self):
        check_err("fn f(x: i32, y: i32) -> i32 { pow(x, y) }", "float")

    def test_maximum_shape_mismatch(self):
        check_err("fn f(x: f32[3], y: f32[4]) -> f32[4] { maximum(x, y) }", "compatible shapes")
class TestClip:
    def test_clip_scalars(self):
        check_ok("fn f(x: f32) -> f32 { clip(x, 0.0, 1.0) }")

    def test_clip_array(self):
        check_ok("fn f(x: f32[4]) -> f32[4] { clip(x, 0.0, 1.0) }")

    def test_clip_all_arrays(self):
        check_ok("fn f(x: f32[4], lo: f32[4], hi: f32[4]) -> f32[4] { clip(x, lo, hi) }")

    def test_clip_broadcast_scalar_bounds(self):
        check_ok("fn f(x: f32[4], lo: f32, hi: f32) -> f32[4] { clip(x, lo, hi) }")

    def test_clip_wrong_arity(self):
        check_err("fn f(x: f32) -> f32 { clip(x, 0.0) }", "expects")

    def test_clip_wrong_type(self):
        check_err("fn f(x: i32) -> i32 { clip(x, 0, 1) }", "float")
class TestOneHot:
    def test_one_hot_scalar(self):
        check_ok("fn f(x: i32) -> f32[5] { one_hot(x, 5) }")

    def test_one_hot_array(self):
        check_ok("fn f(x: i32[3]) -> f32[3, 5] { one_hot(x, 5) }")

    def test_one_hot_2d(self):
        check_ok("fn f(x: i32[3, 4]) -> f32[3, 4, 5] { one_hot(x, 5) }")

    def test_one_hot_wrong_type(self):
        check_err("fn f(x: f32) -> f32[5] { one_hot(x, 5) }", "i32")

    def test_one_hot_no_literal(self):
        check_err("fn f(x: i32, n: i32) -> f32 { one_hot(x, n) }", "literal")

    def test_one_hot_zero_n(self):
        check_err("fn f(x: i32) -> f32[1] { one_hot(x, 0) }", "positive")

    def test_one_hot_wrong_arg_count(self):
        check_err("fn f(x: i32) -> f32[5] { one_hot(x) }", "2 arguments")
class TestLogsumexp:
    def test_logsumexp_all(self):
        check_ok("fn f(x: f32[3, 4]) -> f32 { logsumexp(x) }")

    def test_logsumexp_axis(self):
        check_ok("fn f(x: f32[3, 4]) -> f32[3] { logsumexp(x, axis=1) }")

    def test_logsumexp_keepdims(self):
        check_ok("fn f(x: f32[3, 4]) -> f32[3, 1] { logsumexp(x, axis=1, keepdims=true) }")

    def test_logsumexp_axis0(self):
        check_ok("fn f(x: f32[3, 4]) -> f32[4] { logsumexp(x, axis=0) }")

    def test_logsumexp_1d(self):
        check_ok("fn f(x: f32[5]) -> f32 { logsumexp(x) }")

    def test_logsumexp_axis_error(self):
        check_err(
            "fn f(x: f32[3, 4]) -> f32[3] { logsumexp(x, axis=2) }",
            "axis 2 out of range",
        )
