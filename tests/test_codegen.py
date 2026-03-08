from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.codegen.stablehlo import StableHLOCodegen
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

    def test_builtin_sigmoid(self):
        out = codegen("fn f(x: f32) -> f32 { sigmoid(x) }")
        assert "stablehlo.negate" in out
        assert "stablehlo.exponential" in out
        assert "stablehlo.divide" in out

    def test_builtin_neg(self):
        out = codegen("fn f(x: f32) -> f32 { neg(x) }")
        assert "stablehlo.negate" in out

    def test_builtin_rsqrt(self):
        out = codegen("fn f(x: f32) -> f32 { rsqrt(x) }")
        assert "stablehlo.rsqrt" in out

    def test_builtin_log2(self):
        out = codegen("fn f(x: f32) -> f32 { log2(x) }")
        assert "stablehlo.log" in out
        assert "stablehlo.divide" in out

    def test_builtin_floor(self):
        out = codegen("fn f(x: f32) -> f32 { floor(x) }")
        assert "stablehlo.floor" in out

    def test_builtin_ceil(self):
        out = codegen("fn f(x: f32) -> f32 { ceil(x) }")
        assert "stablehlo.ceil" in out

    def test_builtin_sign(self):
        out = codegen("fn f(x: f32) -> f32 { sign(x) }")
        assert "stablehlo.sign" in out

    def test_builtin_reciprocal(self):
        out = codegen("fn f(x: f32) -> f32 { reciprocal(x) }")
        assert "stablehlo.divide" in out

    def test_log1p(self):
        out = codegen("fn f(x: f32) -> f32 { log1p(x) }")
        assert "stablehlo.log_plus_one" in out

    def test_log1p_array(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { log1p(x) }")
        assert "stablehlo.log_plus_one" in out

    def test_expm1(self):
        out = codegen("fn f(x: f32) -> f32 { expm1(x) }")
        assert "stablehlo.exponential_minus_one" in out

    def test_expm1_array(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { expm1(x) }")
        assert "stablehlo.exponential_minus_one" in out

    def test_builtin_sigmoid_array(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { sigmoid(x) }")
        assert "stablehlo.negate" in out
        assert "tensor<4xf32>" in out

    def test_builtin_square(self):
        out = codegen("fn f(x: f32) -> f32 { square(x) }")
        assert "stablehlo.multiply" in out

    def test_builtin_square_array(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { square(x) }")
        assert "stablehlo.multiply" in out

    def test_builtin_relu(self):
        out = codegen("fn f(x: f32) -> f32 { relu(x) }")
        assert "stablehlo.compare" in out
        assert "stablehlo.select" in out

    def test_builtin_relu_array(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { relu(x) }")
        assert "stablehlo.compare" in out
        assert "stablehlo.select" in out

    def test_builtin_softplus(self):
        out = codegen("fn f(x: f32) -> f32 { softplus(x) }")
        assert "stablehlo.exponential" in out
        assert "stablehlo.log" in out

    def test_builtin_silu(self):
        out = codegen("fn f(x: f32) -> f32 { silu(x) }")
        assert "stablehlo.exponential" in out

    def test_builtin_gelu(self):
        out = codegen("fn f(x: f32) -> f32 { gelu(x) }")
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


class TestGeneralMap:
    def test_matmul_in_map(self):
        out = codegen("""
            fn f(w: f32[64, 10], xs: f32[32, 64]) -> f32[32, 10] {
                map x in xs { x @ w }
            }
        """)
        assert "stablehlo.dot_general" in out
        assert "batching_dims = [0] x [0]" in out
        assert "tensor<32x10xf32>" in out

    def test_sum_in_map(self):
        out = codegen("""
            fn f(xs: f32[32, 64]) -> f32[32] {
                map x in xs { sum(x) }
            }
        """)
        assert "stablehlo.reduce" in out
        assert "dimensions = [1]" in out
        assert "tensor<32xf32>" in out

    def test_mean_in_map(self):
        out = codegen("""
            fn f(xs: f32[32, 64]) -> f32[32] {
                map x in xs { mean(x) }
            }
        """)
        assert "stablehlo.reduce" in out
        assert "stablehlo.divide" in out

    def test_transpose_in_map(self):
        out = codegen("""
            fn f(xs: f32[32, 4, 8]) -> f32[32, 8, 4] {
                map x in xs { transpose(x) }
            }
        """)
        assert "stablehlo.transpose" in out
        assert "dims = [0, 2, 1]" in out

    def test_matmul_sum_in_map(self):
        out = codegen("""
            fn f(w: f32[64, 10], xs: f32[32, 64]) -> f32[32] {
                map x in xs { sum(x @ w) }
            }
        """)
        assert "stablehlo.dot_general" in out
        assert "batching_dims = [0] x [0]" in out
        assert "stablehlo.reduce" in out

    def test_matmul_bias_in_map(self):
        out = codegen("""
            fn f(w: f32[64, 10], b: f32[10], xs: f32[32, 64]) -> f32[32, 10] {
                map x in xs { x @ w + b }
            }
        """)
        assert "stablehlo.dot_general" in out
        assert "stablehlo.add" in out

    def test_index_in_map(self):
        out = codegen("""
            fn f(xs: f32[32, 64]) -> f32[32] {
                map x in xs { x[0] }
            }
        """)
        assert "stablehlo.slice" in out or "stablehlo.dynamic_slice" in out

    def test_let_binding_in_map(self):
        out = codegen("""
            fn f(w: f32[64, 10], xs: f32[32, 64]) -> f32[32, 10] {
                map x in xs {
                    let h = x @ w;
                    h + h
                }
            }
        """)
        assert "stablehlo.dot_general" in out
        assert "stablehlo.add" in out

    def test_nested_map(self):
        out = codegen("""
            fn f(xs: f32[5, 10]) -> f32[5, 10] {
                map row in xs {
                    map elem in row { elem * 2.0 }
                }
            }
        """)
        assert "stablehlo.multiply" in out

    def test_fn_call_in_map(self):
        out = codegen("""
            fn predict(w: f32[64, 10], x: f32[64]) -> f32[10] {
                x @ w
            }
            fn batch(w: f32[64, 10], xs: f32[32, 64]) -> f32[32, 10] {
                map x in xs { predict(w, x) }
            }
        """)
        assert "func.func @predict_vmap_32" in out
        assert "func.call @predict_vmap_32" in out
        assert "batching_dims" in out

    def test_fn_call_with_bias_in_map(self):
        out = codegen("""
            fn layer(w: f32[64, 10], b: f32[10], x: f32[64]) -> f32[10] {
                x @ w + b
            }
            fn batch_layer(w: f32[64, 10], b: f32[10], xs: f32[32, 64]) -> f32[32, 10] {
                map x in xs { layer(w, b, x) }
            }
        """)
        assert "func.func @layer_vmap_32" in out
        assert "func.call @layer_vmap_32" in out

    def test_scan_in_map(self):
        out = codegen("""
            fn f(xs: f32[32, 10]) -> f32[32, 10] {
                map x in xs {
                    scan (acc, e) in (0.0, x) { acc + e }
                }
            }
        """)
        assert "stablehlo.while" in out
        assert "stablehlo.add" in out
        assert "tensor<32x10xf32>" in out


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
    def test_symbolic_dim_generic_not_called(self):
        """A generic function with symbolic dims that's never called should compile (skipped by codegen)."""
        out = codegen("fn f(x: f32[N]) -> f32 { mean(x) }")
        # The function is generic (symbolic dim), so it's skipped — only monomorphized copies are generated
        assert "func.func @f" not in out


class TestCallbackCodegen:
    def test_callback_compiles(self):
        """callback should emit stablehlo.custom_call for FFI callback."""
        out = codegen("""
            fn f(x: f32, y: f32) -> f32 {
                callback(x, y);
                x + y
            }
        """)
        assert "xla_ffi_python_cpu_callback" in out
        assert "has_side_effect = true" in out
        assert "func.func @f" in out

    def test_callback_no_args_compiles(self):
        out = codegen("""
            fn f(x: f32) -> f32 {
                callback();
                x
            }
        """)
        assert "xla_ffi_python_cpu_callback" in out
        assert "func.func @f" in out

    def test_callback_multiple_have_different_indices(self):
        out = codegen("""
            fn f(x: f32) -> f32 {
                callback(x);
                callback(x);
                x
            }
        """)
        assert "index = 0 : ui64" in out
        assert "index = 1 : ui64" in out


class TestStructCodegen:
    def test_struct_literal(self):
        out = codegen("""
            struct Point { x: f32, y: f32 }
            fn f() -> Point { Point { x: 1.0, y: 2.0 } }
        """)
        assert "stablehlo.tuple" in out

    def test_field_access(self):
        out = codegen("""
            struct Point { x: f32, y: f32 }
            fn f(p: Point) -> f32 { p.x }
        """)
        assert "stablehlo.get_tuple_element" in out

    def test_with_update(self):
        out = codegen("""
            struct Point { x: f32, y: f32 }
            fn f(p: Point) -> Point { p with { x = 1.0 } }
        """)
        assert "get_tuple_element" in out
        assert "stablehlo.tuple" in out

    def test_nested_field_access(self):
        out = codegen("""
            struct Inner { x: f32 }
            struct Outer { inner: Inner, y: f32 }
            fn f(b: Outer) -> f32 { b.inner.x }
        """)
        # Should have multiple get_tuple_element: one to extract inner, one to extract x
        assert out.count("get_tuple_element") >= 2


class TestConv2d:
    def test_conv2d_basic(self):
        out = codegen("fn f(x: f32[1, 3, 8, 8], w: f32[16, 3, 3, 3]) -> f32[1, 16, 6, 6] { conv2d(x, w) }")
        assert "stablehlo.convolution" in out
        assert "dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]" in out
        assert "stride = [1, 1]" in out
        assert "pad = [[0, 0], [0, 0]]" in out
        assert "tensor<1x16x6x6xf32>" in out

    def test_conv2d_with_stride_pad(self):
        out = codegen("fn f(x: f32[1, 3, 8, 8], w: f32[16, 3, 3, 3]) -> f32[1, 16, 4, 4] { conv2d(x, w, 2, 1) }")
        assert "stablehlo.convolution" in out
        assert "stride = [2, 2]" in out
        assert "pad = [[1, 1], [1, 1]]" in out


class TestPool:
    def test_max_pool(self):
        out = codegen("fn f(x: f32[1, 16, 8, 8]) -> f32[1, 16, 4, 4] { max_pool(x, 2, 2, 2, 2) }")
        assert "stablehlo.reduce_window" in out
        assert "stablehlo.maximum" in out

    def test_avg_pool(self):
        out = codegen("fn f(x: f32[1, 16, 8, 8]) -> f32[1, 16, 4, 4] { avg_pool(x, 2, 2, 2, 2) }")
        assert "stablehlo.reduce_window" in out
        assert "stablehlo.add" in out
        assert "stablehlo.divide" in out


class TestAxisReductionCodegen:
    def test_sum_axis(self):
        out = codegen("fn f(x: f32[3, 4]) -> f32[3] { sum(x, 1) }")
        assert "across dimensions = [1]" in out
        assert "tensor<3x4xf32>" in out
        assert "-> tensor<3xf32>" in out

    def test_sum_axis_0(self):
        out = codegen("fn f(x: f32[3, 4]) -> f32[4] { sum(x, 0) }")
        assert "across dimensions = [0]" in out

    def test_mean_axis(self):
        out = codegen("fn f(x: f32[3, 4]) -> f32[4] { mean(x, 0) }")
        assert "stablehlo.reduce" in out
        assert "stablehlo.divide" in out
        assert "3.000000e+00" in out  # axis 0 has size 3


class TestSizeOneBroadcastCodegen:
    def test_broadcast_size1(self):
        out = codegen("fn f(x: f32[3, 1], y: f32[3, 4]) -> f32[3, 4] { x * y }")
        assert "broadcast_in_dim" in out
        assert "tensor<3x1xf32>" in out
        assert "tensor<3x4xf32>" in out

    def test_broadcast_both_sides(self):
        out = codegen("fn f(x: f32[3, 1], y: f32[1, 4]) -> f32[3, 4] { x + y }")
        # Both sides need broadcasting
        assert out.count("broadcast_in_dim") == 2


class TestStopGradientCodegen:
    def test_stop_gradient_identity(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { stop_gradient(x) }")
        # stop_gradient is identity — no extra ops, just return the arg
        assert "return %arg0" in out

    def test_stop_gradient_in_expr(self):
        out = codegen("fn f(x: f32[4], y: f32[4]) -> f32[4] { stop_gradient(x) * y }")
        # stop_gradient is identity in codegen — just multiplies x * y
        assert "stablehlo.multiply" in out


class TestWhereCodegen:
    def test_where_basic(self):
        out = codegen("fn f(m: bool[4], x: f32[4], y: f32[4]) -> f32[4] { where(m, x, y) }")
        assert "stablehlo.select" in out

    def test_where_scalar_cond(self):
        out = codegen("fn f(m: bool, x: f32[4], y: f32[4]) -> f32[4] { where(m, x, y) }")
        assert "broadcast_in_dim" in out  # scalar cond broadcast to array
        assert "stablehlo.select" in out


class TestStringCallbackCodegen:
    def test_string_label_filtered_from_operands(self):
        """String args should not appear as tensor operands in custom_call."""
        out = codegen("""
            fn f(x: f32) -> f32 {
                callback("loss", x);
                x
            }
        """)
        assert "xla_ffi_python_cpu_callback" in out
        # Only x should be an operand, not the string
        assert "tensor<f32>" in out

    def test_string_only_callback(self):
        """callback with only string args should emit no-operands custom_call."""
        out = codegen("""
            fn f(x: f32) -> f32 {
                callback("hello");
                x
            }
        """)
        assert "xla_ffi_python_cpu_callback" in out
        assert "() -> ()" in out  # no operands, no results

    def test_callback_labels_stored(self):
        """Codegen should store string labels in _callback_labels."""
        from maomi.lexer import Lexer
        from maomi.parser import Parser
        from maomi.type_checker import TypeChecker
        from maomi.codegen.stablehlo import StableHLOCodegen

        source = 'fn f(x: f32) -> f32 { callback("loss", x); x }'
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        checker = TypeChecker()
        errors = checker.check(program)
        assert errors == []
        cg = StableHLOCodegen(program, checker.type_map)
        cg.generate()
        assert cg._callback_labels == {0: ["loss"]}

    def test_callback_labels_in_compile_result(self):
        """CompileResult should include callback_labels."""
        from maomi.cli import compile_source
        result = compile_source('fn f(x: f32) -> f32 { callback("loss", x); x }')
        assert result.callback_labels == {0: ["loss"]}
        assert result.callback_count == 1


# ---------- General transpose tests ----------


class TestGeneralTranspose:
    def test_2d_shorthand(self):
        """transpose(x) on 2D still works"""
        out = codegen("fn f(x: f32[4, 8]) -> f32[8, 4] { transpose(x) }")
        assert "stablehlo.transpose" in out
        assert "dims = [1, 0]" in out

    def test_2d_explicit(self):
        """transpose(x, 1, 0) same as shorthand"""
        out = codegen("fn f(x: f32[4, 8]) -> f32[8, 4] { transpose(x, 1, 0) }")
        assert "stablehlo.transpose" in out
        assert "dims = [1, 0]" in out

    def test_3d_permutation(self):
        """transpose(x, 0, 2, 1) swaps last two axes of 3D"""
        out = codegen("fn f(x: f32[2, 4, 8]) -> f32[2, 8, 4] { transpose(x, 0, 2, 1) }")
        assert "stablehlo.transpose" in out
        assert "dims = [0, 2, 1]" in out

    def test_4d_permutation(self):
        """transpose(x, 0, 2, 1, 3) — multi-head attention axis swap"""
        out = codegen("fn f(x: f32[2, 16, 4, 8]) -> f32[2, 4, 16, 8] { transpose(x, 0, 2, 1, 3) }")
        assert "stablehlo.transpose" in out
        assert "dims = [0, 2, 1, 3]" in out

    def test_4d_full_reverse(self):
        """transpose(x, 3, 2, 1, 0) reverses all axes"""
        out = codegen("fn f(x: f32[2, 3, 4, 5]) -> f32[5, 4, 3, 2] { transpose(x, 3, 2, 1, 0) }")
        assert "stablehlo.transpose" in out
        assert "dims = [3, 2, 1, 0]" in out

    def test_transpose_in_map_general(self):
        """General transpose inside map"""
        out = codegen("""
            fn f(xs: f32[32, 2, 4, 8]) -> f32[32, 4, 2, 8] {
                map x in xs { transpose(x, 1, 0, 2) }
            }
        """)
        assert "stablehlo.transpose" in out

    def test_grad_through_general_transpose(self):
        """Gradient flows through general transpose via inverse permutation"""
        from maomi.ad import transform_grad
        src = """
            fn f(x: f32[2, 3, 4]) -> f32[2, 3, 4] {
                let t = transpose(x, 0, 2, 1);
                grad(sum(t), x)
            }
        """
        tokens = Lexer(src).tokenize()
        prog = Parser(tokens).parse()
        tc = TypeChecker()
        errors = tc.check(prog)
        assert not errors, errors
        prog = transform_grad(prog, tc.type_map)
        out = StableHLOCodegen(prog, tc.type_map).generate()
        assert "stablehlo.transpose" in out


class TestTransposeTypeErrors:
    def test_shorthand_non_2d(self):
        """transpose(x) on 3D should error"""
        tokens = Lexer("fn f(x: f32[2, 3, 4]) -> f32[4, 3, 2] { transpose(x) }").tokenize()
        prog = Parser(tokens).parse()
        tc = TypeChecker()
        errors = tc.check(prog)
        assert errors
        assert "2D" in errors[0].message

    def test_wrong_axis_count(self):
        """Wrong number of axes"""
        tokens = Lexer("fn f(x: f32[2, 3, 4]) -> f32[2, 3, 4] { transpose(x, 0, 1) }").tokenize()
        prog = Parser(tokens).parse()
        tc = TypeChecker()
        errors = tc.check(prog)
        assert errors
        assert "3 axes" in errors[0].message

    def test_duplicate_axis(self):
        """Duplicate axis in permutation"""
        tokens = Lexer("fn f(x: f32[2, 3, 4]) -> f32[2, 3, 4] { transpose(x, 0, 1, 1) }").tokenize()
        prog = Parser(tokens).parse()
        tc = TypeChecker()
        errors = tc.check(prog)
        assert errors
        assert "permutation" in errors[0].message

    def test_axis_out_of_range(self):
        """Axis index too large"""
        tokens = Lexer("fn f(x: f32[2, 3]) -> f32[3, 2] { transpose(x, 0, 5) }").tokenize()
        prog = Parser(tokens).parse()
        tc = TypeChecker()
        errors = tc.check(prog)
        assert errors
        assert "out of range" in errors[0].message


class TestUtilityBuiltins:
    def test_isfinite_scalar(self):
        out = codegen("fn f(x: f32) -> bool { isfinite(x) }")
        assert "is_finite" in out

    def test_isfinite_array(self):
        out = codegen("fn f(x: f32[4]) -> bool[4] { isfinite(x) }")
        assert "is_finite" in out

    def test_zeros_like_scalar(self):
        out = codegen("fn f(x: f32) -> f32 { zeros_like(x) }")
        assert "stablehlo.constant" in out
        assert "0.000000e+00" in out

    def test_zeros_like_array(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { zeros_like(x) }")
        assert "stablehlo.constant" in out
        assert "0.000000e+00" in out
        assert "broadcast_in_dim" in out

    def test_ones_like_scalar(self):
        out = codegen("fn f(x: f32) -> f32 { ones_like(x) }")
        assert "stablehlo.constant" in out
        assert "1.000000e+00" in out

    def test_ones_like_array(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { ones_like(x) }")
        assert "stablehlo.constant" in out
        assert "1.000000e+00" in out


class TestTwoArgElementwise:
    def test_maximum(self):
        out = codegen("fn f(x: f32, y: f32) -> f32 { maximum(x, y) }")
        assert "stablehlo.maximum" in out

    def test_minimum(self):
        out = codegen("fn f(x: f32, y: f32) -> f32 { minimum(x, y) }")
        assert "stablehlo.minimum" in out

    def test_pow(self):
        out = codegen("fn f(x: f32, y: f32) -> f32 { pow(x, y) }")
        assert "stablehlo.power" in out

    def test_maximum_arrays(self):
        out = codegen("fn f(x: f32[4], y: f32[4]) -> f32[4] { maximum(x, y) }")
        assert "stablehlo.maximum" in out

    def test_pow_broadcast(self):
        out = codegen("fn f(x: f32[4], y: f32) -> f32[4] { pow(x, y) }")
        assert "stablehlo.power" in out
        assert "broadcast_in_dim" in out
class TestClipCodegen:
    def test_clip_scalar(self):
        out = codegen("fn f(x: f32) -> f32 { clip(x, 0.0, 1.0) }")
        assert "stablehlo.clamp" in out

    def test_clip_array(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { clip(x, 0.0, 1.0) }")
        assert "stablehlo.clamp" in out

    def test_clip_broadcast(self):
        out = codegen("fn f(x: f32[4], lo: f32, hi: f32) -> f32[4] { clip(x, lo, hi) }")
        assert "stablehlo.clamp" in out
        assert "broadcast_in_dim" in out  # scalar lo/hi broadcast to array
class TestOneHot:
    def test_one_hot_scalar(self):
        out = codegen("fn f(x: i32) -> f32[5] { one_hot(x, 5) }")
        assert "stablehlo.iota" in out
        assert "stablehlo.compare" in out
        assert "stablehlo.convert" in out
        assert "tensor<5xf32>" in out

    def test_one_hot_array(self):
        out = codegen("fn f(x: i32[3]) -> f32[3, 5] { one_hot(x, 5) }")
        assert "stablehlo.iota" in out
        assert "stablehlo.compare" in out
        assert "stablehlo.convert" in out
        assert "tensor<3x5xf32>" in out

    def test_one_hot_2d(self):
        out = codegen("fn f(x: i32[3, 4]) -> f32[3, 4, 5] { one_hot(x, 5) }")
        assert "stablehlo.iota" in out
        assert "dim = 2" in out
        assert "tensor<3x4x5xf32>" in out
class TestLogsumexp:
    def test_logsumexp_basic(self):
        out = codegen("fn f(x: f32[4]) -> f32 { logsumexp(x) }")
        assert "stablehlo.exponential" in out
        assert "stablehlo.log" in out

    def test_logsumexp_axis(self):
        out = codegen("fn f(x: f32[3, 4]) -> f32[3] { logsumexp(x, axis=1) }")
        assert "stablehlo.exponential" in out
        assert "stablehlo.log" in out
        assert "stablehlo.subtract" in out

    def test_logsumexp_keepdims(self):
        out = codegen("fn f(x: f32[3, 4]) -> f32[3, 1] { logsumexp(x, axis=1, keepdims=true) }")
        assert "stablehlo.exponential" in out
        assert "stablehlo.log" in out
        assert "stablehlo.reshape" in out
class TestBF16:
    def test_bf16_identity(self):
        out = codegen("fn f(x: bf16) -> bf16 { x }")
        assert "bf16" in out

    def test_bf16_array(self):
        out = codegen("fn f(x: bf16[4]) -> bf16[4] { x }")
        assert "tensor<4xbf16>" in out

    def test_cast_f32_to_bf16(self):
        out = codegen("fn f(x: f32) -> bf16 { cast(x, bf16) }")
        assert "stablehlo.convert" in out
        assert "bf16" in out

    def test_cast_bf16_to_f32(self):
        out = codegen("fn f(x: bf16) -> f32 { cast(x, f32) }")
        assert "stablehlo.convert" in out
        assert "f32" in out

    def test_bf16_add(self):
        out = codegen("fn f(x: bf16, y: bf16) -> bf16 { x + y }")
        assert "stablehlo.add" in out
        assert "bf16" in out

    def test_bf16_array_add(self):
        out = codegen("fn f(x: bf16[4], y: bf16[4]) -> bf16[4] { x + y }")
        assert "stablehlo.add" in out
        assert "tensor<4xbf16>" in out

    def test_bf16_mul(self):
        out = codegen("fn f(x: bf16, y: bf16) -> bf16 { x * y }")
        assert "stablehlo.multiply" in out
        assert "bf16" in out


class TestValueAndGrad:
    def _codegen_with_ad(self, source: str) -> str:
        from maomi.ad import transform_grad
        tokens = Lexer(source).tokenize()
        prog = Parser(tokens).parse()
        tc = TypeChecker()
        errors = tc.check(prog)
        assert not errors, f"Type errors: {[e.message for e in errors]}"
        prog = transform_grad(prog, tc.type_map)
        return StableHLOCodegen(prog, tc.type_map).generate()

    def test_value_and_grad_produces_tuple(self):
        out = self._codegen_with_ad("""
            fn f(x: f32) -> f32 {
                let vg = value_and_grad(x * x, x);
                vg.value
            }
        """)
        assert "stablehlo.tuple" in out

    def test_value_and_grad_get_gradient(self):
        out = self._codegen_with_ad("""
            fn f(x: f32) -> f32 {
                let vg = value_and_grad(x * x, x);
                vg.gradient
            }
        """)
        assert "get_tuple_element" in out

    def test_value_and_grad_both_fields(self):
        out = self._codegen_with_ad("""
            fn f(x: f32) -> f32 {
                let vg = value_and_grad(x * x, x);
                vg.value + vg.gradient
            }
        """)
        assert "stablehlo.tuple" in out
        assert "stablehlo.add" in out

    def test_destructure_value_and_grad(self):
        out = self._codegen_with_ad("""
            fn f(x: f32) -> f32 {
                let { value, gradient } = value_and_grad(x * x, x);
                value + gradient
            }
        """)
        assert "stablehlo.tuple" in out
        assert "stablehlo.add" in out

    def test_destructure_rebind(self):
        out = self._codegen_with_ad("""
            fn f(x: f32) -> f32 {
                let { value: v, gradient: g } = value_and_grad(x * x, x);
                v + g
            }
        """)
        assert "stablehlo.tuple" in out

    def test_destructure_partial(self):
        out = self._codegen_with_ad("""
            fn f(x: f32) -> f32 {
                let { gradient } = value_and_grad(x * x, x);
                gradient
            }
        """)
        assert "get_tuple_element" in out


class TestDestructureStruct:
    def test_basic(self):
        out = codegen("""
            struct S { x: f32, y: f32 }
            fn f() -> f32 {
                let { x, y } = S { x: 1.0, y: 2.0 };
                x + y
            }
        """)
        assert "stablehlo.add" in out

    def test_rebind(self):
        out = codegen("""
            struct S { x: f32, y: f32 }
            fn f() -> f32 {
                let { x: a, y: b } = S { x: 1.0, y: 2.0 };
                a + b
            }
        """)
        assert "stablehlo.add" in out
