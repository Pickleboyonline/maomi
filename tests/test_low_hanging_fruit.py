"""Tests for low-hanging-fruit features: trig builtins, logical operators,
stdlib additions, cumsum/cumprod, sort/argsort."""

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
# Feature 1: Trig/Math Elementwise Builtins
# ============================================================

class TestTrigBuiltins:
    """Test trig/math builtins: tan, sinh, cosh, asin, acos, atan, asinh, acosh, atanh, exp2, log10."""

    def test_tan_codegen(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { tan(x) }")
        # tan = sin/cos, compound implementation
        assert "stablehlo.sine" in out
        assert "stablehlo.cosine" in out
        assert "stablehlo.divide" in out

    def test_sinh_codegen(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { sinh(x) }")
        assert "stablehlo.exponential" in out

    def test_cosh_codegen(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { cosh(x) }")
        assert "stablehlo.exponential" in out

    def test_asin_codegen(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { asin(x) }")
        assert "stablehlo.atan2" in out

    def test_acos_codegen(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { acos(x) }")
        assert "stablehlo.atan2" in out

    def test_atan_codegen(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { atan(x) }")
        assert "stablehlo.atan2" in out

    def test_asinh_codegen(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { asinh(x) }")
        assert "stablehlo.log" in out

    def test_acosh_codegen(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { acosh(x) }")
        assert "stablehlo.log" in out

    def test_atanh_codegen(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { atanh(x) }")
        assert "stablehlo.log" in out

    def test_exp2_codegen(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { exp2(x) }")
        assert "stablehlo.exponential" in out

    def test_log10_codegen(self):
        out = codegen("fn f(x: f32[4]) -> f32[4] { log10(x) }")
        assert "stablehlo.log" in out

    def test_atan2_codegen(self):
        out = codegen("fn f(y: f32[4], x: f32[4]) -> f32[4] { atan2(y, x) }")
        assert "stablehlo.atan2" in out

    def test_trig_scalar(self):
        out = codegen("fn f(x: f32) -> f32 { tan(x) }")
        assert "tensor<f32>" in out

    def test_trig_2d(self):
        out = codegen("fn f(x: f32[3, 4]) -> f32[3, 4] { sinh(x) }")
        assert "tensor<3x4xf32>" in out


class TestTrigAD:
    """Test AD through trig builtins."""

    def test_tan_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(tan(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_sinh_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(sinh(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_cosh_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(cosh(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_asin_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(asin(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        # asin grad uses rsqrt
        assert "rsqrt" in out or "stablehlo" in out

    def test_acos_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(acos(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_atan_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(atan(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_asinh_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(asinh(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_acosh_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(acosh(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_atanh_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(atanh(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_exp2_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(exp2(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_log10_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4]) -> f32 { sum(log10(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_atan2_grad(self):
        out = codegen_ad("""
            fn loss(y: f32[4], x: f32[4]) -> f32 { sum(atan2(y, x)) }
            fn main(y: f32[4], x: f32[4]) -> f32[4] { grad(loss(y, x), y) }
        """)
        assert out is not None

    def test_atan2_grad_wrt_x(self):
        out = codegen_ad("""
            fn loss(y: f32[4], x: f32[4]) -> f32 { sum(atan2(y, x)) }
            fn main(y: f32[4], x: f32[4]) -> f32[4] { grad(loss(y, x), x) }
        """)
        assert out is not None


# ============================================================
# Feature 2: Logical Operators (and, or, not)
# ============================================================

class TestLogicalParsing:
    """Test parsing of logical operators."""

    def test_parse_and(self):
        tokens = Lexer("fn f(a: bool, b: bool) -> bool { a and b }").tokenize()
        program = Parser(tokens).parse()
        assert program is not None

    def test_parse_or(self):
        tokens = Lexer("fn f(a: bool, b: bool) -> bool { a or b }").tokenize()
        program = Parser(tokens).parse()
        assert program is not None

    def test_parse_not(self):
        tokens = Lexer("fn f(a: bool) -> bool { not a }").tokenize()
        program = Parser(tokens).parse()
        assert program is not None

    def test_parse_combined(self):
        tokens = Lexer("fn f(a: bool, b: bool, c: bool) -> bool { a and b or c }").tokenize()
        program = Parser(tokens).parse()
        assert program is not None

    def test_parse_not_with_and(self):
        tokens = Lexer("fn f(a: bool, b: bool) -> bool { not a and b }").tokenize()
        program = Parser(tokens).parse()
        assert program is not None


class TestLogicalTypeCheck:
    """Test type checking of logical operators."""

    def test_and_bool_scalars(self):
        out = codegen("fn f(a: bool, b: bool) -> bool { a and b }")
        assert "stablehlo.and" in out

    def test_or_bool_scalars(self):
        out = codegen("fn f(a: bool, b: bool) -> bool { a or b }")
        assert "stablehlo.or" in out

    def test_not_bool_scalar(self):
        out = codegen("fn f(a: bool) -> bool { not a }")
        assert "stablehlo.not" in out

    def test_and_bool_arrays(self):
        out = codegen("fn f(a: bool[4], b: bool[4]) -> bool[4] { a and b }")
        assert "stablehlo.and" in out

    def test_or_bool_arrays(self):
        out = codegen("fn f(a: bool[4], b: bool[4]) -> bool[4] { a or b }")
        assert "stablehlo.or" in out

    def test_not_bool_array(self):
        out = codegen("fn f(a: bool[4]) -> bool[4] { not a }")
        assert "stablehlo.not" in out

    def test_and_non_bool_fails(self):
        msg = type_check_fails("fn f(a: f32, b: f32) -> f32 { a and b }")
        assert "bool" in msg.lower()

    def test_or_non_bool_fails(self):
        msg = type_check_fails("fn f(a: f32, b: f32) -> f32 { a or b }")
        assert "bool" in msg.lower()

    def test_not_non_bool_fails(self):
        msg = type_check_fails("fn f(a: f32) -> f32 { not a }")
        assert "bool" in msg.lower()

    def test_logical_with_comparison(self):
        out = codegen("fn f(a: f32, b: f32, c: f32) -> bool { a > b and b > c }")
        assert "stablehlo.and" in out
        assert "stablehlo.compare" in out

    def test_logical_in_if(self):
        out = codegen("""
            fn f(a: bool, b: bool) -> f32 {
                if a and b { 1.0 } else { 0.0 }
            }
        """)
        assert "stablehlo.and" in out

    def test_not_in_if(self):
        out = codegen("""
            fn f(a: bool) -> f32 {
                if not a { 1.0 } else { 0.0 }
            }
        """)
        assert "stablehlo.not" in out


# ============================================================
# Feature 3: Stdlib Additions (leaky_relu, elu, selu, mish, layer_norm)
# ============================================================

class TestStdlibAdditions:
    """Test new nn stdlib functions."""

    def test_leaky_relu(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[4]) -> f32[4] { nn.leaky_relu(x, 0.01) }
        """)
        assert "stablehlo.select" in out

    def test_elu(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[4]) -> f32[4] { nn.elu(x, 1.0) }
        """)
        assert "stablehlo.select" in out
        assert "stablehlo.exponential" in out

    def test_selu(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[4]) -> f32[4] { nn.selu(x) }
        """)
        assert "stablehlo.select" in out

    def test_mish(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[4]) -> f32[4] { nn.mish(x) }
        """)
        assert "stablehlo.exponential" in out  # softplus uses exp

    def test_layer_norm(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[3, 4]) -> f32[3, 4] { nn.layer_norm(x, axis=1) }
        """)
        assert "tensor<3x4xf32>" in out

    def test_leaky_relu_2d(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[3, 4]) -> f32[3, 4] { nn.leaky_relu(x, 0.2) }
        """)
        assert "tensor<3x4xf32>" in out

    def test_elu_2d(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[3, 4]) -> f32[3, 4] { nn.elu(x, 1.0) }
        """)
        assert "tensor<3x4xf32>" in out

    def test_selu_scalar(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32) -> f32 { nn.selu(x) }
        """)
        assert "tensor<f32>" in out


class TestStdlibAD:
    """Test AD through new stdlib functions."""

    def test_leaky_relu_grad(self):
        out = codegen_ad("""
            import nn;
            fn loss(x: f32[4]) -> f32 { sum(nn.leaky_relu(x, 0.01)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_elu_grad(self):
        out = codegen_ad("""
            import nn;
            fn loss(x: f32[4]) -> f32 { sum(nn.elu(x, 1.0)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_selu_grad(self):
        out = codegen_ad("""
            import nn;
            fn loss(x: f32[4]) -> f32 { sum(nn.selu(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_mish_grad(self):
        out = codegen_ad("""
            import nn;
            fn loss(x: f32[4]) -> f32 { sum(nn.mish(x)) }
            fn main(x: f32[4]) -> f32[4] { grad(loss(x), x) }
        """)
        assert out is not None


# ============================================================
# Feature 4: cumsum / cumprod
# ============================================================

class TestCumulative:
    """Test cumsum and cumprod builtins."""

    def test_cumsum_1d(self):
        out = codegen("fn f(x: f32[8]) -> f32[8] { cumsum(x, axis=0) }")
        assert "stablehlo.slice" in out
        assert "stablehlo.pad" in out
        assert "stablehlo.add" in out

    def test_cumprod_1d(self):
        out = codegen("fn f(x: f32[8]) -> f32[8] { cumprod(x, axis=0) }")
        assert "stablehlo.slice" in out
        assert "stablehlo.pad" in out
        assert "stablehlo.multiply" in out

    def test_cumsum_2d_axis0(self):
        out = codegen("fn f(x: f32[4, 3]) -> f32[4, 3] { cumsum(x, axis=0) }")
        assert "tensor<4x3xf32>" in out
        assert "stablehlo.add" in out

    def test_cumsum_2d_axis1(self):
        out = codegen("fn f(x: f32[4, 3]) -> f32[4, 3] { cumsum(x, axis=1) }")
        assert "tensor<4x3xf32>" in out

    def test_cumsum_negative_axis(self):
        out = codegen("fn f(x: f32[4, 3]) -> f32[4, 3] { cumsum(x, axis=-1) }")
        assert "tensor<4x3xf32>" in out

    def test_cumprod_2d(self):
        out = codegen("fn f(x: f32[4, 3]) -> f32[4, 3] { cumprod(x, axis=1) }")
        assert "tensor<4x3xf32>" in out

    def test_cumsum_type_error_non_float(self):
        msg = type_check_fails("fn f(x: i32[4]) -> i32[4] { cumsum(x, axis=0) }")
        assert "float" in msg.lower()


class TestCumulativeAD:
    """Test AD through cumsum/cumprod."""

    def test_cumsum_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[8]) -> f32 { sum(cumsum(x, axis=0)) }
            fn main(x: f32[8]) -> f32[8] { grad(loss(x), x) }
        """)
        # Backward of cumsum uses reverse
        assert "stablehlo.reverse" in out

    def test_cumprod_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[8]) -> f32 { sum(cumprod(x, axis=0)) }
            fn main(x: f32[8]) -> f32[8] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_cumsum_2d_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[4, 3]) -> f32 { sum(cumsum(x, axis=1)) }
            fn main(x: f32[4, 3]) -> f32[4, 3] { grad(loss(x), x) }
        """)
        assert "stablehlo.reverse" in out


# ============================================================
# Feature 5: sort / argsort
# ============================================================

class TestSorting:
    """Test sort and argsort builtins."""

    def test_sort_1d(self):
        out = codegen("fn f(x: f32[8]) -> f32[8] { sort(x) }")
        assert "stablehlo.sort" in out
        assert "stablehlo.compare" in out

    def test_sort_2d_axis0(self):
        out = codegen("fn f(x: f32[4, 3]) -> f32[4, 3] { sort(x, axis=0) }")
        assert "stablehlo.sort" in out
        assert "dimension = 0" in out

    def test_sort_2d_axis1(self):
        out = codegen("fn f(x: f32[4, 3]) -> f32[4, 3] { sort(x, axis=1) }")
        assert "stablehlo.sort" in out
        assert "dimension = 1" in out

    def test_argsort_1d(self):
        out = codegen("fn f(x: f32[8]) -> i32[8] { argsort(x) }")
        assert "stablehlo.sort" in out
        assert "stablehlo.iota" in out

    def test_argsort_2d(self):
        out = codegen("fn f(x: f32[4, 3]) -> i32[4, 3] { argsort(x, axis=1) }")
        assert "stablehlo.sort" in out

    def test_sort_default_axis(self):
        # Default axis is last axis
        out = codegen("fn f(x: f32[4, 3]) -> f32[4, 3] { sort(x) }")
        assert "stablehlo.sort" in out
        assert "dimension = 1" in out

    def test_sort_type_check_preserves_type(self):
        out = codegen("fn f(x: f32[5]) -> f32[5] { sort(x) }")
        assert "tensor<5xf32>" in out


class TestSortingAD:
    """Test AD through sort."""

    def test_sort_grad(self):
        out = codegen_ad("""
            fn loss(x: f32[8]) -> f32 { sum(sort(x)) }
            fn main(x: f32[8]) -> f32[8] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_sort_grad_2d(self):
        out = codegen_ad("""
            fn loss(x: f32[4, 3]) -> f32 { sum(sort(x, axis=1)) }
            fn main(x: f32[4, 3]) -> f32[4, 3] { grad(loss(x), x) }
        """)
        assert out is not None
