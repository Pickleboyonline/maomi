"""Tests for the standard library modules (nn, optim, math)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maomi.cli import compile_source
from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.ad import transform_grad
from maomi.codegen.stablehlo import StableHLOCodegen


# -- Helpers --

def compile_ok(source: str) -> str:
    result = compile_source(source, filename="/tmp/test.mao")
    return result.mlir_text


def compile_ad(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    from maomi.resolver import resolve
    program = resolve(program, "/tmp/test.mao")
    tc = TypeChecker()
    errors = tc.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    program = transform_grad(program, tc.type_map)
    return StableHLOCodegen(program, tc.type_map).generate()


# -- Import resolution --

class TestNnImport:
    def test_qualified_import(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[3]) -> f32[3] { nn.relu(x) }
        """)
        assert "stablehlo.select" in out  # where generates select

    def test_selective_import(self):
        out = compile_ok("""
            from nn import { relu };
            fn main(x: f32[3]) -> f32[3] { relu(x) }
        """)
        assert "stablehlo.select" in out

    def test_selective_import_multiple(self):
        out = compile_ok("""
            from nn import { relu, sigmoid };
            fn main(x: f32[3]) -> f32[3] { sigmoid(relu(x)) }
        """)
        assert "stablehlo.select" in out
        assert "stablehlo.exponential" in out


# -- nn.relu --

class TestNnRelu:
    def test_relu_1d(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[3]) -> f32[3] { nn.relu(x) }
        """)
        assert "tensor<3xf32>" in out

    def test_relu_2d(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[3, 4]) -> f32[3, 4] { nn.relu(x) }
        """)
        assert "tensor<3x4xf32>" in out

    def test_relu_3d(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[2, 3, 4]) -> f32[2, 3, 4] { nn.relu(x) }
        """)
        assert "tensor<2x3x4xf32>" in out


# -- nn.sigmoid --

class TestNnSigmoid:
    def test_sigmoid_1d(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[3]) -> f32[3] { nn.sigmoid(x) }
        """)
        assert "stablehlo.exponential" in out

    def test_sigmoid_2d(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[3, 4]) -> f32[3, 4] { nn.sigmoid(x) }
        """)
        assert "tensor<3x4xf32>" in out


# -- nn.softmax --

class TestNnSoftmax:
    def test_softmax_1d(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[5]) -> f32[5] { nn.softmax(x, axis=0) }
        """)
        assert "stablehlo.exponential" in out
        assert "tensor<5xf32>" in out

    def test_softmax_different_size(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[10]) -> f32[10] { nn.softmax(x, axis=0) }
        """)
        assert "tensor<10xf32>" in out

    def test_softmax_2d(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[4, 8]) -> f32[4, 8] { nn.softmax(x, axis=1) }
        """)
        assert "stablehlo.exponential" in out
        assert "tensor<4x8xf32>" in out


# -- nn.log_softmax --

class TestNnLogSoftmax:
    def test_log_softmax_1d(self):
        out = compile_ok("""
            import nn;
            fn main(x: f32[5]) -> f32[5] { nn.log_softmax(x, axis=0) }
        """)
        assert "stablehlo.log" in out
        assert "stablehlo.exponential" in out


# -- AD through nn functions --

class TestNnAD:
    def test_grad_relu(self):
        out = compile_ad("""
            import nn;
            fn loss(x: f32[3]) -> f32 { sum(nn.relu(x)) }
            fn main(x: f32[3]) -> f32[3] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_grad_sigmoid(self):
        out = compile_ad("""
            import nn;
            fn loss(x: f32[3]) -> f32 { sum(nn.sigmoid(x)) }
            fn main(x: f32[3]) -> f32[3] { grad(loss(x), x) }
        """)
        assert out is not None


# -- math module: import resolution --

class TestMathImport:
    def test_qualified_import(self):
        out = compile_ok("""
            import math;
            fn main(x: f32[3,4]) -> f32[3] {
                math.var(x, axis=1)
            }
        """)
        assert out is not None

    def test_selective_import_var(self):
        out = compile_ok("""
            from math import { var };
            fn main(x: f32[3,4]) -> f32[3] {
                var(x, axis=1)
            }
        """)
        assert out is not None

    def test_selective_import_std(self):
        out = compile_ok("""
            from math import { std };
            fn main(x: f32[3,4]) -> f32[3] {
                std(x, axis=1)
            }
        """)
        assert out is not None

    def test_selective_import_normalize(self):
        out = compile_ok("""
            from math import { normalize };
            fn main(x: f32[3,4]) -> f32[3,4] {
                normalize(x, axis=1)
            }
        """)
        assert out is not None

    def test_selective_import_multiple(self):
        out = compile_ok("""
            from math import { var, std, normalize };
            fn main(x: f32[3,4]) -> f32[3] {
                std(x, axis=1)
            }
        """)
        assert out is not None


# -- math.var --

class TestMathVar:
    def test_var_2d_axis1(self):
        out = compile_ok("""
            import math;
            fn main(x: f32[3,4]) -> f32[3] {
                math.var(x, axis=1)
            }
        """)
        assert "tensor<3xf32>" in out

    def test_var_2d_axis0(self):
        out = compile_ok("""
            import math;
            fn main(x: f32[3,4]) -> f32[4] {
                math.var(x, axis=0)
            }
        """)
        assert "tensor<4xf32>" in out


# -- math.std --

class TestMathStd:
    def test_std_2d_axis1(self):
        out = compile_ok("""
            import math;
            fn main(x: f32[3,4]) -> f32[3] {
                math.std(x, axis=1)
            }
        """)
        assert "tensor<3xf32>" in out
        assert "stablehlo.sqrt" in out


# -- math.normalize --

class TestMathNormalize:
    def test_normalize_2d_axis1(self):
        out = compile_ok("""
            import math;
            fn main(x: f32[3,4]) -> f32[3,4] {
                math.normalize(x, axis=1)
            }
        """)
        assert "tensor<3x4xf32>" in out
        assert "stablehlo.sqrt" in out

    def test_normalize_2d_axis0(self):
        out = compile_ok("""
            import math;
            fn main(x: f32[3,4]) -> f32[3,4] {
                math.normalize(x, axis=0)
            }
        """)
        assert "tensor<3x4xf32>" in out


# -- AD through math functions --

class TestMathAD:
    def test_grad_var(self):
        out = compile_ad("""
            import math;
            fn loss(x: f32[3,4]) -> f32 { sum(math.var(x, axis=1)) }
            fn main(x: f32[3,4]) -> f32[3,4] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_grad_normalize(self):
        out = compile_ad("""
            import math;
            fn loss(x: f32[3,4]) -> f32 { sum(math.normalize(x, axis=1)) }
            fn main(x: f32[3,4]) -> f32[3,4] { grad(loss(x), x) }
        """)
        assert out is not None
