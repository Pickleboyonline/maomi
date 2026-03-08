"""Tests for f32[..] shape wildcard (rank polymorphism via monomorphization)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.ad import transform_grad
from maomi.codegen.stablehlo import StableHLOCodegen
from maomi.types import ArrayType, ScalarType


# -- Helpers --

def check(source: str) -> list[str]:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    errors = TypeChecker().check(program)
    return [e.message for e in errors]


def check_ok(source: str):
    errors = check(source)
    assert errors == [], f"Expected no errors, got: {errors}"


def codegen(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    tc = TypeChecker()
    errors = tc.check(program)
    assert errors == [], f"Expected no errors, got: {[e.message for e in errors]}"
    return StableHLOCodegen(program, tc.type_map).generate()


def ad_codegen(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    tc = TypeChecker()
    errors = tc.check(program)
    assert errors == [], f"Expected no errors, got: {[e.message for e in errors]}"
    program = transform_grad(program, tc.type_map)
    return StableHLOCodegen(program, tc.type_map).generate()


# -- Parsing --

class TestWildcardParsing:
    def test_parse_wildcard_type(self):
        tokens = Lexer("fn f(x: f32[..]) -> f32[..] { x }").tokenize()
        prog = Parser(tokens).parse()
        fn = prog.functions[0]
        assert fn.params[0].type_annotation.wildcard is True
        assert fn.params[0].type_annotation.dims is None
        assert fn.return_type.wildcard is True

    def test_parse_regular_type_not_wildcard(self):
        tokens = Lexer("fn f(x: f32[3]) -> f32[3] { x }").tokenize()
        prog = Parser(tokens).parse()
        fn = prog.functions[0]
        assert fn.params[0].type_annotation.wildcard is False


# -- Type checking --

class TestWildcardTypeCheck:
    def test_wildcard_identity_1d(self):
        check_ok("""
            fn f(x: f32[..]) -> f32[..] { x }
            fn main(a: f32[3]) -> f32[3] { f(a) }
        """)

    def test_wildcard_identity_2d(self):
        check_ok("""
            fn f(x: f32[..]) -> f32[..] { x }
            fn main(a: f32[3, 4]) -> f32[3, 4] { f(a) }
        """)

    def test_wildcard_identity_scalar(self):
        check_ok("""
            fn f(x: f32[..]) -> f32[..] { x }
            fn main(a: f32) -> f32 { f(a) }
        """)

    def test_wildcard_with_where(self):
        check_ok("""
            fn relu(x: f32[..]) -> f32[..] { where(x > 0.0, x, 0.0) }
            fn main(a: f32[3, 4]) -> f32[3, 4] { relu(a) }
        """)

    def test_wildcard_with_exp(self):
        check_ok("""
            fn sigmoid(x: f32[..]) -> f32[..] { 1.0 / (1.0 + exp(0.0 - x)) }
            fn main(a: f32[2, 3]) -> f32[2, 3] { sigmoid(a) }
        """)

    def test_wildcard_dtype_mismatch(self):
        errors = check("""
            fn f(x: f32[..]) -> f32[..] { x }
            fn main(a: i32[3]) -> i32[3] { f(a) }
        """)
        assert len(errors) > 0
        assert "f32[..]" in errors[0]

    def test_wildcard_shape_mismatch_between_params(self):
        errors = check("""
            fn f(x: f32[..], y: f32[..]) -> f32[..] { x + y }
            fn main(a: f32[3], b: f32[4]) -> f32[3] { f(a, b) }
        """)
        assert len(errors) > 0
        assert "shape mismatch" in errors[0]

    def test_wildcard_multiple_monomorphizations(self):
        check_ok("""
            fn f(x: f32[..]) -> f32[..] { x }
            fn main(a: f32[3], b: f32[4, 5]) -> f32[3] {
                let ra = f(a);
                let rb = f(b);
                ra
            }
        """)

    def test_wildcard_wrong_arg_count(self):
        errors = check("""
            fn f(x: f32[..]) -> f32[..] { x }
            fn main(a: f32[3], b: f32[3]) -> f32[3] { f(a, b) }
        """)
        assert len(errors) > 0
        assert "expects 1 arguments" in errors[0]


# -- Codegen --

class TestWildcardCodegen:
    def test_monomorphized_function_generated(self):
        out = codegen("""
            fn f(x: f32[..]) -> f32[..] { x }
            fn main(a: f32[3]) -> f32[3] { f(a) }
        """)
        assert "tensor<3xf32>" in out
        # Original generic function should NOT be generated
        assert "func.func @f(" not in out

    def test_multiple_monomorphizations(self):
        out = codegen("""
            fn f(x: f32[..]) -> f32[..] { x }
            fn main(a: f32[3], b: f32[4, 5]) -> f32[3] {
                let ra = f(a);
                let rb = f(b);
                ra
            }
        """)
        assert "tensor<3xf32>" in out
        assert "tensor<4x5xf32>" in out

    def test_wildcard_relu_codegen(self):
        out = codegen("""
            fn relu(x: f32[..]) -> f32[..] { where(x > 0.0, x, 0.0) }
            fn main(a: f32[3]) -> f32[3] { relu(a) }
        """)
        assert "stablehlo.compare" in out  # x > 0.0
        assert "stablehlo.select" in out   # where


# -- AD --

class TestWildcardAD:
    def test_grad_through_wildcard_relu(self):
        out = ad_codegen("""
            fn relu(x: f32[..]) -> f32[..] { where(x > 0.0, x, 0.0) }
            fn loss(x: f32[3]) -> f32 { sum(relu(x)) }
            fn main(x: f32[3]) -> f32[3] { grad(loss(x), x) }
        """)
        assert out is not None

    def test_grad_through_wildcard_sigmoid(self):
        out = ad_codegen("""
            fn sigmoid(x: f32[..]) -> f32[..] { 1.0 / (1.0 + exp(0.0 - x)) }
            fn loss(x: f32[3]) -> f32 { sum(sigmoid(x)) }
            fn main(x: f32[3]) -> f32[3] { grad(loss(x), x) }
        """)
        assert out is not None


# -- Symbolic dim monomorphization --

class TestSymbolicDimMonomorphization:
    def test_symbolic_dim_function_called(self):
        """fn f(x: f32[N]) should be monomorphized when called with f32[3]."""
        check_ok("""
            fn f(x: f32[N]) -> f32 { mean(x) }
            fn main(a: f32[3]) -> f32 { f(a) }
        """)

    def test_symbolic_dim_codegen(self):
        out = codegen("""
            fn f(x: f32[N]) -> f32 { mean(x) }
            fn main(a: f32[3]) -> f32 { f(a) }
        """)
        assert "tensor<3xf32>" in out
