"""Tests for array creation builtins: arange, linspace, eye."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.codegen.stablehlo import StableHLOCodegen
from maomi.errors import MaomiError


# -- Helpers --

def codegen(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    return StableHLOCodegen(program, checker.type_map).generate()


def type_check_fails(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert len(errors) > 0, "Expected type error"
    return errors[0].message


# ============================================================
# arange(start, stop, step)
# ============================================================

class TestArange:
    def test_basic(self):
        out = codegen("fn f() -> i32[5] { arange(0, 10, 2) }")
        assert "stablehlo.iota dim = 0" in out
        assert "tensor<5xi32>" in out

    def test_step_one(self):
        out = codegen("fn f() -> i32[4] { arange(1, 5, 1) }")
        assert "stablehlo.iota dim = 0" in out
        assert "tensor<4xi32>" in out

    def test_start_zero_step_one(self):
        # arange(0, 3, 1) => i32[3] = [0, 1, 2], just iota
        out = codegen("fn f() -> i32[3] { arange(0, 3, 1) }")
        assert "stablehlo.iota dim = 0" in out
        # No multiply or add needed for start=0, step=1
        assert "stablehlo.multiply" not in out
        assert "stablehlo.add" not in out

    def test_nonliteral_args_fail(self):
        msg = type_check_fails("fn f(x: i32) -> i32[5] { arange(x, 10, 2) }")
        assert "integer literal" in msg

    def test_step_zero_fails(self):
        msg = type_check_fails("fn f() -> i32[5] { arange(0, 10, 0) }")
        assert "zero" in msg

    def test_wrong_arg_count_fails(self):
        msg = type_check_fails("fn f() -> i32[5] { arange(0, 10) }")
        assert "3 arguments" in msg

    def test_empty_range_fails(self):
        msg = type_check_fails("fn f() -> i32[1] { arange(10, 0, 1) }")
        assert "empty" in msg


# ============================================================
# linspace(start, stop, n)
# ============================================================

class TestLinspace:
    def test_basic(self):
        out = codegen("fn f() -> f32[5] { linspace(0.0, 1.0, 5) }")
        assert "stablehlo.iota dim = 0" in out
        assert "stablehlo.convert" in out
        assert "stablehlo.divide" in out
        assert "tensor<5xf32>" in out

    def test_single_element(self):
        # linspace(0.0, 1.0, 1) => f32[1] = [0.0]
        out = codegen("fn f() -> f32[1] { linspace(0.0, 1.0, 1) }")
        assert "tensor<1xf32>" in out
        # Edge case: n=1, just constant
        assert "stablehlo.constant" in out

    def test_negative_range(self):
        out = codegen("fn f() -> f32[3] { linspace(1.0, -1.0, 3) }")
        assert "tensor<3xf32>" in out

    def test_int_start_fails(self):
        msg = type_check_fails("fn f() -> f32[5] { linspace(0, 1.0, 5) }")
        assert "float literal" in msg

    def test_int_stop_fails(self):
        msg = type_check_fails("fn f() -> f32[5] { linspace(0.0, 1, 5) }")
        assert "float literal" in msg

    def test_n_zero_fails(self):
        msg = type_check_fails("fn f() -> f32[1] { linspace(0.0, 1.0, 0) }")
        assert "positive" in msg

    def test_wrong_arg_count_fails(self):
        msg = type_check_fails("fn f() -> f32[5] { linspace(0.0, 1.0) }")
        assert "3 arguments" in msg


# ============================================================
# eye(n) / eye(n, m)
# ============================================================

class TestEye:
    def test_square(self):
        out = codegen("fn f() -> f32[3,3] { eye(3) }")
        assert "stablehlo.iota dim = 0" in out
        assert "stablehlo.iota dim = 1" in out
        assert "stablehlo.compare EQ" in out
        assert "stablehlo.convert" in out
        assert "tensor<3x3xf32>" in out

    def test_rectangular(self):
        out = codegen("fn f() -> f32[2,3] { eye(2, 3) }")
        assert "stablehlo.iota dim = 0" in out
        assert "stablehlo.iota dim = 1" in out
        assert "tensor<2x3xi32>" in out
        assert "tensor<2x3xf32>" in out

    def test_1x1(self):
        out = codegen("fn f() -> f32[1,1] { eye(1) }")
        assert "tensor<1x1xf32>" in out

    def test_zero_dim_fails(self):
        msg = type_check_fails("fn f() -> f32[1,1] { eye(0) }")
        assert "positive" in msg

    def test_nonliteral_fails(self):
        msg = type_check_fails("fn f(n: i32) -> f32[3,3] { eye(n) }")
        assert "integer literal" in msg

    def test_too_many_args_fails(self):
        msg = type_check_fails("fn f() -> f32[3,3] { eye(3, 3, 3) }")
        assert "1 or 2" in msg
