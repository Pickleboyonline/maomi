"""Tests for array indexing and slicing."""
import pytest

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.codegen_stablehlo import StableHLOCodegen
from maomi.ad import transform_grad
from maomi.ast_nodes import (
    IndexExpr,
    IndexComponent,
    IntLiteral,
    Identifier,
    Program,
    UnaryOp,
)


def _parse(src: str) -> Program:
    tokens = Lexer(src).tokenize()
    return Parser(tokens).parse()


def _check(src: str):
    prog = _parse(src)
    tc = TypeChecker()
    errors = tc.check(prog)
    return prog, tc, errors


def _compile(src: str) -> str:
    prog = _parse(src)
    tc = TypeChecker()
    errors = tc.check(prog)
    assert not errors, errors
    prog = transform_grad(prog, tc.type_map)
    gen = StableHLOCodegen(prog, tc.type_map)
    return gen.generate()


# ---------- Parser tests ----------


class TestParserIndexing:
    def test_single_index(self):
        prog = _parse("fn f(x: f32[10]) -> f32 { x[0] }")
        body_expr = prog.functions[0].body.expr
        assert isinstance(body_expr, IndexExpr)
        assert len(body_expr.indices) == 1
        assert body_expr.indices[0].kind == "single"
        assert isinstance(body_expr.indices[0].value, IntLiteral)
        assert body_expr.indices[0].value.value == 0

    def test_variable_index(self):
        prog = _parse("fn f(x: f32[10], i: i32) -> f32 { x[i] }")
        body_expr = prog.functions[0].body.expr
        assert isinstance(body_expr, IndexExpr)
        assert body_expr.indices[0].kind == "single"
        assert isinstance(body_expr.indices[0].value, Identifier)
        assert body_expr.indices[0].value.name == "i"

    def test_slice(self):
        prog = _parse("fn f(x: f32[10]) -> f32[3] { x[1:4] }")
        body_expr = prog.functions[0].body.expr
        assert isinstance(body_expr, IndexExpr)
        assert len(body_expr.indices) == 1
        assert body_expr.indices[0].kind == "slice"
        assert isinstance(body_expr.indices[0].start, IntLiteral)
        assert body_expr.indices[0].start.value == 1
        assert isinstance(body_expr.indices[0].end, IntLiteral)
        assert body_expr.indices[0].end.value == 4

    def test_full_axis_and_single(self):
        prog = _parse("fn f(x: f32[10, 20]) -> f32[10] { x[:, 0] }")
        body_expr = prog.functions[0].body.expr
        assert isinstance(body_expr, IndexExpr)
        assert len(body_expr.indices) == 2
        assert body_expr.indices[0].kind == "full"
        assert body_expr.indices[1].kind == "single"

    def test_chained_indexing(self):
        prog = _parse("fn f(x: f32[10, 20]) -> f32 { x[0][1] }")
        body_expr = prog.functions[0].body.expr
        # Outer IndexExpr wraps inner IndexExpr
        assert isinstance(body_expr, IndexExpr)
        assert isinstance(body_expr.base, IndexExpr)
        # Inner: x[0]
        assert body_expr.base.indices[0].kind == "single"
        # Outer: ...[1]
        assert body_expr.indices[0].kind == "single"

    def test_index_after_call(self):
        prog = _parse(
            "fn g(x: f32[10]) -> f32[10] { x }\n"
            "fn f(x: f32[10]) -> f32 { g(x)[0] }"
        )
        body_expr = prog.functions[1].body.expr
        assert isinstance(body_expr, IndexExpr)
        # base is a CallExpr
        from maomi.ast_nodes import CallExpr
        assert isinstance(body_expr.base, CallExpr)

    def test_index_in_expression(self):
        prog = _parse("fn f(x: f32[10], i: i32) -> f32 { x[i] * 2.0 }")
        from maomi.ast_nodes import BinOp
        body_expr = prog.functions[0].body.expr
        assert isinstance(body_expr, BinOp)
        assert isinstance(body_expr.left, IndexExpr)


# ---------- Type checker tests ----------


class TestTypeCheckerIndexing:
    def test_single_index_1d_to_scalar(self):
        _, tc, errors = _check("fn f(x: f32[10]) -> f32 { x[0] }")
        assert not errors

    def test_single_index_2d_removes_axis(self):
        _, tc, errors = _check("fn f(x: f32[10, 20]) -> f32[20] { x[0] }")
        assert not errors

    def test_dynamic_index(self):
        _, tc, errors = _check("fn f(x: f32[10], i: i32) -> f32 { x[i] }")
        assert not errors

    def test_full_axis_and_single(self):
        _, tc, errors = _check("fn f(x: f32[10, 20]) -> f32[10] { x[:, 0] }")
        assert not errors

    def test_slice_computes_size(self):
        _, tc, errors = _check("fn f(x: f32[10]) -> f32[3] { x[1:4] }")
        assert not errors

    def test_slice_2d(self):
        _, tc, errors = _check("fn f(x: f32[10, 20]) -> f32[2, 20] { x[1:3] }")
        assert not errors

    def test_multi_axis_mixed(self):
        # f32[10, 20, 30][:, 2, 1:5] → f32[10, 4]
        _, tc, errors = _check("fn f(x: f32[10, 20, 30]) -> f32[10, 4] { x[:, 2, 1:5] }")
        assert not errors

    def test_chained_indexing_types(self):
        _, tc, errors = _check("fn f(x: f32[10, 20]) -> f32 { x[0][1] }")
        assert not errors

    def test_error_indexing_scalar(self):
        _, tc, errors = _check("fn f(x: f32) -> f32 { x[0] }")
        assert len(errors) == 1
        assert "indexing requires an array" in errors[0].message

    def test_error_too_many_indices(self):
        _, tc, errors = _check("fn f(x: f32[10]) -> f32 { x[0, 1] }")
        assert len(errors) == 1
        assert "too many indices" in errors[0].message

    def test_error_non_i32_index(self):
        _, tc, errors = _check("fn f(x: f32[10], i: f32) -> f32 { x[i] }")
        assert len(errors) >= 1
        assert any("index must be i32" in e.message for e in errors)

    def test_error_empty_slice(self):
        _, tc, errors = _check("fn f(x: f32[10]) -> f32[0] { x[3:3] }")
        assert len(errors) >= 1
        assert any("empty or negative" in e.message for e in errors)


# ---------- Codegen tests ----------


class TestCodegenIndexing:
    def test_static_single_index(self):
        mlir = _compile("fn f(x: f32[10]) -> f32 { x[0] }")
        assert "stablehlo.slice" in mlir
        assert "stablehlo.reshape" in mlir

    def test_dynamic_single_index(self):
        mlir = _compile("fn f(x: f32[10], i: i32) -> f32 { x[i] }")
        assert "stablehlo.dynamic_slice" in mlir
        assert "stablehlo.reshape" in mlir

    def test_static_slice(self):
        mlir = _compile("fn f(x: f32[10]) -> f32[3] { x[1:4] }")
        assert "stablehlo.slice" in mlir

    def test_full_axis_single(self):
        mlir = _compile("fn f(x: f32[10, 20]) -> f32[10] { x[:, 0] }")
        # Should emit slice + reshape to remove axis 1
        assert "stablehlo.slice" in mlir or "stablehlo.dynamic_slice" in mlir
        assert "stablehlo.reshape" in mlir

    def test_2d_single_index(self):
        mlir = _compile("fn f(x: f32[10, 20]) -> f32[20] { x[0] }")
        assert "stablehlo.slice" in mlir
        assert "stablehlo.reshape" in mlir

    def test_chained_index(self):
        mlir = _compile("fn f(x: f32[10, 20]) -> f32 { x[0][1] }")
        # Two slice+reshape pairs
        assert mlir.count("stablehlo.reshape") == 2

    def test_dynamic_index_2d(self):
        mlir = _compile("fn f(x: f32[10, 20], i: i32) -> f32[20] { x[i] }")
        assert "stablehlo.dynamic_slice" in mlir
        assert "stablehlo.reshape" in mlir

    def test_mixed_multi_axis(self):
        mlir = _compile("fn f(x: f32[10, 20, 30]) -> f32[10, 4] { x[:, 2, 1:5] }")
        assert "stablehlo.slice" in mlir
        assert "stablehlo.reshape" in mlir


# ---------- AD tests ----------


class TestADIndexing:
    def test_grad_through_static_index(self):
        src = """
        fn f(x: f32[10]) -> f32[10] {
            grad(x[0] * 2.0, x)
        }
        """
        mlir = _compile(src)
        # Should emit dynamic_update_slice in the backward pass
        assert "stablehlo.dynamic_update_slice" in mlir

    def test_grad_through_dynamic_index(self):
        src = """
        fn f(x: f32[10], i: i32) -> f32[10] {
            let y: f32 = x[i];
            grad(y * 2.0, x)
        }
        fn main(x: f32[10], i: i32) -> f32[10] { f(x, i) }
        """
        mlir = _compile(src)
        assert "stablehlo.dynamic_update_slice" in mlir

    def test_grad_through_index_produces_correct_shape(self):
        # Gradient of x[0] * 2.0 w.r.t. x should be f32[10]
        src = """
        fn f(x: f32[10]) -> f32[10] {
            grad(x[0] * 2.0, x)
        }
        """
        prog, tc, errors = _check(src)
        assert not errors


# ---------- Open-ended slices and negative indices ----------


class TestParserOpenEndedSlices:
    def test_open_end_slice(self):
        prog = _parse("fn f(x: f32[10]) -> f32[9] { x[1:] }")
        body_expr = prog.functions[0].body.expr
        assert isinstance(body_expr, IndexExpr)
        ic = body_expr.indices[0]
        assert ic.kind == "slice"
        assert isinstance(ic.start, IntLiteral)
        assert ic.start.value == 1
        assert ic.end is None

    def test_open_start_slice(self):
        prog = _parse("fn f(x: f32[10]) -> f32[3] { x[:3] }")
        body_expr = prog.functions[0].body.expr
        ic = body_expr.indices[0]
        assert ic.kind == "slice"
        assert ic.start is None
        assert isinstance(ic.end, IntLiteral)
        assert ic.end.value == 3

    def test_open_start_negative_end(self):
        prog = _parse("fn f(x: f32[10]) -> f32[9] { x[:-1] }")
        body_expr = prog.functions[0].body.expr
        ic = body_expr.indices[0]
        assert ic.kind == "slice"
        assert ic.start is None
        assert isinstance(ic.end, UnaryOp)
        assert ic.end.op == "-"

    def test_negative_single_index(self):
        prog = _parse("fn f(x: f32[10]) -> f32 { x[-1] }")
        body_expr = prog.functions[0].body.expr
        ic = body_expr.indices[0]
        assert ic.kind == "single"
        assert isinstance(ic.value, UnaryOp)
        assert ic.value.op == "-"

    def test_negative_start_open_end(self):
        prog = _parse("fn f(x: f32[10]) -> f32[2] { x[-2:] }")
        body_expr = prog.functions[0].body.expr
        ic = body_expr.indices[0]
        assert ic.kind == "slice"
        assert isinstance(ic.start, UnaryOp)
        assert ic.end is None


class TestTypeCheckerOpenEndedSlices:
    def test_open_end_slice(self):
        _, tc, errors = _check("fn f(x: f32[10]) -> f32[9] { x[1:] }")
        assert not errors

    def test_open_start_slice(self):
        _, tc, errors = _check("fn f(x: f32[10]) -> f32[3] { x[:3] }")
        assert not errors

    def test_open_start_negative_end(self):
        _, tc, errors = _check("fn f(x: f32[10]) -> f32[9] { x[:-1] }")
        assert not errors

    def test_negative_single_index(self):
        _, tc, errors = _check("fn f(x: f32[10]) -> f32 { x[-1] }")
        assert not errors

    def test_negative_start_negative_end(self):
        # x[1:-1] on f32[10] → start=1, end=9, size=8
        _, tc, errors = _check("fn f(x: f32[10]) -> f32[8] { x[1:-1] }")
        assert not errors

    def test_negative_index_2d(self):
        _, tc, errors = _check("fn f(x: f32[10, 20]) -> f32[10] { x[:, -1] }")
        assert not errors

    def test_negative_start_open_end(self):
        # x[-2:] on f32[10] → start=8, end=10, size=2
        _, tc, errors = _check("fn f(x: f32[10]) -> f32[2] { x[-2:] }")
        assert not errors

    def test_error_negative_out_of_bounds(self):
        _, tc, errors = _check("fn f(x: f32[10]) -> f32 { x[-11] }")
        assert len(errors) >= 1
        assert any("out of bounds" in e.message for e in errors)

    def test_error_negative_slice_out_of_bounds(self):
        # x[:-11] on f32[10] → end = 10 + (-11) = -1 → normalized, then end < start
        _, tc, errors = _check("fn f(x: f32[10]) -> f32[0] { x[:-11] }")
        assert len(errors) >= 1


class TestCodegenOpenEndedSlices:
    def test_negative_static_index_uses_slice(self):
        mlir = _compile("fn f(x: f32[10]) -> f32 { x[-1] }")
        # After normalization, -1 → 9, so should use static stablehlo.slice
        assert "stablehlo.slice" in mlir
        assert "stablehlo.reshape" in mlir

    def test_open_end_slice(self):
        mlir = _compile("fn f(x: f32[10]) -> f32[9] { x[1:] }")
        assert "stablehlo.slice" in mlir

    def test_open_start_slice(self):
        mlir = _compile("fn f(x: f32[10]) -> f32[3] { x[:3] }")
        assert "stablehlo.slice" in mlir

    def test_open_start_negative_end(self):
        mlir = _compile("fn f(x: f32[10]) -> f32[9] { x[:-1] }")
        assert "stablehlo.slice" in mlir

    def test_dynamic_index_emits_normalization(self):
        mlir = _compile("fn f(x: f32[10], i: i32) -> f32 { x[i] }")
        # Should emit select-based normalization for dynamic indices
        assert "stablehlo.compare" in mlir
        assert "stablehlo.select" in mlir
        assert "stablehlo.dynamic_slice" in mlir


class TestADOpenEndedSlices:
    def test_grad_through_negative_index(self):
        src = """
        fn f(x: f32[10]) -> f32[10] {
            grad(x[-1] * 2.0, x)
        }
        """
        mlir = _compile(src)
        assert "stablehlo.dynamic_update_slice" in mlir

    def test_grad_through_open_end_slice(self):
        src = """
        fn f(x: f32[10]) -> f32[10] {
            let s: f32[9] = x[1:];
            grad(s[0] * 2.0, x)
        }
        """
        mlir = _compile(src)
        assert "stablehlo.dynamic_update_slice" in mlir
