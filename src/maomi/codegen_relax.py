"""TVM Relax code generation backend for Maomi.

Walks the typed AST and builds a TVM IRModule using the Relax BlockBuilder API.
This is a parallel backend to codegen_stablehlo.py — same input (Program + type_map),
different output (tvm.IRModule instead of MLIR text).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tvm
from tvm import relax as rx

from .ast_nodes import (
    BinOp,
    Block,
    BoolLiteral,
    CallExpr,
    ExprStmt,
    FloatLiteral,
    FnDef,
    Identifier,
    IntLiteral,
    LetStmt,
    UnaryOp,
)
from .types import ArrayType, MaomiType, ScalarType, StructType

if TYPE_CHECKING:
    from .ast_nodes import Expr, Program

# Maomi dtype → TVM dtype string
_TVM_DTYPE = {
    "f32": "float32",
    "f64": "float64",
    "i32": "int32",
    "i64": "int64",
    "bool": "bool",
}


class RelaxCodegen:
    """Generate a TVM IRModule from a typed Maomi AST."""

    def __init__(self, program: Program, type_map: dict[int, MaomiType]) -> None:
        self.program = program
        self.type_map = type_map
        self.bb = rx.BlockBuilder()

    def generate(self) -> tvm.IRModule:
        for fn in self.program.functions:
            self._gen_function(fn)
        return self.bb.get()

    # ---- Type conversion ----

    def _relax_type(self, t: MaomiType) -> rx.TensorStructInfo | rx.TupleStructInfo:
        match t:
            case ScalarType(base=base):
                return rx.TensorStructInfo((), _TVM_DTYPE[base])
            case ArrayType(base=base, dims=dims):
                return rx.TensorStructInfo(tuple(dims), _TVM_DTYPE[base])
            case StructType(fields=fields):
                return rx.TupleStructInfo(
                    [self._relax_type(ft) for _, ft in fields]
                )
        raise ValueError(f"Unknown type: {t}")

    def _type_of(self, expr: Expr) -> MaomiType:
        return self.type_map[id(expr)]

    def _resolve_param_type(self, param) -> MaomiType:
        ta = param.type_annotation
        if ta.dims is None:
            return ScalarType(ta.base)
        return ArrayType(ta.base, tuple(d.value for d in ta.dims))

    # ---- Function generation ----

    def _gen_function(self, fn: FnDef) -> None:
        params = []
        env: dict[str, rx.Expr] = {}
        for p in fn.params:
            pt = self._resolve_param_type(p)
            var = rx.Var(p.name, self._relax_type(pt))
            params.append(var)
            env[p.name] = var

        with self.bb.function(fn.name, params):
            with self.bb.dataflow():
                result = self._gen_block(fn.body, env)
                out = self.bb.emit_output(result)
            self.bb.emit_func_output(out)

    # ---- Block generation ----

    def _gen_block(self, block: Block, env: dict[str, rx.Expr]) -> rx.Expr:
        child_env = dict(env)
        for stmt in block.stmts:
            match stmt:
                case LetStmt(name=name, value=value):
                    val = self._gen_expr(value, child_env)
                    child_env[name] = val
                case ExprStmt(expr=expr):
                    self._gen_expr(expr, child_env)
        if block.expr is not None:
            return self._gen_expr(block.expr, child_env)
        raise ValueError("Block has no trailing expression")

    # ---- Expression generation ----

    def _gen_expr(self, expr: Expr, env: dict[str, rx.Expr]) -> rx.Expr:
        match expr:
            case IntLiteral(value=v):
                t = self._type_of(expr)
                dtype = _TVM_DTYPE[t.base]
                return rx.const(v, dtype)

            case FloatLiteral(value=v):
                t = self._type_of(expr)
                dtype = _TVM_DTYPE[t.base]
                return rx.const(v, dtype)

            case BoolLiteral(value=v):
                return rx.const(v, "bool")

            case Identifier(name=name):
                return env[name]

            case UnaryOp(op="-", operand=operand):
                val = self._gen_expr(operand, env)
                return self.bb.emit(rx.op.negative(val))

            case BinOp(op=op, left=left, right=right):
                return self._gen_binop(op, left, right, env)

            case CallExpr(callee=callee, args=args):
                return self._gen_call(callee, args, env)

        raise NotImplementedError(f"Relax codegen: unsupported expr {type(expr).__name__}")

    # ---- Binary operations ----

    _BINOP_MAP = {
        "+": rx.op.add,
        "-": rx.op.subtract,
        "*": rx.op.multiply,
        "/": rx.op.divide,
        "**": rx.op.power,
    }

    def _gen_binop(self, op: str, left: Expr, right: Expr, env: dict[str, rx.Expr]) -> rx.Expr:
        lhs = self._gen_expr(left, env)
        rhs = self._gen_expr(right, env)

        if op == "@":
            return self.bb.emit(rx.op.matmul(lhs, rhs))

        relax_op = self._BINOP_MAP.get(op)
        if relax_op is not None:
            return self.bb.emit(relax_op(lhs, rhs))

        raise NotImplementedError(f"Relax codegen: unsupported binary op '{op}'")

    # ---- Function calls (builtins) ----

    def _gen_call(self, callee: str, args: list[Expr], env: dict[str, rx.Expr]) -> rx.Expr:
        raise NotImplementedError(f"Relax codegen: unsupported call '{callee}'")
