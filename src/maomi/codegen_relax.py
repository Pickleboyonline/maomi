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
    FieldAccess,
    FloatLiteral,
    FnDef,
    Identifier,
    IfExpr,
    IntLiteral,
    LetStmt,
    StructLiteral,
    UnaryOp,
    WithExpr,
    _BroadcastExpr,
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
        return self._resolve_annotation_type(param.type_annotation)

    def _resolve_annotation_type(self, ta) -> MaomiType:
        if ta.base in ("f32", "f64", "i32", "i64", "bool"):
            if ta.dims is None:
                return ScalarType(ta.base)
            return ArrayType(ta.base, tuple(d.value for d in ta.dims))
        if ta.base == "Key":
            return ArrayType("i32", (4,))
        # Struct type — look up from program's struct_defs
        for sd in self.program.struct_defs:
            if sd.name == ta.base:
                field_types = []
                for field_name, field_ta in sd.fields:
                    field_types.append((field_name, self._resolve_annotation_type(field_ta)))
                return StructType(sd.name, tuple(field_types))
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

            case IfExpr(condition=cond, then_block=then_b, else_block=else_b):
                return self._gen_if(cond, then_b, else_b, env)

            case CallExpr(callee=callee, args=args):
                return self._gen_call(callee, args, expr, env)

            case StructLiteral(fields=fields):
                return self._gen_struct_literal(fields, env)

            case FieldAccess(object=obj, field=field_name):
                return self._gen_field_access(obj, field_name, expr, env)

            case WithExpr(base=base, updates=updates):
                return self._gen_with(base, updates, env)

            case _BroadcastExpr(expr=inner, target_dims=target_dims):
                return self._gen_broadcast(inner, target_dims, env)

        raise NotImplementedError(f"Relax codegen: unsupported expr {type(expr).__name__}")

    # ---- Binary operations ----

    _BINOP_MAP = {
        "+": rx.op.add,
        "-": rx.op.subtract,
        "*": rx.op.multiply,
        "/": rx.op.divide,
        "**": rx.op.power,
    }

    _COMPARISON_MAP = {
        "==": rx.op.equal,
        "!=": rx.op.not_equal,
        "<": rx.op.less,
        ">": rx.op.greater,
        "<=": rx.op.less_equal,
        ">=": rx.op.greater_equal,
    }

    def _gen_binop(self, op: str, left: Expr, right: Expr, env: dict[str, rx.Expr]) -> rx.Expr:
        lhs = self._gen_expr(left, env)
        rhs = self._gen_expr(right, env)

        if op == "@":
            return self.bb.emit(rx.op.matmul(lhs, rhs))

        cmp_op = self._COMPARISON_MAP.get(op)
        if cmp_op is not None:
            return self.bb.emit(cmp_op(lhs, rhs))

        relax_op = self._BINOP_MAP.get(op)
        if relax_op is not None:
            return self.bb.emit(relax_op(lhs, rhs))

        raise NotImplementedError(f"Relax codegen: unsupported binary op '{op}'")

    # ---- If/else ----

    def _gen_if(self, cond_expr: Expr, then_block: Block, else_block: Block,
                env: dict[str, rx.Expr]) -> rx.Expr:
        cond = self._gen_expr(cond_expr, env)
        then_val = self._gen_block(then_block, env)
        else_val = self._gen_block(else_block, env)
        return self.bb.emit(rx.op.where(cond, then_val, else_val))

    # ---- Struct operations ----

    def _gen_struct_literal(self, fields: list[tuple[str, Expr]],
                            env: dict[str, rx.Expr]) -> rx.Expr:
        field_vals = [self._gen_expr(fv, env) for _, fv in fields]
        return self.bb.emit(rx.Tuple(field_vals))

    def _gen_field_access(self, obj_expr: Expr, field_name: str, expr: Expr,
                          env: dict[str, rx.Expr]) -> rx.Expr:
        obj = self._gen_expr(obj_expr, env)
        obj_type = self._type_of(obj_expr)
        if not isinstance(obj_type, StructType):
            raise ValueError(f"Field access on non-struct type: {obj_type}")
        field_idx = next(i for i, (fn, _) in enumerate(obj_type.fields) if fn == field_name)
        return self.bb.emit(rx.TupleGetItem(obj, field_idx))

    def _gen_with(self, base_expr: Expr, updates: list[tuple[list[str], Expr]],
                  env: dict[str, rx.Expr]) -> rx.Expr:
        base = self._gen_expr(base_expr, env)
        base_type = self._type_of(base_expr)
        if not isinstance(base_type, StructType):
            raise ValueError("'with' on non-struct type")
        return self._gen_with_struct(base, base_type, updates, env)

    def _gen_with_struct(self, base: rx.Expr, stype: StructType,
                         updates: list[tuple[list[str], Expr]],
                         env: dict[str, rx.Expr]) -> rx.Expr:
        # Group updates by top-level field
        top_updates: dict[str, list[tuple[list[str], Expr]]] = {}
        for path, value_expr in updates:
            top = path[0]
            rest = path[1:]
            if top not in top_updates:
                top_updates[top] = []
            top_updates[top].append((rest, value_expr))

        field_vals = []
        for i, (field_name, field_type) in enumerate(stype.fields):
            if field_name in top_updates:
                field_updates = top_updates[field_name]
                if any(len(rest) == 0 for rest, _ in field_updates):
                    _, value_expr = next((r, ve) for r, ve in field_updates if len(r) == 0)
                    field_vals.append(self._gen_expr(value_expr, env))
                else:
                    if not isinstance(field_type, StructType):
                        raise ValueError(f"Nested 'with' on non-struct field '{field_name}'")
                    extracted = self.bb.emit(rx.TupleGetItem(base, i))
                    field_vals.append(self._gen_with_struct(extracted, field_type, field_updates, env))
            else:
                field_vals.append(self.bb.emit(rx.TupleGetItem(base, i)))

        return self.bb.emit(rx.Tuple(field_vals))

    # ---- Broadcast (AD internal node) ----

    def _gen_broadcast(self, inner_expr: Expr, target_dims: tuple[int, ...],
                       env: dict[str, rx.Expr]) -> rx.Expr:
        inner = self._gen_expr(inner_expr, env)
        inner_type = self._type_of(inner_expr)
        dtype = _TVM_DTYPE[inner_type.base]
        target_shape = list(target_dims)
        return self.bb.emit(rx.op.broadcast_to(inner, target_shape))

    # ---- Function calls (builtins + user) ----

    _ELEMENTWISE_BUILTINS = {
        "exp": rx.op.exp,
        "log": rx.op.log,
        "tanh": rx.op.tanh,
        "sqrt": rx.op.sqrt,
        "abs": rx.op.abs,
    }

    _CALLBACK_BUILTINS = {"callback"}

    def _gen_call(self, callee: str, args: list[Expr], call_expr: CallExpr,
                  env: dict[str, rx.Expr]) -> rx.Expr:
        # Callback: no-op
        if callee in self._CALLBACK_BUILTINS:
            return rx.const(0, "int32")  # placeholder

        # Elementwise builtins
        if callee in self._ELEMENTWISE_BUILTINS:
            arg = self._gen_expr(args[0], env)
            return self.bb.emit(self._ELEMENTWISE_BUILTINS[callee](arg))

        if callee == "sum":
            return self._gen_sum(args, env)
        if callee == "mean":
            return self._gen_mean(args, call_expr, env)
        if callee == "transpose":
            return self._gen_transpose(args, env)
        if callee == "reshape":
            return self._gen_reshape(args, call_expr, env)
        if callee == "concat":
            return self._gen_concat(args, call_expr, env)

        # User-defined function call
        arg_vals = [self._gen_expr(a, env) for a in args]
        gv = self.bb.get().get_global_var(callee)
        return self.bb.emit(rx.Call(gv, arg_vals))

    # ---- Builtin implementations ----

    def _gen_sum(self, args: list[Expr], env: dict[str, rx.Expr]) -> rx.Expr:
        arg = self._gen_expr(args[0], env)
        arg_type = self._type_of(args[0])
        if not isinstance(arg_type, ArrayType):
            return arg
        axes = list(range(len(arg_type.dims)))
        return self.bb.emit(rx.op.sum(arg, axis=axes))

    def _gen_mean(self, args: list[Expr], call_expr: CallExpr,
                  env: dict[str, rx.Expr]) -> rx.Expr:
        arg = self._gen_expr(args[0], env)
        arg_type = self._type_of(args[0])
        if not isinstance(arg_type, ArrayType):
            return arg
        axes = list(range(len(arg_type.dims)))
        return self.bb.emit(rx.op.mean(arg, axis=axes))

    def _gen_transpose(self, args: list[Expr], env: dict[str, rx.Expr]) -> rx.Expr:
        arg = self._gen_expr(args[0], env)
        return self.bb.emit(rx.op.permute_dims(arg, [1, 0]))

    def _gen_reshape(self, args: list[Expr], call_expr: CallExpr,
                     env: dict[str, rx.Expr]) -> rx.Expr:
        arg = self._gen_expr(args[0], env)
        result_type = self._type_of(call_expr)
        if isinstance(result_type, ArrayType):
            target_shape = list(result_type.dims)
        else:
            target_shape = []
        return self.bb.emit(rx.op.reshape(arg, target_shape))

    def _gen_concat(self, args: list[Expr], call_expr: CallExpr,
                    env: dict[str, rx.Expr]) -> rx.Expr:
        # Detect axis: if last arg is IntLiteral with scalar type, it's the axis
        if (isinstance(args[-1], IntLiteral)
                and isinstance(self._type_of(args[-1]), ScalarType)):
            axis = args[-1].value
            array_args = args[:-1]
        else:
            axis = 0
            array_args = args
        arg_vals = [self._gen_expr(a, env) for a in array_args]
        return self.bb.emit(rx.op.concat(arg_vals, axis=axis))
