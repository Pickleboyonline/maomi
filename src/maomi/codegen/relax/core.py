"""TVM Relax code generation backend for Maomi.

Walks the typed AST and builds a TVM IRModule using the Relax BlockBuilder API.
This is a parallel backend to codegen_stablehlo.py — same input (Program + type_map),
different output (tvm.IRModule instead of MLIR text).
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import numpy as np
import tvm
from tvm import relax as rx
from tvm.script import tir as T

from ...ast_nodes import (
    BinOp,
    Block,
    BoolLiteral,
    CallExpr,
    CastExpr,
    ExprStmt,
    FieldAccess,
    FloatLiteral,
    FnDef,
    FoldExpr,
    Identifier,
    IfExpr,
    IndexExpr,
    IntLiteral,
    LetStmt,
    MapExpr,
    ScanExpr,
    StructLiteral,
    UnaryOp,
    WhileExpr,
    WithExpr,
    _AvgPoolGrad,
    _BroadcastExpr,
    _Conv2dGrad,
    _GatherGrad,
    _IndexGrad,
    _MaxPoolGrad,
    _ReduceSum,
    _ScanGrad,
    _WhileGrad,
)
from ...types import ArrayType, MaomiType, ScalarType, StructType

if TYPE_CHECKING:
    from ...ast_nodes import Expr, Program


# ---- Standalone helpers (AST-level, backend-agnostic) ----

def _collect_body_refs(block: Block) -> set[str]:
    """Collect all Identifier names referenced in a block."""
    refs: set[str] = set()
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt):
            _collect_refs_expr(stmt.value, refs)
        elif isinstance(stmt, ExprStmt):
            _collect_refs_expr(stmt.expr, refs)
    if block.expr is not None:
        _collect_refs_expr(block.expr, refs)
    return refs


def _collect_refs_expr(expr, refs: set[str]):
    """Recursively collect Identifier names from an expression."""
    match expr:
        case Identifier(name=name):
            refs.add(name)
        case BinOp(left=left, right=right):
            _collect_refs_expr(left, refs)
            _collect_refs_expr(right, refs)
        case UnaryOp(operand=operand):
            _collect_refs_expr(operand, refs)
        case CallExpr(args=args):
            for a in args:
                _collect_refs_expr(a, refs)
        case IfExpr(condition=cond, then_block=tb, else_block=eb):
            _collect_refs_expr(cond, refs)
            refs.update(_collect_body_refs(tb))
            refs.update(_collect_body_refs(eb))
        case MapExpr(sequence=seq, body=body):
            _collect_refs_expr(seq, refs)
            inner = _collect_body_refs(body)
            inner.discard(expr.elem_var)
            refs.update(inner)
        case ScanExpr(init=init, sequences=seqs, body=body):
            _collect_refs_expr(init, refs)
            for s in seqs:
                _collect_refs_expr(s, refs)
            inner = _collect_body_refs(body)
            inner.discard(expr.carry_var)
            for ev in expr.elem_vars:
                inner.discard(ev)
            refs.update(inner)
        case StructLiteral(fields=fields):
            for _, fv in fields:
                _collect_refs_expr(fv, refs)
        case FieldAccess(object=obj):
            _collect_refs_expr(obj, refs)
        case WithExpr(base=base, updates=updates):
            _collect_refs_expr(base, refs)
            for _, ve in updates:
                _collect_refs_expr(ve, refs)
        case _BroadcastExpr(expr=e):
            _collect_refs_expr(e, refs)
        case CastExpr(expr=e):
            _collect_refs_expr(e, refs)
        case IndexExpr(base=base, indices=indices):
            _collect_refs_expr(base, refs)
            for ic in indices:
                if ic.kind == "single" and ic.value is not None:
                    _collect_refs_expr(ic.value, refs)


# ---- TensorIR PrimFuncs for RNG ----

def _make_xorshift_prim(n: int):
    """Create a TensorIR PrimFunc implementing xorshift128 PRNG.

    Takes i32[4] key, produces uint32[n] random bits.
    """

    @T.prim_func
    def xorshift(key: T.Buffer((4,), "int32"), out: T.Buffer((n,), "uint32")):
        s = T.alloc_buffer((4,), "uint32")
        for i in range(4):
            s[i] = T.reinterpret("uint32", key[i])
        for j in range(n):
            t: T.uint32 = s[0] ^ (s[0] << T.uint32(11))
            s[0] = s[1]
            s[1] = s[2]
            s[2] = s[3]
            s[3] = (s[3] ^ (s[3] >> T.uint32(19))) ^ (t ^ (t >> T.uint32(8)))
            out[j] = s[3]

    return xorshift


def _make_bits_to_uniform_prim(n: int):
    """Create a TensorIR PrimFunc converting uint32 bits to uniform f32 in [0, 1).

    Uses bits >> 9 | 0x3F800000 -> reinterpret as f32 -> subtract 1.0.
    """

    @T.prim_func
    def bits_to_uniform(
        bits: T.Buffer((n,), "uint32"), out: T.Buffer((n,), "float32")
    ):
        for i in range(n):
            mantissa: T.uint32 = (bits[i] >> T.uint32(9)) | T.uint32(0x3F800000)
            out[i] = T.reinterpret("float32", mantissa) - T.float32(1.0)

    return bits_to_uniform


# Maomi dtype -> TVM dtype string
_TVM_DTYPE = {
    "f32": "float32",
    "f64": "float64",
    "bf16": "bfloat16",
    "i32": "int32",
    "i64": "int64",
    "bool": "bool",
}


class RelaxCodegen:
    """Generate a TVM IRModule from a typed Maomi AST."""

    _DEFAULT_WHILE_UNROLL = 1000

    def __init__(self, program: Program, type_map: dict[int, MaomiType]) -> None:
        self.program = program
        self.type_map = type_map
        self.bb = rx.BlockBuilder()
        # Map (vmap) state
        self._batch_depth = 0
        self._batch_dims: list[int] = []
        self._batched_fns: dict[tuple[str, tuple[int, ...]], str] = {}
        # RNG TIR function cache
        self._tir_cache: set[str] = set()

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
        if ta.base in ("f32", "f64", "bf16", "i32", "i64", "bool"):
            if ta.dims is None:
                return ScalarType(ta.base)
            return ArrayType(ta.base, tuple(d.value for d in ta.dims))
        if ta.base == "Key":
            return ArrayType("i32", (4,))
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

            case CastExpr():
                return self._gen_cast(expr, env)

            case IndexExpr():
                return self._gen_index(expr, env)

            case _BroadcastExpr():
                return self._gen_broadcast(expr, env)

            case _ReduceSum():
                return self._gen_reduce_sum_axes(expr, env)

            case _IndexGrad():
                return self._gen_index_grad(expr, env)

            case _GatherGrad():
                return self._gen_gather_grad(expr, env)

            case ScanExpr():
                return self._gen_scan(expr, env)

            case FoldExpr():
                return self._gen_fold(expr, env)

            case WhileExpr():
                return self._gen_while(expr, env)

            case _ScanGrad():
                return self._gen_scan_grad(expr, env)

            case _WhileGrad():
                return self._gen_while_grad(expr, env)

            case MapExpr():
                return self._gen_map(expr, env)

            case _Conv2dGrad():
                return self._gen_conv2d_grad(expr, env)

            case _MaxPoolGrad():
                return self._gen_max_pool_grad(expr, env)

            case _AvgPoolGrad():
                return self._gen_avg_pool_grad(expr, env)

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

    # ---- If/else + where helper ----

    def _where(self, cond: rx.Expr, true_val: rx.Expr, false_val: rx.Expr,
               val_type: MaomiType) -> rx.Expr:
        """Element-wise where that handles scalars, arrays, and structs (tuples)."""
        if isinstance(val_type, StructType):
            field_vals = []
            for i, (_, ft) in enumerate(val_type.fields):
                t_field = self.bb.emit(rx.TupleGetItem(true_val, i))
                f_field = self.bb.emit(rx.TupleGetItem(false_val, i))
                field_vals.append(self._where(cond, t_field, f_field, ft))
            return self.bb.emit(rx.Tuple(field_vals))
        return self.bb.emit(rx.op.where(cond, true_val, false_val))

    def _gen_if(self, cond_expr: Expr, then_block: Block, else_block: Block,
                env: dict[str, rx.Expr]) -> rx.Expr:
        cond = self._gen_expr(cond_expr, env)
        then_val = self._gen_block(then_block, env)
        else_val = self._gen_block(else_block, env)
        return self._where(cond, then_val, else_val, self._type_of(then_block.expr))

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

    # ---- Cast ----

    def _gen_cast(self, expr: CastExpr, env: dict[str, rx.Expr]) -> rx.Expr:
        val = self._gen_expr(expr.expr, env)
        return self.bb.emit(rx.op.astype(val, _TVM_DTYPE[expr.target_type]))

    # ---- Index normalization helper ----

    def _normalize_index(self, idx_val: rx.Expr, dim: int) -> rx.Expr:
        """Emit runtime normalization for potentially negative index: where(i < 0, i + dim, i)."""
        dim_const = rx.const(dim, "int32")
        zero_i = rx.const(0, "int32")
        is_neg = self.bb.emit(rx.op.less(idx_val, zero_i))
        added = self.bb.emit(rx.op.add(idx_val, dim_const))
        return self.bb.emit(rx.op.where(is_neg, added, idx_val))

    # ---- Forward indexing ----

    def _gen_index(self, expr: IndexExpr, env: dict[str, rx.Expr]) -> rx.Expr:
        """Emit forward array indexing (slicing, single index, gather)."""
        base_type = self._type_of(expr.base)
        if not isinstance(base_type, ArrayType):
            raise ValueError("codegen: indexing non-array")

        # Check for array-based indexing (gather)
        for ic in expr.indices:
            if ic.kind == "single":
                idx_type = self._type_of(ic.value)
                if isinstance(idx_type, ArrayType):
                    return self._gen_gather(expr, env)

        base = self._gen_expr(expr.base, env)

        # Build strided_slice parameters
        begin: list[int] = []
        end: list[int] = []
        strides: list[int] = []
        squeezed_axes: list[int] = []

        for i, ic in enumerate(expr.indices):
            dim = base_type.dims[i]
            if ic.kind == "single":
                if isinstance(ic.value, IntLiteral):
                    v = ic.value.value
                    idx = v if v >= 0 else v + dim
                else:
                    # Dynamic single index: use dynamic_strided_slice
                    idx_val = self._gen_expr(ic.value, env)
                    idx_val = self._normalize_index(idx_val, dim)
                    idx_val = self.bb.emit(rx.op.astype(idx_val, "int64"))
                    rank = len(base_type.dims)
                    dyn_begins = []
                    dyn_ends = []
                    for j in range(rank):
                        if j == i:
                            dyn_begins.append(idx_val)
                            dyn_ends.append(self.bb.emit(
                                rx.op.add(idx_val, rx.const(1, "int64"))
                            ))
                        elif j < len(expr.indices):
                            ic_j = expr.indices[j]
                            if ic_j.kind == "slice":
                                dyn_begins.append(rx.const(ic_j.start.value, "int64"))
                                dyn_ends.append(rx.const(ic_j.end.value, "int64"))
                            else:
                                dyn_begins.append(rx.const(0, "int64"))
                                dyn_ends.append(rx.const(base_type.dims[j], "int64"))
                        else:
                            dyn_begins.append(rx.const(0, "int64"))
                            dyn_ends.append(rx.const(base_type.dims[j], "int64"))

                    def _stack_scalars(vals: list[rx.Expr]) -> rx.Expr:
                        return self.bb.emit(rx.op.concat(
                            [self.bb.emit(rx.op.reshape(v, [1])) for v in vals], axis=0
                        ))

                    begins_t = _stack_scalars(dyn_begins)
                    ends_t = _stack_scalars(dyn_ends)
                    strides_t = _stack_scalars([rx.const(1, "int64")] * rank)

                    sliced = self.bb.emit(
                        rx.op.dynamic_strided_slice(base, begins_t, ends_t, strides_t)
                    )

                    sq = {j for j, ic_j in enumerate(expr.indices) if ic_j.kind == "single"}
                    slice_sizes = []
                    for j in range(rank):
                        if j == i:
                            slice_sizes.append(1)
                        elif j < len(expr.indices):
                            ic_j = expr.indices[j]
                            if ic_j.kind == "full":
                                slice_sizes.append(base_type.dims[j])
                            elif ic_j.kind == "slice":
                                slice_sizes.append(ic_j.end.value - ic_j.start.value)
                            else:
                                slice_sizes.append(1)
                        else:
                            slice_sizes.append(base_type.dims[j])
                    result_shape = [slice_sizes[j] for j in range(rank) if j not in sq]
                    if result_shape != slice_sizes:
                        return self.bb.emit(rx.op.reshape(sliced, result_shape or []))
                    return sliced

                begin.append(idx)
                end.append(idx + 1)
                strides.append(1)
                squeezed_axes.append(i)
            elif ic.kind == "slice":
                begin.append(ic.start.value)
                end.append(ic.end.value)
                strides.append(1)
            elif ic.kind == "full":
                begin.append(0)
                end.append(dim)
                strides.append(1)

        # Trailing unindexed axes
        for i in range(len(expr.indices), len(base_type.dims)):
            begin.append(0)
            end.append(base_type.dims[i])
            strides.append(1)

        sliced = self.bb.emit(rx.op.strided_slice(base, begin, end, strides))

        if squeezed_axes:
            squeezed_set = set(squeezed_axes)
            result_dims = [
                e - b for i, (b, e) in enumerate(zip(begin, end))
                if i not in squeezed_set
            ]
            return self.bb.emit(rx.op.reshape(sliced, result_dims or []))
        return sliced

    def _gen_gather(self, expr: IndexExpr, env: dict[str, rx.Expr]) -> rx.Expr:
        """Emit gather for array-based indexing (e.g., table[ids])."""
        base = self._gen_expr(expr.base, env)

        gather_axis = 0
        indices_expr = None
        for i, ic in enumerate(expr.indices):
            if ic.kind == "single":
                idx_type = self._type_of(ic.value)
                if isinstance(idx_type, ArrayType):
                    gather_axis = i
                    indices_expr = ic.value
                    break

        indices = self._gen_expr(indices_expr, env)
        idx_i64 = self.bb.emit(rx.op.astype(indices, "int64"))
        return self.bb.emit(rx.op.take(base, idx_i64, axis=gather_axis))

    # ---- _IndexGrad (backward pass for array indexing) ----

    def _gen_index_grad(self, expr: _IndexGrad, env: dict[str, rx.Expr]) -> rx.Expr:
        """Emit backward pass for indexing: zeros + scatter update at indexed position."""
        base_type = self._type_of(expr.base_expr)
        adj = self._gen_expr(expr.adj, env)

        if not isinstance(base_type, ArrayType):
            raise ValueError("codegen: _IndexGrad base must be array")

        dtype = _TVM_DTYPE[base_type.base]
        zeros = self.bb.emit(rx.op.zeros(tuple(base_type.dims), dtype))

        starts: list[int | None] = []
        slice_sizes: list[int] = []
        squeezed_axes: list[int] = []
        dynamic_idx: dict[int, object] = {}

        for i, ic in enumerate(expr.indices):
            dim = base_type.dims[i]
            if ic.kind == "single":
                if isinstance(ic.value, IntLiteral):
                    v = ic.value.value
                    starts.append(v if v >= 0 else v + dim)
                else:
                    starts.append(None)
                    dynamic_idx[i] = ic.value
                slice_sizes.append(1)
                squeezed_axes.append(i)
            elif ic.kind == "slice":
                starts.append(ic.start.value)
                slice_sizes.append(ic.end.value - ic.start.value)
            elif ic.kind == "full":
                starts.append(0)
                slice_sizes.append(dim)

        for i in range(len(expr.indices), len(base_type.dims)):
            starts.append(0)
            slice_sizes.append(base_type.dims[i])

        update_shape = tuple(slice_sizes)
        if squeezed_axes:
            adj = self.bb.emit(rx.op.reshape(adj, list(update_shape)))

        for axis_i in range(len(starts)):
            start_val = starts[axis_i]
            size_val = slice_sizes[axis_i]
            full_dim = base_type.dims[axis_i]

            if start_val == 0 and size_val == full_dim:
                continue

            if start_val is not None:
                return self.bb.emit(
                    rx.op.slice_scatter(zeros, adj, start_val, start_val + size_val, 1, axis_i)
                )
            else:
                idx_val = self._gen_expr(dynamic_idx[axis_i], env)
                idx_val = self._normalize_index(idx_val, full_dim)
                idx_expanded = self.bb.emit(rx.op.reshape(idx_val, [1] * len(update_shape)))
                idx_broadcasted = self.bb.emit(
                    rx.op.broadcast_to(idx_expanded, list(update_shape))
                )
                idx_i64 = self.bb.emit(rx.op.astype(idx_broadcasted, "int64"))
                return self.bb.emit(
                    rx.op.scatter_elements(zeros, idx_i64, adj, axis=axis_i, reduction="update")
                )

        return self.bb.emit(rx.op.add(zeros, adj))

    # ---- _GatherGrad (backward pass for gather) ----

    def _gen_gather_grad(self, expr: _GatherGrad, env: dict[str, rx.Expr]) -> rx.Expr:
        """Emit scatter with add-accumulation for gather gradient."""
        base_type = self._type_of(expr.base_expr)
        adj = self._gen_expr(expr.adj, env)
        adj_type = self._type_of(expr.adj)
        indices = self._gen_expr(expr.indices, env)
        indices_type = self._type_of(expr.indices)
        k = expr.gather_axis
        rank = len(base_type.dims)
        B = indices_type.dims[0]

        dtype = _TVM_DTYPE[base_type.base]

        updates_dims = list(base_type.dims)
        updates_dims[k] = B

        if isinstance(adj_type, ScalarType):
            adj = self.bb.emit(rx.op.broadcast_to(adj, updates_dims))
        elif isinstance(adj_type, ArrayType) and adj_type.dims != tuple(updates_dims):
            adj = self.bb.emit(rx.op.broadcast_to(adj, updates_dims))

        zeros = self.bb.emit(rx.op.zeros(tuple(base_type.dims), dtype))

        if rank == 1:
            idx_for_scatter = indices
        else:
            reshape_shape = [1] * rank
            reshape_shape[k] = B
            idx_reshaped = self.bb.emit(rx.op.reshape(indices, reshape_shape))
            idx_for_scatter = self.bb.emit(rx.op.broadcast_to(idx_reshaped, updates_dims))

        idx_i64 = self.bb.emit(rx.op.astype(idx_for_scatter, "int64"))
        return self.bb.emit(
            rx.op.scatter_elements(zeros, idx_i64, adj, axis=k, reduction="add")
        )

    # ---- Broadcast (AD internal node) ----

    def _gen_broadcast(self, expr: _BroadcastExpr, env: dict[str, rx.Expr]) -> rx.Expr:
        inner = self._gen_expr(expr.expr, env)
        inner_type = self._type_of(expr.expr)
        target_shape = list(expr.target_dims)

        if isinstance(inner_type, ScalarType):
            return self.bb.emit(rx.op.broadcast_to(inner, target_shape))

        assert isinstance(inner_type, ArrayType)
        if expr.broadcast_dims is not None:
            ndim_target = len(expr.target_dims)
            intermediate_shape = [1] * ndim_target
            for src_dim, tgt_dim in enumerate(expr.broadcast_dims):
                intermediate_shape[tgt_dim] = inner_type.dims[src_dim]
            reshaped = self.bb.emit(rx.op.reshape(inner, intermediate_shape))
            return self.bb.emit(rx.op.broadcast_to(reshaped, target_shape))
        else:
            ndim_target = len(expr.target_dims)
            ndim_inner = len(inner_type.dims)
            if ndim_inner < ndim_target:
                intermediate_shape = [1] * (ndim_target - ndim_inner) + list(inner_type.dims)
                reshaped = self.bb.emit(rx.op.reshape(inner, intermediate_shape))
                return self.bb.emit(rx.op.broadcast_to(reshaped, target_shape))
            return self.bb.emit(rx.op.broadcast_to(inner, target_shape))

    # ---- _ReduceSum (AD internal node for map free var gradients) ----

    def _gen_reduce_sum_axes(self, expr: _ReduceSum,
                             env: dict[str, rx.Expr]) -> rx.Expr:
        inner = self._gen_expr(expr.expr, env)
        return self.bb.emit(rx.op.sum(inner, axis=list(expr.axes)))

    # ---- Scan (unrolled forward) ----

    @staticmethod
    def _elem_types_from_seq_types(seq_types: list[MaomiType], context: str) -> list[MaomiType]:
        """Strip the first (iteration) dim from each sequence type to get element types."""
        elem_types: list[MaomiType] = []
        for st in seq_types:
            if not isinstance(st, ArrayType):
                raise ValueError(f"codegen: {context} sequence must be array")
            if len(st.dims) == 1:
                elem_types.append(ScalarType(st.base))
            else:
                elem_types.append(ArrayType(st.base, st.dims[1:]))
        return elem_types

    def _gen_scan(self, expr: ScanExpr, env: dict[str, rx.Expr]) -> rx.Expr:
        """Unrolled forward scan: iterate over sequence, collect carries."""
        init_val = self._gen_expr(expr.init, env)
        seq_vals = [self._gen_expr(s, env) for s in expr.sequences]

        init_type = self._type_of(expr.init)
        seq_types = [self._type_of(s) for s in expr.sequences]

        seq_type0 = seq_types[0]
        if not isinstance(seq_type0, ArrayType):
            raise ValueError("codegen: scan sequence must be array")
        seq_len = seq_type0.dims[0]

        self._elem_types_from_seq_types(seq_types, "scan")

        if expr.reverse:
            indices = range(seq_len - 1, -1, -1)
        else:
            indices = range(seq_len)

        carry = init_val
        carries = []

        for t in indices:
            body_env = dict(env)
            body_env[expr.carry_var] = carry

            for i, (ev, sv) in enumerate(zip(expr.elem_vars, seq_vals)):
                elem_t = self._relax_strided_slice(sv, t, seq_types[i])
                body_env[ev] = elem_t

            carry = self._gen_block(expr.body, body_env)
            carries.append(carry)

        if expr.reverse:
            carries = list(reversed(carries))

        reshaped = []
        for c in carries:
            r = self.bb.emit(rx.op.reshape(c, [1] + list(self._shape_of_type(init_type))))
            reshaped.append(r)

        result = reshaped[0]
        for r in reshaped[1:]:
            result = self.bb.emit(rx.op.concat([result, r], axis=0))

        return result

    # ---- Fold (unrolled, returns final carry only) ----

    def _gen_fold(self, expr: FoldExpr, env: dict[str, rx.Expr]) -> rx.Expr:
        """Unrolled fold: same as scan but returns only the final carry."""
        carry = self._gen_expr(expr.init, env)
        seq_vals = [self._gen_expr(s, env) for s in expr.sequences]
        seq_types = [self._type_of(s) for s in expr.sequences]

        seq_type0 = seq_types[0]
        if not isinstance(seq_type0, ArrayType):
            raise ValueError("codegen: fold sequence must be array")
        seq_len = seq_type0.dims[0]

        for t in range(seq_len):
            body_env = dict(env)
            body_env[expr.carry_var] = carry
            for i, ev in enumerate(expr.elem_vars):
                body_env[ev] = self._relax_strided_slice(seq_vals[i], t, seq_types[i])
            carry = self._gen_block(expr.body, body_env)

        return carry

    # ---- While loop (bounded unrolling) ----

    def _gen_while(self, expr: WhileExpr, env: dict[str, rx.Expr]) -> rx.Expr:
        """Emit while loop via bounded unrolling with conditional state update."""
        init_val = self._gen_expr(expr.init, env)
        state_type = self._type_of(expr.init)
        max_iters = expr.max_iters if expr.max_iters is not None else self._DEFAULT_WHILE_UNROLL

        state = init_val
        for _ in range(max_iters):
            loop_env = dict(env)
            loop_env[expr.state_var] = state
            cond_val = self._gen_block(expr.cond, loop_env)
            new_state = self._gen_block(expr.body, loop_env)
            state = self._where(cond_val, new_state, state, state_type)

        return state

    # ---- Scan backward (_ScanGrad) ----

    def _gen_scan_grad(self, expr: _ScanGrad, env: dict[str, rx.Expr]) -> rx.Expr:
        """Reverse unrolled loop for scan backward pass."""
        fwd_val = self._gen_expr(expr.forward_result, env)
        init_val = self._gen_expr(expr.init, env)
        seq_vals = [self._gen_expr(s, env) for s in expr.sequences]
        adj_val = self._gen_expr(expr.adj, env)

        init_type = self._type_of(expr.init)
        fwd_type = self._type_of(expr.forward_result)
        adj_type = self._type_of(expr.adj)
        seq_types = [self._type_of(s) for s in expr.sequences]

        if not isinstance(fwd_type, ArrayType):
            raise ValueError("codegen: _ScanGrad forward_result must be array")

        seq_len = fwd_type.dims[0]

        self._elem_types_from_seq_types(seq_types, "_ScanGrad")

        if isinstance(adj_type, ScalarType) and isinstance(fwd_type, ArrayType):
            adj_val = self.bb.emit(rx.op.broadcast_to(adj_val, list(fwd_type.dims)))
            adj_type = fwd_type

        n_seqs = len(expr.sequences)
        adj_carry = self._relax_zeros_like(init_type)
        adj_seqs = [self._relax_zeros_like(st) for st in seq_types]

        for t in range(seq_len - 1, -1, -1):
            adj_t = self._relax_strided_slice(adj_val, t, adj_type)
            adj_total = self.bb.emit(rx.op.add(adj_carry, adj_t))

            if t > 0:
                carry_t = self._relax_strided_slice(fwd_val, t - 1, fwd_type)
            else:
                carry_t = init_val

            b_elems = []
            for i in range(n_seqs):
                elem_t = self._relax_strided_slice(seq_vals[i], t, seq_types[i])
                b_elems.append(elem_t)

            deriv_env = dict(env)
            deriv_env[expr.carry_var] = carry_t
            for ev, elem_val in zip(expr.elem_vars, b_elems):
                deriv_env[ev] = elem_val

            d_carry_val = self._gen_expr(expr.d_body_d_carry, deriv_env)
            adj_carry = self.bb.emit(rx.op.multiply(adj_total, d_carry_val))

            for i in range(n_seqs):
                d_elem_val = self._gen_expr(expr.d_body_d_elems[i], deriv_env)
                adj_elem_t = self.bb.emit(rx.op.multiply(adj_total, d_elem_val))
                adj_seqs[i] = self._relax_update_element(adj_seqs[i], adj_elem_t, t, seq_types[i])

        if expr.wrt == "__init__":
            return adj_carry
        else:
            for i, seq_expr in enumerate(expr.sequences):
                if isinstance(seq_expr, Identifier) and seq_expr.name == expr.wrt:
                    return adj_seqs[i]
            return adj_seqs[0]

    # ---- While backward (_WhileGrad) ----

    def _gen_while_grad(self, expr: _WhileGrad, env: dict[str, rx.Expr]) -> rx.Expr:
        """Augmented forward + reverse unrolled loop for while backward pass."""
        self._gen_expr(expr.forward_result, env)

        max_iters = expr.max_iters

        # Augmented forward: save trajectory
        init_val = self._gen_expr(expr.init, env)
        trajectory = [init_val]
        state = init_val
        active = []

        for _ in range(max_iters):
            loop_env = dict(env)
            loop_env[expr.state_var] = state
            cond_val = self._gen_block(expr.cond, loop_env)
            active.append(cond_val)
            new_state = self._gen_block(expr.body, loop_env)
            state = self.bb.emit(rx.op.where(cond_val, new_state, state))
            trajectory.append(state)

        # Backward pass
        adj = self._gen_expr(expr.adj, env)

        for t in range(max_iters - 1, -1, -1):
            deriv_env = dict(env)
            deriv_env[expr.state_var] = trajectory[t]
            d_val = self._gen_expr(expr.d_body_d_state, deriv_env)
            new_adj = self.bb.emit(rx.op.multiply(adj, d_val))
            adj = self.bb.emit(rx.op.where(active[t], new_adj, adj))

        return adj

    # ---- Scan/While utility helpers ----

    def _relax_strided_slice(self, seq: rx.Expr, idx: int,
                              seq_type: ArrayType) -> rx.Expr:
        """Slice a single element at static index from dimension 0."""
        ndim = len(seq_type.dims)
        begin = [idx] + [0] * (ndim - 1)
        end = [idx + 1] + [int(d) for d in seq_type.dims[1:]]
        sliced = self.bb.emit(rx.op.strided_slice(seq, axes=list(range(ndim)),
                                                   begin=begin, end=end))
        if len(seq_type.dims) == 1:
            target_shape: list[int] = []
        else:
            target_shape = [int(d) for d in seq_type.dims[1:]]
        return self.bb.emit(rx.op.reshape(sliced, target_shape))

    def _shape_of_type(self, t: MaomiType) -> list[int]:
        """Return the shape list of a Maomi type."""
        if isinstance(t, ScalarType):
            return []
        if isinstance(t, ArrayType):
            return [int(d) for d in t.dims]
        return []

    def _relax_zeros_like(self, t: MaomiType) -> rx.Expr:
        """Create a zero tensor with the same shape and dtype as the given type."""
        dtype = _TVM_DTYPE[t.base]
        if isinstance(t, ScalarType):
            return rx.const(0.0, dtype)
        if isinstance(t, ArrayType):
            shape = [int(d) for d in t.dims]
            return rx.const(np.zeros(shape, dtype=dtype), dtype)
        raise ValueError(f"Cannot create zeros for type: {t}")

    def _relax_update_element(self, arr: rx.Expr, elem: rx.Expr, idx: int,
                               arr_type: ArrayType) -> rx.Expr:
        """Update a single element at static index along dimension 0."""
        ndim = len(arr_type.dims)
        if len(arr_type.dims) == 1:
            elem_reshaped = self.bb.emit(rx.op.reshape(elem, [1]))
        else:
            inner_shape = [int(d) for d in arr_type.dims[1:]]
            elem_reshaped = self.bb.emit(rx.op.reshape(elem, [1] + inner_shape))

        parts = []
        if idx > 0:
            begin = [0] + [0] * (ndim - 1)
            end = [idx] + [int(d) for d in arr_type.dims[1:]]
            before = self.bb.emit(rx.op.strided_slice(arr, axes=list(range(ndim)),
                                                       begin=begin, end=end))
            parts.append(before)

        parts.append(elem_reshaped)

        if idx < arr_type.dims[0] - 1:
            begin = [idx + 1] + [0] * (ndim - 1)
            end = [int(arr_type.dims[0])] + [int(d) for d in arr_type.dims[1:]]
            after = self.bb.emit(rx.op.strided_slice(arr, axes=list(range(ndim)),
                                                      begin=begin, end=end))
            parts.append(after)

        if len(parts) == 1:
            return parts[0]
        result = parts[0]
        for p in parts[1:]:
            result = self.bb.emit(rx.op.concat([result, p], axis=0))
        return result

    # ---- Map (vmap) codegen ----

    def _gen_map(self, expr: MapExpr, env: dict[str, rx.Expr]) -> rx.Expr:
        """Generate map using type-lifting approach.

        Instead of looping, we "lift" all types in the body to include the
        batch dimension, then run the body once. Relax ops naturally broadcast
        over leading dimensions.
        """
        seq_val = self._gen_expr(expr.sequence, env)
        seq_type = self._type_of(expr.sequence)

        if not isinstance(seq_type, ArrayType):
            raise ValueError("codegen: map sequence must be array")

        batch_dim = seq_type.dims[0]

        body_refs = _collect_body_refs(expr.body)
        free_vars = (body_refs - {expr.elem_var}) & set(env.keys())

        self._batch_depth += 1
        self._batch_dims.append(batch_dim)

        self._lift_body_types(expr.body, batch_dim, free_vars)

        body_env = dict(env)
        body_env[expr.elem_var] = seq_val
        result = self._gen_block(expr.body, body_env)

        self._batch_depth -= 1
        self._batch_dims.pop()
        return result

    def _lift_body_types(self, block: Block, batch_dim: int, free_vars: set[str]):
        """Prepend batch_dim to all types in a block for map codegen."""
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                self._lift_expr_type(stmt.value, batch_dim, free_vars)
            elif isinstance(stmt, ExprStmt):
                self._lift_expr_type(stmt.expr, batch_dim, free_vars)
        if block.expr is not None:
            self._lift_expr_type(block.expr, batch_dim, free_vars)

    def _lift_expr_type(self, expr, batch_dim: int, free_vars: set[str]):
        """Recursively lift an expression's type to include batch_dim."""
        if isinstance(expr, (IntLiteral, FloatLiteral, BoolLiteral)):
            return
        if isinstance(expr, Identifier) and expr.name in free_vars:
            return
        is_inner_compound = isinstance(expr, MapExpr)

        if not is_inner_compound:
            t = self.type_map.get(id(expr))
            if t is not None:
                if isinstance(t, ScalarType):
                    self.type_map[id(expr)] = ArrayType(t.base, (batch_dim,))
                elif isinstance(t, ArrayType):
                    self.type_map[id(expr)] = ArrayType(t.base, (batch_dim,) + t.dims)

        match expr:
            case BinOp(left=left, right=right):
                self._lift_expr_type(left, batch_dim, free_vars)
                self._lift_expr_type(right, batch_dim, free_vars)
            case UnaryOp(operand=operand):
                self._lift_expr_type(operand, batch_dim, free_vars)
            case CallExpr(args=args):
                for a in args:
                    self._lift_expr_type(a, batch_dim, free_vars)
            case IfExpr():
                self._lift_expr_type(expr.condition, batch_dim, free_vars)
                self._lift_body_types(expr.then_block, batch_dim, free_vars)
                self._lift_body_types(expr.else_block, batch_dim, free_vars)
            case StructLiteral(fields=fields):
                for _, fv in fields:
                    self._lift_expr_type(fv, batch_dim, free_vars)
            case FieldAccess(object=obj):
                self._lift_expr_type(obj, batch_dim, free_vars)
            case WithExpr(base=base, updates=updates):
                self._lift_expr_type(base, batch_dim, free_vars)
                for _, ve in updates:
                    self._lift_expr_type(ve, batch_dim, free_vars)
            case MapExpr(sequence=seq):
                self._lift_expr_type(seq, batch_dim, free_vars)
            case _BroadcastExpr(expr=e):
                self._lift_expr_type(e, batch_dim, free_vars)
            case CastExpr(expr=e):
                self._lift_expr_type(e, batch_dim, free_vars)
            case _:
                pass

    def _broadcast_to_batched(self, val: rx.Expr, from_type: MaomiType) -> rx.Expr:
        """Broadcast an unbatched value to include batch dims at front."""
        batch_dims_tuple = tuple(self._batch_dims)
        if isinstance(from_type, ScalarType):
            target_shape = list(batch_dims_tuple)
            return self.bb.emit(rx.op.broadcast_to(val, target_shape))
        elif isinstance(from_type, ArrayType):
            target_shape = list(batch_dims_tuple) + list(from_type.dims)
            return self.bb.emit(rx.op.broadcast_to(val, target_shape))
        return val

    # ---- Batched user function calls ----

    def _gen_batched_call(self, callee: str, args: list, call_expr: CallExpr,
                          env: dict[str, rx.Expr]) -> rx.Expr:
        """Emit a batched version of a user function and call it."""
        bd = self._batch_depth
        batch_dims_tuple = tuple(self._batch_dims)
        key = (callee, batch_dims_tuple)

        fn_def = None
        for fn in self.program.functions:
            if fn.name == callee:
                fn_def = fn
                break
        if fn_def is None:
            raise ValueError(f"codegen: unknown function '{callee}'")

        batched_name = self._batched_fns.get(key)
        if batched_name is None:
            batched_name = f"{callee}_vmap_{'x'.join(str(d) for d in batch_dims_tuple)}"
            self._batched_fns[key] = batched_name
            self._emit_batched_function(fn_def, batched_name, batch_dims_tuple)

        arg_vals = []
        for a in args:
            val = self._gen_expr(a, env)
            at = self._type_of(a)
            is_batched = (isinstance(at, ArrayType)
                          and len(at.dims) >= bd
                          and at.dims[:bd] == batch_dims_tuple)
            if not is_batched:
                val = self._broadcast_to_batched(val, at)
            arg_vals.append(val)

        gv = self.bb.get().get_global_var(batched_name)
        return self.bb.emit(rx.Call(gv, arg_vals))

    def _emit_batched_function(self, fn_def: FnDef, name: str, batch_dims: tuple[int, ...]):
        """Emit a batched copy of a function with batch-aware codegen."""
        bd = len(batch_dims)

        saved_batch_depth = self._batch_depth
        saved_batch_dims = self._batch_dims

        self._batch_depth = bd
        self._batch_dims = list(batch_dims)

        params = []
        env: dict[str, rx.Expr] = {}
        for p in fn_def.params:
            orig_type = self._resolve_param_type(p)
            if isinstance(orig_type, ScalarType):
                batched_type = ArrayType(orig_type.base, batch_dims)
            elif isinstance(orig_type, ArrayType):
                batched_type = ArrayType(orig_type.base, batch_dims + orig_type.dims)
            else:
                batched_type = orig_type
            var = rx.Var(p.name, self._relax_type(batched_type))
            params.append(var)
            env[p.name] = var

        body_copy = copy.deepcopy(fn_def.body)
        self._copy_type_map(fn_def.body, body_copy)

        for dim in reversed(batch_dims):
            self._lift_body_types(body_copy, dim, set())

        with self.bb.function(name, params):
            with self.bb.dataflow():
                result = self._gen_block(body_copy, env)
                out = self.bb.emit_output(result)
            self.bb.emit_func_output(out)

        self._batch_depth = saved_batch_depth
        self._batch_dims = saved_batch_dims

    def _copy_type_map(self, orig_node, copy_node):
        """Copy type_map entries from original AST nodes to deep-copied nodes."""
        orig_type = self.type_map.get(id(orig_node))
        if orig_type is not None:
            self.type_map[id(copy_node)] = orig_type

        match orig_node:
            case Block(stmts=stmts, expr=expr):
                for os, cs in zip(stmts, copy_node.stmts):
                    self._copy_type_map(os, cs)
                if expr is not None:
                    self._copy_type_map(expr, copy_node.expr)
            case LetStmt(value=val):
                self._copy_type_map(val, copy_node.value)
            case ExprStmt(expr=ex):
                self._copy_type_map(ex, copy_node.expr)
            case BinOp(left=left, right=right):
                self._copy_type_map(left, copy_node.left)
                self._copy_type_map(right, copy_node.right)
            case UnaryOp(operand=operand):
                self._copy_type_map(operand, copy_node.operand)
            case CallExpr(args=args):
                for oa, ca in zip(args, copy_node.args):
                    self._copy_type_map(oa, ca)
            case IfExpr(condition=cond, then_block=tb, else_block=eb):
                self._copy_type_map(cond, copy_node.condition)
                self._copy_type_map(tb, copy_node.then_block)
                self._copy_type_map(eb, copy_node.else_block)
            case MapExpr(sequence=seq, body=body):
                self._copy_type_map(seq, copy_node.sequence)
                self._copy_type_map(body, copy_node.body)
            case StructLiteral(fields=fields):
                for (_, ov), (_, cv) in zip(fields, copy_node.fields):
                    self._copy_type_map(ov, cv)
            case FieldAccess(object=obj):
                self._copy_type_map(obj, copy_node.object)
            case WithExpr(base=base, updates=updates):
                self._copy_type_map(base, copy_node.base)
                for (_, ov), (_, cv) in zip(updates, copy_node.updates):
                    self._copy_type_map(ov, cv)
            case _BroadcastExpr(expr=e):
                self._copy_type_map(e, copy_node.expr)
            case CastExpr(expr=e):
                self._copy_type_map(e, copy_node.expr)
            case _:
                pass

    # ---- Function calls (builtins + user) ----

    _ELEMENTWISE_BUILTINS = {
        "exp": rx.op.exp,
        "log": rx.op.log,
        "tanh": rx.op.tanh,
        "sqrt": rx.op.sqrt,
        "abs": rx.op.abs,
    }

    _CALLBACK_BUILTINS = {"callback"}
    _RNG_BUILTINS = {"rng_key", "rng_split", "rng_uniform", "rng_normal"}

    def _gen_call(self, callee: str, args: list[Expr], call_expr: CallExpr,
                  env: dict[str, rx.Expr]) -> rx.Expr:
        if callee in self._CALLBACK_BUILTINS:
            return rx.const(0, "int32")

        if callee in self._RNG_BUILTINS:
            return self._gen_rng(call_expr, env)

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
        if callee == "iota":
            return self._gen_iota(args)
        if callee == "where":
            return self._gen_where_builtin(args, env)
        if callee == "stop_gradient":
            return self._gen_expr(args[0], env)
        if callee in ("max", "min"):
            return self._gen_max_min(callee, args, env)
        if callee in ("argmax", "argmin"):
            return self._gen_argmax_argmin(callee, args, env)
        if callee == "conv2d":
            return self._gen_conv2d(args, call_expr, env)
        if callee == "max_pool":
            return self._gen_max_pool(args, env)
        if callee == "avg_pool":
            return self._gen_avg_pool(args, env)

        # User-defined function call
        if self._batch_depth > 0:
            return self._gen_batched_call(callee, args, call_expr, env)
        arg_vals = [self._gen_expr(a, env) for a in args]
        gv = self.bb.get().get_global_var(callee)
        return self.bb.emit(rx.Call(gv, arg_vals))

    # ---- Builtin implementations ----

    def _gen_sum(self, args: list[Expr], env: dict[str, rx.Expr]) -> rx.Expr:
        arg = self._gen_expr(args[0], env)
        arg_type = self._type_of(args[0])
        if not isinstance(arg_type, ArrayType):
            return arg
        bd = self._batch_depth
        if len(args) == 2:
            axis = args[1].value
            actual_axis = bd + axis
            return self.bb.emit(rx.op.sum(arg, axis=[actual_axis]))
        axes = list(range(bd, len(arg_type.dims)))
        return self.bb.emit(rx.op.sum(arg, axis=axes))

    def _gen_mean(self, args: list[Expr], call_expr: CallExpr,
                  env: dict[str, rx.Expr]) -> rx.Expr:
        arg = self._gen_expr(args[0], env)
        arg_type = self._type_of(args[0])
        if not isinstance(arg_type, ArrayType):
            return arg
        bd = self._batch_depth
        if len(args) == 2:
            axis = args[1].value
            actual_axis = bd + axis
            return self.bb.emit(rx.op.mean(arg, axis=[actual_axis]))
        axes = list(range(bd, len(arg_type.dims)))
        return self.bb.emit(rx.op.mean(arg, axis=axes))

    def _gen_transpose(self, args: list[Expr], env: dict[str, rx.Expr]) -> rx.Expr:
        arg = self._gen_expr(args[0], env)
        bd = self._batch_depth
        perm = list(range(bd)) + [bd + 1, bd]
        return self.bb.emit(rx.op.permute_dims(arg, perm))

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
        if (isinstance(args[-1], IntLiteral)
                and isinstance(self._type_of(args[-1]), ScalarType)):
            axis = args[-1].value
            array_args = args[:-1]
        else:
            axis = 0
            array_args = args
        arg_vals = [self._gen_expr(a, env) for a in array_args]
        return self.bb.emit(rx.op.concat(arg_vals, axis=axis))

    def _gen_iota(self, args: list[Expr]) -> rx.Expr:
        n = args[0].value
        return self.bb.emit(rx.const(np.arange(n, dtype="int32")))

    def _gen_where_builtin(self, args: list[Expr], env: dict[str, rx.Expr]) -> rx.Expr:
        cond = self._gen_expr(args[0], env)
        x = self._gen_expr(args[1], env)
        y = self._gen_expr(args[2], env)
        return self.bb.emit(rx.op.where(cond, x, y))

    def _gen_max_min(self, callee: str, args: list[Expr],
                     env: dict[str, rx.Expr]) -> rx.Expr:
        arg = self._gen_expr(args[0], env)
        arg_type = self._type_of(args[0])
        op_fn = rx.op.max if callee == "max" else rx.op.min
        if not isinstance(arg_type, ArrayType):
            return arg
        if len(args) == 2:
            axis_val = args[1].value
            return self.bb.emit(op_fn(arg, axis=[axis_val]))
        axes = list(range(len(arg_type.dims)))
        return self.bb.emit(op_fn(arg, axis=axes))

    def _gen_argmax_argmin(self, callee: str, args: list[Expr],
                           env: dict[str, rx.Expr]) -> rx.Expr:
        arg = self._gen_expr(args[0], env)
        arg_type = self._type_of(args[0])
        op_fn = rx.op.argmax if callee == "argmax" else rx.op.argmin
        if not isinstance(arg_type, ArrayType):
            return arg
        if len(args) == 2:
            axis_val = args[1].value
            return self.bb.emit(op_fn(arg, axis=axis_val))
        return self.bb.emit(op_fn(arg, axis=0))

    # ---- Conv2d / Pool forward ops ----

    def _extract_conv2d_params(self, args: list[Expr]) -> tuple[int, int, int, int]:
        nargs = len(args)
        if nargs == 2:
            return (1, 1, 0, 0)
        elif nargs == 4:
            return (args[2].value, args[2].value,
                    args[3].value, args[3].value)
        else:
            return (args[2].value, args[3].value,
                    args[4].value, args[5].value)

    def _gen_conv2d(self, args: list[Expr], call_expr: CallExpr,
                    env: dict[str, rx.Expr]) -> rx.Expr:
        data = self._gen_expr(args[0], env)
        weight = self._gen_expr(args[1], env)
        sh, sw, ph, pw = self._extract_conv2d_params(args)
        return self.bb.emit(
            rx.op.nn.conv2d(
                data, weight,
                strides=(sh, sw),
                padding=(ph, pw),
                data_layout="NCHW",
                kernel_layout="OIHW",
            )
        )

    def _gen_max_pool(self, args: list[Expr], env: dict[str, rx.Expr]) -> rx.Expr:
        data = self._gen_expr(args[0], env)
        wh, ww = args[1].value, args[2].value
        sh, sw = args[3].value, args[4].value
        return self.bb.emit(
            rx.op.nn.max_pool2d(
                data,
                pool_size=(wh, ww),
                strides=(sh, sw),
                layout="NCHW",
            )
        )

    def _gen_avg_pool(self, args: list[Expr], env: dict[str, rx.Expr]) -> rx.Expr:
        data = self._gen_expr(args[0], env)
        wh, ww = args[1].value, args[2].value
        sh, sw = args[3].value, args[4].value
        return self.bb.emit(
            rx.op.nn.avg_pool2d(
                data,
                pool_size=(wh, ww),
                strides=(sh, sw),
                layout="NCHW",
            )
        )

    # ---- Conv2d / Pool backward ops ----

    @staticmethod
    def _dilate_dim(d: int, dilation: int) -> int:
        return max(0, 1 + dilation * (d - 1))

    @staticmethod
    def _conv_vjp_lhs_padding(
        in_spatial: tuple[int, ...], kernel_spatial: tuple[int, ...],
        strides: tuple[int, ...], out_spatial: tuple[int, ...],
        padding: tuple[tuple[int, int], ...],
        lhs_dilation: tuple[int, ...], rhs_dilation: tuple[int, ...],
    ) -> list[tuple[int, int]]:
        result = []
        for i in range(len(in_spatial)):
            ld = RelaxCodegen._dilate_dim(in_spatial[i], lhs_dilation[i])
            rd = RelaxCodegen._dilate_dim(kernel_spatial[i], rhs_dilation[i])
            od = RelaxCodegen._dilate_dim(out_spatial[i], strides[i])
            pad_before = rd - padding[i][0] - 1
            pad_after = ld + rd - 1 - od - pad_before
            result.append((pad_before, pad_after))
        return result

    @staticmethod
    def _conv_vjp_rhs_padding(
        in_spatial: tuple[int, ...], kernel_spatial: tuple[int, ...],
        strides: tuple[int, ...], out_spatial: tuple[int, ...],
        padding: tuple[tuple[int, int], ...],
        lhs_dilation: tuple[int, ...], rhs_dilation: tuple[int, ...],
    ) -> list[tuple[int, int]]:
        result = []
        for i in range(len(in_spatial)):
            ld = RelaxCodegen._dilate_dim(in_spatial[i], lhs_dilation[i])
            rd = RelaxCodegen._dilate_dim(kernel_spatial[i], rhs_dilation[i])
            od = RelaxCodegen._dilate_dim(out_spatial[i], strides[i])
            pad_lo = padding[i][0]
            pads_from_lhs = od - ld
            pads_from_rhs = rd - pad_lo - 1
            pad_hi = pads_from_lhs + pads_from_rhs
            result.append((pad_lo, pad_hi))
        return result

    def _gen_conv2d_grad(self, expr: _Conv2dGrad, env: dict[str, rx.Expr]) -> rx.Expr:
        adj = self._gen_expr(expr.adj, env)
        adj_type = self._type_of(expr.adj)
        input_type = self._type_of(expr.input_expr)
        kernel_type = self._type_of(expr.kernel_expr)
        assert isinstance(input_type, ArrayType) and isinstance(kernel_type, ArrayType)
        assert isinstance(adj_type, ArrayType)

        _, _, H, W = input_type.dims
        _, _, Kh, Kw = kernel_type.dims
        _, _, OH, OW = adj_type.dims
        sh, sw = expr.strides
        ph, pw = expr.padding

        if expr.wrt == "lhs":
            kernel = self._gen_expr(expr.kernel_expr, env)
            kernel_t = self.bb.emit(rx.op.permute_dims(kernel, [1, 0, 2, 3]))
            return self.bb.emit(
                rx.op.nn.conv2d_transpose(
                    adj, kernel_t,
                    strides=(sh, sw),
                    padding=(ph, pw),
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                )
            )
        else:
            input_val = self._gen_expr(expr.input_expr, env)
            fwd_padding = ((ph, ph), (pw, pw))
            vjp_pad = self._conv_vjp_rhs_padding(
                (H, W), (Kh, Kw), (sh, sw), (OH, OW),
                fwd_padding, (1, 1), (1, 1),
            )
            input_t = self.bb.emit(rx.op.permute_dims(input_val, [1, 0, 2, 3]))
            adj_t = self.bb.emit(rx.op.permute_dims(adj, [1, 0, 2, 3]))
            conv_result = self.bb.emit(
                rx.op.nn.conv2d(
                    input_t, adj_t,
                    strides=(1, 1),
                    padding=(vjp_pad[0][0], vjp_pad[1][0],
                             vjp_pad[0][1], vjp_pad[1][1]),
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                )
            )
            return self.bb.emit(rx.op.permute_dims(conv_result, [1, 0, 2, 3]))

    def _gen_max_pool_grad(self, expr: _MaxPoolGrad, env: dict[str, rx.Expr]) -> rx.Expr:
        input_val = self._gen_expr(expr.input_expr, env)
        adj = self._gen_expr(expr.adj, env)
        input_type = self._type_of(expr.input_expr)
        assert isinstance(input_type, ArrayType)

        wh, ww = expr.window
        sh, sw = expr.strides
        _, _, H, W = input_type.dims

        pooled = self.bb.emit(
            rx.op.nn.max_pool2d(
                input_val,
                pool_size=(wh, ww),
                strides=(sh, sw),
                layout="NCHW",
            )
        )
        adj_upsampled = self.bb.emit(
            rx.op.image.resize2d(
                adj,
                size=rx.ShapeExpr([H, W]),
                layout="NCHW",
                method="nearest_neighbor",
            )
        )
        pooled_upsampled = self.bb.emit(
            rx.op.image.resize2d(
                pooled,
                size=rx.ShapeExpr([H, W]),
                layout="NCHW",
                method="nearest_neighbor",
            )
        )
        mask = self.bb.emit(rx.op.equal(input_val, pooled_upsampled))
        mask_f32 = self.bb.emit(rx.op.astype(mask, "float32"))
        return self.bb.emit(rx.op.multiply(adj_upsampled, mask_f32))

    def _gen_avg_pool_grad(self, expr: _AvgPoolGrad, env: dict[str, rx.Expr]) -> rx.Expr:
        adj = self._gen_expr(expr.adj, env)
        input_type = self._type_of(expr.input_expr)
        assert isinstance(input_type, ArrayType)

        wh, ww = expr.window
        _, _, H, W = input_type.dims

        count = float(wh * ww)
        scale = rx.const(1.0 / count, "float32")
        scaled = self.bb.emit(rx.op.multiply(adj, scale))
        return self.bb.emit(
            rx.op.image.resize2d(
                scaled,
                size=rx.ShapeExpr([H, W]),
                layout="NCHW",
                method="nearest_neighbor",
            )
        )

    # ---- RNG builtins ----

    def _gen_rng(self, expr: CallExpr, env: dict[str, rx.Expr]) -> rx.Expr:
        callee = expr.callee
        if callee == "rng_key":
            return self._gen_rng_key(expr, env)
        elif callee == "rng_split":
            return self._gen_rng_split(expr, env)
        elif callee == "rng_uniform":
            return self._gen_rng_uniform(expr, env)
        elif callee == "rng_normal":
            return self._gen_rng_normal(expr, env)
        raise ValueError(f"Unknown RNG builtin: {callee}")

    _TIR_FACTORIES = {
        "xorshift": _make_xorshift_prim,
        "bits_to_uniform": _make_bits_to_uniform_prim,
    }

    def _ensure_tir_func(self, kind: str, n: int) -> str:
        name = f"{kind}_{n}"
        if name not in self._tir_cache:
            prim = self._TIR_FACTORIES[kind](n)
            self.bb.add_func(prim, name)
            self._tir_cache.add(name)
        return name

    def _gen_rng_bits(self, key: rx.Expr, n: int) -> rx.Expr:
        func_name = self._ensure_tir_func("xorshift", n)
        gv = self.bb.get().get_global_var(func_name)
        return self.bb.emit(
            rx.call_tir(gv, [key], out_sinfo=rx.TensorStructInfo((n,), "uint32"))
        )

    def _gen_bits_to_uniform(self, bits: rx.Expr, n: int) -> rx.Expr:
        func_name = self._ensure_tir_func("bits_to_uniform", n)
        gv = self.bb.get().get_global_var(func_name)
        return self.bb.emit(
            rx.call_tir(gv, [bits], out_sinfo=rx.TensorStructInfo((n,), "float32"))
        )

    def _gen_rng_key(self, expr: CallExpr, env: dict[str, rx.Expr]) -> rx.Expr:
        seed = self._gen_expr(expr.args[0], env)
        zeros = rx.const(np.zeros(3, dtype="int32"))
        seed_1d = self.bb.emit(rx.op.reshape(seed, (1,)))
        return self.bb.emit(rx.op.concat([zeros, seed_1d], axis=0))

    def _gen_rng_split(self, expr: CallExpr, env: dict[str, rx.Expr]) -> rx.Expr:
        key = self._gen_expr(expr.args[0], env)
        n = expr.args[1].value
        bits = self._gen_rng_bits(key, n * 4)
        reshaped = self.bb.emit(rx.op.reshape(bits, (n, 4)))
        return self.bb.emit(rx.op.astype(reshaped, "int32"))

    def _gen_rng_uniform(self, expr: CallExpr, env: dict[str, rx.Expr]) -> rx.Expr:
        key = self._gen_expr(expr.args[0], env)
        low = self._gen_expr(expr.args[1], env)
        high = self._gen_expr(expr.args[2], env)

        result_type = self._type_of(expr)
        assert isinstance(result_type, ArrayType)
        shape = tuple(result_type.dims)
        numel = math.prod(shape)

        bits = self._gen_rng_bits(key, numel)
        uniform = self._gen_bits_to_uniform(bits, numel)

        range_val = self.bb.emit(rx.op.subtract(high, low))
        range_bc = self.bb.emit(rx.op.broadcast_to(range_val, (numel,)))
        low_bc = self.bb.emit(rx.op.broadcast_to(low, (numel,)))
        scaled = self.bb.emit(rx.op.multiply(uniform, range_bc))
        result = self.bb.emit(rx.op.add(scaled, low_bc))

        if len(shape) == 1 and shape[0] == numel:
            return result
        return self.bb.emit(rx.op.reshape(result, shape))

    def _gen_rng_normal(self, expr: CallExpr, env: dict[str, rx.Expr]) -> rx.Expr:
        key = self._gen_expr(expr.args[0], env)
        mean = self._gen_expr(expr.args[1], env)
        std = self._gen_expr(expr.args[2], env)

        result_type = self._type_of(expr)
        assert isinstance(result_type, ArrayType)
        shape = tuple(result_type.dims)
        numel = math.prod(shape)

        bits = self._gen_rng_bits(key, 2 * numel)
        uniform_2n = self._gen_bits_to_uniform(bits, 2 * numel)

        u1 = self.bb.emit(rx.op.strided_slice(uniform_2n, axes=[0], begin=[0], end=[numel]))
        u2 = self.bb.emit(rx.op.strided_slice(uniform_2n, axes=[0], begin=[numel], end=[2 * numel]))

        eps = rx.const(np.float32(1e-7))
        eps_bc = self.bb.emit(rx.op.broadcast_to(eps, (numel,)))
        u1_safe = self.bb.emit(rx.op.maximum(u1, eps_bc))

        log_u1 = self.bb.emit(rx.op.log(u1_safe))
        neg2 = rx.const(np.float32(-2.0))
        neg2_bc = self.bb.emit(rx.op.broadcast_to(neg2, (numel,)))
        neg2_log = self.bb.emit(rx.op.multiply(neg2_bc, log_u1))
        radius = self.bb.emit(rx.op.sqrt(neg2_log))

        two_pi = rx.const(np.float32(2.0 * math.pi))
        two_pi_bc = self.bb.emit(rx.op.broadcast_to(two_pi, (numel,)))
        angle = self.bb.emit(rx.op.multiply(two_pi_bc, u2))
        cos_angle = self.bb.emit(rx.op.cos(angle))

        z_flat = self.bb.emit(rx.op.multiply(radius, cos_angle))

        std_bc = self.bb.emit(rx.op.broadcast_to(std, (numel,)))
        mean_bc = self.bb.emit(rx.op.broadcast_to(mean, (numel,)))
        scaled = self.bb.emit(rx.op.multiply(z_flat, std_bc))
        result = self.bb.emit(rx.op.add(scaled, mean_bc))

        if len(shape) == 1 and shape[0] == numel:
            return result
        return self.bb.emit(rx.op.reshape(result, shape))
