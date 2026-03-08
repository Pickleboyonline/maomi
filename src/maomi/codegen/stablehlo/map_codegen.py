from __future__ import annotations

import copy

from ...ast_nodes import (
    Block,
    LetStmt,
    ExprStmt,
    IntLiteral,
    FloatLiteral,
    BoolLiteral,
    Identifier,
    UnaryOp,
    BinOp,
    IfExpr,
    CallExpr,
    ScanExpr,
    MapExpr,
    FnDef,
    IndexExpr,
    StructLiteral,
    FieldAccess,
    WithExpr,
    _BroadcastExpr,
    _ReduceSum,
)
from ...types import MaomiType, ScalarType, ArrayType
from ...errors import MaomiError
from .utils import _mlir_type, _collect_body_refs


class MapCodegenMixin:

    def _gen_batched_call(self, expr: CallExpr, env: dict[str, str]) -> str:
        """Emit a batched version of a user function and call it."""
        bd = self._batch_depth
        batch_dims_tuple = tuple(self._batch_dims)
        key = (expr.callee, batch_dims_tuple)

        # Find the FnDef
        fn_def = None
        for fn in self.program.functions:
            if fn.name == expr.callee:
                fn_def = fn
                break
        if fn_def is None:
            raise MaomiError(f"codegen: unknown function '{expr.callee}'", "<codegen>", 0, 0)

        batched_name = self._batched_fns.get(key)
        if batched_name is None:
            batched_name = f"{expr.callee}_vmap_{'x'.join(str(d) for d in batch_dims_tuple)}"
            self._batched_fns[key] = batched_name
            self._emit_batched_function(fn_def, batched_name, batch_dims_tuple)

        # Generate args — broadcast unbatched free vars to add batch dims
        args = []
        arg_types = []
        for a in expr.args:
            ssa = self._gen_expr(a, env)
            at = self._type_of(a)
            # Check if this arg is batched (its type was lifted)
            is_batched = isinstance(at, ArrayType) and len(at.dims) >= bd and at.dims[:bd] == batch_dims_tuple
            if not is_batched:
                ssa = self._broadcast_to_batched(ssa, at)
                if isinstance(at, ScalarType):
                    at = ArrayType(at.base, batch_dims_tuple)
                elif isinstance(at, ArrayType):
                    at = ArrayType(at.base, batch_dims_tuple + at.dims)
            args.append(ssa)
            arg_types.append(at)

        result_type = self._type_of(expr)
        args_str = ", ".join(args)
        types_str = ", ".join(_mlir_type(t) for t in arg_types)
        var = self._fresh()
        self._emit(f"{var} = func.call @{batched_name}({args_str}) : ({types_str}) -> {_mlir_type(result_type)}")
        return var

    def _emit_batched_function(self, fn_def: FnDef, name: str, batch_dims: tuple[int, ...]):
        """Emit a batched copy of a function with batch-aware codegen."""
        bd = len(batch_dims)

        # Save codegen state
        saved_lines = self._lines
        saved_counter = self._counter
        saved_indent = self._indent
        saved_batch_depth = self._batch_depth
        saved_batch_dims = self._batch_dims

        # Set up for batched function emission
        self._lines = []
        self._counter = 0
        self._indent = 1  # Inside module
        self._batch_depth = bd
        self._batch_dims = list(batch_dims)

        # Build batched param types (all params get batch dims prepended)
        env: dict[str, str] = {}
        params = []
        for i, p in enumerate(fn_def.params):
            ssa = f"%arg{i}"
            orig_type = self._resolve_param_type(p)
            if isinstance(orig_type, ScalarType):
                batched_type = ArrayType(orig_type.base, batch_dims)
            elif isinstance(orig_type, ArrayType):
                batched_type = ArrayType(orig_type.base, batch_dims + orig_type.dims)
            else:
                batched_type = orig_type
            params.append(f"{ssa}: {_mlir_type(batched_type)}")
            env[p.name] = ssa
            # Update type_map for params in the batched context
            self.type_map[id(p)] = batched_type

        # Batched return type
        orig_ret = self._resolve_annotation_type(fn_def.return_type)
        if isinstance(orig_ret, ScalarType):
            batched_ret = ArrayType(orig_ret.base, batch_dims)
        elif isinstance(orig_ret, ArrayType):
            batched_ret = ArrayType(orig_ret.base, batch_dims + orig_ret.dims)
        else:
            batched_ret = orig_ret

        params_str = ", ".join(params)
        self._emit(f"func.func @{name}({params_str}) -> {_mlir_type(batched_ret)} {{")
        self._indent += 1

        # Lift body types for the batched function
        body_copy = copy.deepcopy(fn_def.body)
        self._copy_type_map(fn_def.body, body_copy)
        # All variables inside are batched, no free vars to skip
        for dim in reversed(batch_dims):
            self._lift_body_types(body_copy, dim, set())

        result = self._gen_block(body_copy, env)
        self._emit(f"return {result} : {_mlir_type(batched_ret)}")

        self._indent -= 1
        self._emit("}")

        # Capture emitted lines
        fn_lines = self._lines

        # Restore codegen state
        self._lines = saved_lines
        self._counter = saved_counter
        self._indent = saved_indent
        self._batch_depth = saved_batch_depth
        self._batch_dims = saved_batch_dims

        # Insert batched function before current position in the module
        # Find the last "func.func" line to insert before it
        insert_idx = 0
        for i, line in enumerate(self._lines):
            if "func.func @" in line:
                insert_idx = i
                break
        for i, line in enumerate(fn_lines):
            self._lines.insert(insert_idx + i, line)

    def _copy_type_map(self, orig_node, copy_node):
        """Copy type_map entries from original AST nodes to deep-copied nodes."""
        orig_type = self.type_map.get(id(orig_node))
        if orig_type is not None:
            self.type_map[id(copy_node)] = orig_type

        # Recurse into sub-nodes
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
            case IndexExpr(base=base, indices=indices):
                self._copy_type_map(base, copy_node.base)
                for oi, ci in zip(indices, copy_node.indices):
                    if oi.value is not None:
                        self._copy_type_map(oi.value, ci.value)
                    if oi.start is not None:
                        self._copy_type_map(oi.start, ci.start)
                    if oi.end is not None:
                        self._copy_type_map(oi.end, ci.end)
            case MapExpr(sequence=seq, body=body):
                self._copy_type_map(seq, copy_node.sequence)
                self._copy_type_map(body, copy_node.body)
            case ScanExpr(init=init, sequences=seqs, body=body):
                self._copy_type_map(init, copy_node.init)
                for os, cs in zip(seqs, copy_node.sequences):
                    self._copy_type_map(os, cs)
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
            case _ReduceSum(expr=e):
                self._copy_type_map(e, copy_node.expr)
            case _:
                pass

    def _gen_map(self, expr: MapExpr, env: dict[str, str]) -> str:
        seq_val = self._gen_expr(expr.sequence, env)
        seq_type = self._type_of(expr.sequence)

        if not isinstance(seq_type, ArrayType):
            raise MaomiError("codegen: map sequence must be array", "<codegen>", 0, 0)

        batch_dim = seq_type.dims[0]

        # Compute free variables (captured from outer scope, NOT the elem_var)
        body_refs = _collect_body_refs(expr.body)
        free_vars = (body_refs - {expr.elem_var}) & set(env.keys())

        # Push batch context
        self._batch_depth += 1
        self._batch_dims.append(batch_dim)

        # Lift body types, skipping free variables
        self._lift_body_types(expr.body, batch_dim, free_vars)

        body_env = dict(env)
        body_env[expr.elem_var] = seq_val
        result = self._gen_block(expr.body, body_env)

        # Pop batch context
        self._batch_depth -= 1
        self._batch_dims.pop()
        return result

    def _lift_body_types(self, block, batch_dim, free_vars: set[str]):
        """Prepend batch_dim to all types in a block for map codegen."""
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                self._lift_expr_type(stmt.value, batch_dim, free_vars)
            elif isinstance(stmt, ExprStmt):
                self._lift_expr_type(stmt.expr, batch_dim, free_vars)
        if block.expr is not None:
            self._lift_expr_type(block.expr, batch_dim, free_vars)

    def _lift_expr_type(self, expr, batch_dim, free_vars: set[str]):
        """Recursively lift an expression's type to include batch_dim.
        Literals stay scalar. Free variables are not lifted."""
        # Don't lift literals — they remain scalar and broadcast naturally
        if isinstance(expr, (IntLiteral, FloatLiteral, BoolLiteral)):
            return

        # Don't lift free variables — they keep their outer-scope type
        if isinstance(expr, Identifier) and expr.name in free_vars:
            return

        # Don't lift MapExpr own type — it manages its own batch context
        # ScanExpr DOES get lifted — its body runs within the outer batch context
        is_inner_compound = isinstance(expr, MapExpr)

        if not is_inner_compound:
            t = self.type_map.get(id(expr))
            if t is not None:
                if isinstance(t, ScalarType):
                    self.type_map[id(expr)] = ArrayType(t.base, (batch_dim,))
                elif isinstance(t, ArrayType):
                    self.type_map[id(expr)] = ArrayType(t.base, (batch_dim,) + t.dims)

        # Recurse into sub-expressions
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
            case IndexExpr(base=base, indices=indices):
                self._lift_expr_type(base, batch_dim, free_vars)
                for ic in indices:
                    if ic.value is not None:
                        self._lift_expr_type(ic.value, batch_dim, free_vars)
                    if ic.start is not None:
                        self._lift_expr_type(ic.start, batch_dim, free_vars)
                    if ic.end is not None:
                        self._lift_expr_type(ic.end, batch_dim, free_vars)
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
                # Lift the sequence (it may reference outer batched vars)
                # but do NOT lift the inner map's body — it manages its own batch
                self._lift_expr_type(seq, batch_dim, free_vars)
            case ScanExpr(init=init, sequences=seqs, body=body):
                self._lift_expr_type(init, batch_dim, free_vars)
                for s in seqs:
                    self._lift_expr_type(s, batch_dim, free_vars)
                # Lift body expressions — scan body runs in the outer batch context
                self._lift_body_types(body, batch_dim, free_vars)
            case _BroadcastExpr(expr=e):
                self._lift_expr_type(e, batch_dim, free_vars)
            case _ReduceSum(expr=e):
                self._lift_expr_type(e, batch_dim, free_vars)
            case _:
                pass

    def _broadcast_to_batched(self, ssa: str, from_type: MaomiType) -> str:
        """Broadcast an unbatched value to include batch dims at front."""
        batch_dims_tuple = tuple(self._batch_dims)
        if isinstance(from_type, ScalarType):
            to_type = ArrayType(from_type.base, batch_dims_tuple)
            broadcast_dims: list[int] = []
        elif isinstance(from_type, ArrayType):
            to_type = ArrayType(from_type.base, batch_dims_tuple + from_type.dims)
            bd = self._batch_depth
            broadcast_dims = list(range(bd, bd + len(from_type.dims)))
        else:
            return ssa
        dims_str = ", ".join(str(d) for d in broadcast_dims)
        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.broadcast_in_dim {ssa}, "
            f"dims = [{dims_str}] : ({_mlir_type(from_type)}) -> {_mlir_type(to_type)}"
        )
        return var

    # -- Broadcasting helpers --

    def _maybe_broadcast(self, ssa: str, from_type: MaomiType, to_type: MaomiType) -> str:
        """Insert broadcast_in_dim if from_type needs to be broadcast to to_type."""
        from .utils import _types_equal
        if _types_equal(from_type, to_type):
            return ssa

        if isinstance(from_type, ScalarType) and isinstance(to_type, ArrayType):
            var = self._fresh()
            self._emit(
                f"{var} = stablehlo.broadcast_in_dim {ssa}, "
                f"dims = [] : ({_mlir_type(from_type)}) -> {_mlir_type(to_type)}"
            )
            return var

        if isinstance(from_type, ArrayType) and isinstance(to_type, ArrayType):
            if from_type.dims != to_type.dims:
                # Compute broadcast dims: right-align from_type dims to to_type dims
                offset = len(to_type.dims) - len(from_type.dims)
                dims = list(range(offset, offset + len(from_type.dims)))
                dims_str = ", ".join(str(d) for d in dims)
                var = self._fresh()
                self._emit(
                    f"{var} = stablehlo.broadcast_in_dim {ssa}, "
                    f"dims = [{dims_str}] : ({_mlir_type(from_type)}) -> {_mlir_type(to_type)}"
                )
                return var

        return ssa
