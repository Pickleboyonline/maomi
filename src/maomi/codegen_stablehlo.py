from __future__ import annotations

from .ast_nodes import (
    Program,
    FnDef,
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
    StructLiteral,
    FieldAccess,
    WithExpr,
    IndexExpr,
    IndexComponent,
    _ScanGrad,
    _IndexGrad,
    Expr,
)
from .types import MaomiType, ScalarType, ArrayType, StructType
from .errors import MaomiError


# Maps Maomi base types to MLIR element types
_MLIR_ETYPE = {
    "f32": "f32",
    "f64": "f64",
    "i32": "i32",
    "i64": "i64",
    "bool": "i1",
}

_COMPARISON_MAP = {
    "==": "EQ",
    "!=": "NE",
    "<": "LT",
    ">": "GT",
    "<=": "LE",
    ">=": "GE",
}

_BUILTIN_OPS = {
    "exp": "stablehlo.exponential",
    "log": "stablehlo.log",
    "tanh": "stablehlo.tanh",
    "sqrt": "stablehlo.sqrt",
    "abs": "stablehlo.abs",
}


def _mlir_type(t: MaomiType) -> str:
    """Convert a MaomiType to an MLIR tensor type string."""
    if isinstance(t, ScalarType):
        return f"tensor<{_MLIR_ETYPE[t.base]}>"
    if isinstance(t, ArrayType):
        for d in t.dims:
            if isinstance(d, str):
                raise MaomiError(
                    f"codegen: unresolved symbolic dimension '{d}' — all dimensions must be concrete",
                    "<codegen>", 0, 0,
                )
        shape = "x".join(str(d) for d in t.dims)
        return f"tensor<{shape}x{_MLIR_ETYPE[t.base]}>"
    if isinstance(t, StructType):
        field_types = ", ".join(_mlir_type(ft) for _, ft in t.fields)
        return f"tuple<{field_types}>"
    raise MaomiError("codegen: unknown type", "<codegen>", 0, 0)


class StableHLOCodegen:
    def __init__(self, program: Program, type_map: dict[int, MaomiType] | None = None):
        self.program = program
        self.type_map = type_map or {}
        self._counter = 0
        self._lines: list[str] = []
        self._indent = 0

    def generate(self) -> str:
        self._emit("module {")
        self._indent += 1
        for fn in self.program.functions:
            self._gen_function(fn)
        self._indent -= 1
        self._emit("}")
        return "\n".join(self._lines)

    # -- Helpers --

    def _fresh(self) -> str:
        name = f"%{self._counter}"
        self._counter += 1
        return name

    def _emit(self, line: str):
        self._lines.append("  " * self._indent + line)

    def _type_of(self, expr: Expr) -> MaomiType:
        t = self.type_map.get(id(expr))
        if t is None:
            raise MaomiError(
                "codegen: missing type info for expression",
                "<codegen>",
                expr.span.line_start,
                expr.span.col_start,
            )
        return t

    # -- Function generation --

    def _gen_function(self, fn: FnDef):
        env: dict[str, str] = {}  # maomi variable name -> SSA name

        # Build parameter list
        params = []
        for i, p in enumerate(fn.params):
            ssa = f"%arg{i}"
            t = self.type_map.get(id(p)) if hasattr(p, 'span') else None
            # Look up param type from the fn signature in the type_map
            # We need to resolve param types from annotations
            param_type = self._resolve_param_type(p)
            params.append(f"{ssa}: {_mlir_type(param_type)}")
            env[p.name] = ssa

        ret_type = self._resolve_annotation_type(fn.return_type)
        params_str = ", ".join(params)
        self._emit(f"func.func @{fn.name}({params_str}) -> {_mlir_type(ret_type)} {{")
        self._indent += 1

        result = self._gen_block(fn.body, env)
        self._emit(f"return {result} : {_mlir_type(ret_type)}")

        self._indent -= 1
        self._emit("}")

    def _resolve_param_type(self, p) -> MaomiType:
        return self._resolve_annotation_type(p.type_annotation)

    def _resolve_annotation_type(self, ta) -> MaomiType:
        if ta.base in ("f32", "f64", "i32", "i64", "bool"):
            if ta.dims is None:
                return ScalarType(ta.base)
            dims = tuple(d.value for d in ta.dims)
            return ArrayType(ta.base, dims)
        # Struct type — look up from program's struct_defs via type_map or build from AST
        for sd in self.program.struct_defs:
            if sd.name == ta.base:
                field_types = []
                for field_name, field_ta in sd.fields:
                    field_types.append((field_name, self._resolve_annotation_type(field_ta)))
                return StructType(sd.name, tuple(field_types))
        # Fallback: treat as scalar (will likely error later)
        if ta.dims is None:
            return ScalarType(ta.base)
        dims = tuple(d.value for d in ta.dims)
        return ArrayType(ta.base, dims)

    # -- Block generation --

    def _gen_block(self, block: Block, env: dict[str, str]) -> str:
        child_env = dict(env)
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                val = self._gen_expr(stmt.value, child_env)
                child_env[stmt.name] = val
            elif isinstance(stmt, ExprStmt):
                self._gen_expr(stmt.expr, child_env)

        if block.expr is not None:
            return self._gen_expr(block.expr, child_env)
        raise MaomiError("codegen: block has no return expression", "<codegen>", block.span.line_start, block.span.col_start)

    # -- Expression generation --

    def _gen_expr(self, expr: Expr, env: dict[str, str]) -> str:
        match expr:
            case IntLiteral(value=v):
                return self._gen_literal(v, self._type_of(expr))
            case FloatLiteral(value=v):
                return self._gen_literal(v, self._type_of(expr))
            case BoolLiteral(value=v):
                return self._gen_literal(v, self._type_of(expr))
            case Identifier(name=name):
                if name not in env:
                    raise MaomiError(
                        f"codegen: undefined variable '{name}'",
                        "<codegen>", expr.span.line_start, expr.span.col_start,
                    )
                return env[name]
            case UnaryOp(op=op, operand=operand):
                return self._gen_unary(op, operand, expr, env)
            case BinOp(op=op, left=left, right=right):
                return self._gen_binop(op, left, right, expr, env)
            case IfExpr():
                return self._gen_if(expr, env)
            case CallExpr():
                return self._gen_call(expr, env)
            case ScanExpr():
                return self._gen_scan(expr, env)
            case MapExpr():
                return self._gen_map(expr, env)
            case StructLiteral():
                return self._gen_struct_literal(expr, env)
            case FieldAccess():
                return self._gen_field_access(expr, env)
            case WithExpr():
                return self._gen_with(expr, env)
            case IndexExpr():
                return self._gen_index(expr, env)
            case _ScanGrad():
                return self._gen_scan_grad(expr, env)
            case _IndexGrad():
                return self._gen_index_grad(expr, env)
            case _:
                raise MaomiError(
                    f"codegen: unsupported expression type {type(expr).__name__}",
                    "<codegen>", expr.span.line_start, expr.span.col_start,
                )

    def _gen_literal(self, value, result_type: MaomiType) -> str:
        var = self._fresh()
        mlir_t = _mlir_type(result_type)
        if isinstance(value, bool):
            val_str = "true" if value else "false"
        elif isinstance(value, float):
            val_str = f"{value:e}"
        else:
            val_str = str(value)
        self._emit(f"{var} = stablehlo.constant dense<{val_str}> : {mlir_t}")
        return var

    def _gen_unary(self, op: str, operand: Expr, expr: Expr, env: dict[str, str]) -> str:
        val = self._gen_expr(operand, env)
        result_type = self._type_of(expr)
        mlir_t = _mlir_type(result_type)
        var = self._fresh()
        if op == "-":
            self._emit(f"{var} = stablehlo.negate {val} : {mlir_t}")
        return var

    def _gen_binop(self, op: str, left: Expr, right: Expr, expr: Expr, env: dict[str, str]) -> str:
        if op == "@":
            return self._gen_matmul(left, right, expr, env)

        lhs = self._gen_expr(left, env)
        rhs = self._gen_expr(right, env)

        lt = self._type_of(left)
        rt = self._type_of(right)
        result_type = self._type_of(expr)
        mlir_result = _mlir_type(result_type)

        var = self._fresh()

        if op in _COMPARISON_MAP:
            cmp = _COMPARISON_MAP[op]
            # Broadcast operands to a common numeric type (NOT the bool result)
            if isinstance(result_type, ArrayType):
                operand_type = ArrayType(_base_of_type(lt), result_type.dims)
            else:
                operand_type = lt if isinstance(lt, ScalarType) else rt
            lhs = self._maybe_broadcast(lhs, lt, operand_type)
            rhs = self._maybe_broadcast(rhs, rt, operand_type)
            mlir_operand = _mlir_type(operand_type)
            self._emit(
                f"{var} = stablehlo.compare {cmp}, {lhs}, {rhs} "
                f": ({mlir_operand}, {mlir_operand}) -> {mlir_result}"
            )
            return var

        # Broadcast for arithmetic ops
        lhs = self._maybe_broadcast(lhs, lt, result_type)
        rhs = self._maybe_broadcast(rhs, rt, result_type)

        op_map = {
            "+": "stablehlo.add",
            "-": "stablehlo.subtract",
            "*": "stablehlo.multiply",
            "/": "stablehlo.divide",
            "**": "stablehlo.power",
        }
        stablehlo_op = op_map.get(op)
        if stablehlo_op is None:
            raise MaomiError(f"codegen: unsupported op '{op}'", "<codegen>", expr.span.line_start, expr.span.col_start)

        self._emit(f"{var} = {stablehlo_op} {lhs}, {rhs} : {mlir_result}")
        return var

    def _gen_matmul(self, left: Expr, right: Expr, expr: Expr, env: dict[str, str]) -> str:
        lhs = self._gen_expr(left, env)
        rhs = self._gen_expr(right, env)

        lt = self._type_of(left)
        rt = self._type_of(right)
        result_type = self._type_of(expr)

        if not isinstance(lt, ArrayType) or not isinstance(rt, ArrayType):
            raise MaomiError("codegen: matmul requires array operands", "<codegen>", expr.span.line_start, expr.span.col_start)

        # Contracting dimensions: last of left, first of right
        l_contract = len(lt.dims) - 1
        r_contract = 0

        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.dot_general {lhs}, {rhs}, "
            f"contracting_dims = [{l_contract}] x [{r_contract}], "
            f"precision = [DEFAULT, DEFAULT] "
            f": ({_mlir_type(lt)}, {_mlir_type(rt)}) -> {_mlir_type(result_type)}"
        )
        return var

    def _gen_if(self, expr: IfExpr, env: dict[str, str]) -> str:
        cond = self._gen_expr(expr.condition, env)
        then_val = self._gen_block(expr.then_block, env)
        else_val = self._gen_block(expr.else_block, env)

        result_type = self._type_of(expr)
        cond_type = self._type_of(expr.condition)
        mlir_result = _mlir_type(result_type)

        # Broadcast then/else to result type if needed
        then_type = self._type_of(expr.then_block.expr) if expr.then_block.expr else result_type
        else_type = self._type_of(expr.else_block.expr) if expr.else_block.expr else result_type
        then_val = self._maybe_broadcast(then_val, then_type, result_type)
        else_val = self._maybe_broadcast(else_val, else_type, result_type)

        # If cond is scalar bool but result is array, broadcast cond
        if isinstance(cond_type, ScalarType) and isinstance(result_type, ArrayType):
            broadcast_cond = self._fresh()
            bool_array_type = ArrayType("bool", result_type.dims)
            self._emit(
                f"{broadcast_cond} = stablehlo.broadcast_in_dim {cond}, "
                f"dims = [] : ({_mlir_type(cond_type)}) -> {_mlir_type(bool_array_type)}"
            )
            cond = broadcast_cond

        var = self._fresh()
        # Determine cond type for select signature
        if isinstance(cond_type, ScalarType) and isinstance(result_type, ArrayType):
            cond_mlir = _mlir_type(ArrayType("bool", result_type.dims))
        elif isinstance(cond_type, ArrayType):
            cond_mlir = _mlir_type(cond_type)
        else:
            cond_mlir = _mlir_type(ScalarType("bool"))
        self._emit(
            f"{var} = stablehlo.select {cond}, {then_val}, {else_val} "
            f": ({cond_mlir}, {mlir_result}, {mlir_result}) -> {mlir_result}"
        )
        return var

    _CALLBACK_BUILTINS = {"callback"}

    def _gen_call(self, expr: CallExpr, env: dict[str, str]) -> str:
        # Callback: no-op in codegen (host callbacks are future work)
        if expr.callee in self._CALLBACK_BUILTINS:
            self._emit(f"// callback (no-op)")
            return ""

        # Handle builtins
        if expr.callee in _BUILTIN_OPS:
            return self._gen_elementwise_builtin(expr, env)
        if expr.callee == "mean":
            return self._gen_mean(expr, env)
        if expr.callee == "sum":
            return self._gen_sum(expr, env)
        if expr.callee == "transpose":
            return self._gen_transpose(expr, env)

        # User-defined function call
        args = [self._gen_expr(a, env) for a in expr.args]
        arg_types = [self._type_of(a) for a in expr.args]
        result_type = self._type_of(expr)

        args_str = ", ".join(args)
        types_str = ", ".join(_mlir_type(t) for t in arg_types)
        var = self._fresh()
        self._emit(f"{var} = func.call @{expr.callee}({args_str}) : ({types_str}) -> {_mlir_type(result_type)}")
        return var

    def _gen_elementwise_builtin(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        result_type = self._type_of(expr)
        mlir_t = _mlir_type(result_type)
        op = _BUILTIN_OPS[expr.callee]
        var = self._fresh()
        self._emit(f"{var} = {op} {arg} : {mlir_t}")
        return var

    def _gen_mean(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)

        if not isinstance(arg_type, ArrayType):
            return arg  # mean of scalar is itself

        # Sum over all dimensions
        sum_var = self._gen_reduce_sum(arg, arg_type, result_type)

        # Compute element count
        numel = 1
        for d in arg_type.dims:
            if isinstance(d, int):
                numel *= d
            else:
                raise MaomiError(f"codegen: cannot compute mean with symbolic dim '{d}'", "<codegen>", 0, 0)

        # Divide by count
        count_var = self._fresh()
        mlir_result = _mlir_type(result_type)
        self._emit(f"{count_var} = stablehlo.constant dense<{float(numel):e}> : {mlir_result}")

        var = self._fresh()
        self._emit(f"{var} = stablehlo.divide {sum_var}, {count_var} : {mlir_result}")
        return var

    def _gen_sum(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)

        if not isinstance(arg_type, ArrayType):
            return arg

        return self._gen_reduce_sum(arg, arg_type, result_type)

    def _gen_reduce_sum(self, arg: str, arg_type: ArrayType, result_type: MaomiType) -> str:
        """Generate a sum reduction over all dimensions."""
        ndims = len(arg_type.dims)
        dims_str = ", ".join(str(i) for i in range(ndims))

        # Initial value (0)
        init_var = self._fresh()
        scalar_type = ScalarType(arg_type.base)
        mlir_scalar = _mlir_type(scalar_type)
        self._emit(f"{init_var} = stablehlo.constant dense<0.000000e+00> : {mlir_scalar}")

        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.reduce({arg} init: {init_var}) "
            f"across dimensions = [{dims_str}] "
            f": ({_mlir_type(arg_type)}, {mlir_scalar}) -> {mlir_scalar}"
        )
        # Emit reducer region
        self._indent += 1
        a_var = self._fresh()
        b_var = self._fresh()
        self._emit(f"reducer({a_var}: {mlir_scalar}, {b_var}: {mlir_scalar}) {{")
        self._indent += 1
        sum_var = self._fresh()
        self._emit(f"{sum_var} = stablehlo.add {a_var}, {b_var} : {mlir_scalar}")
        self._emit(f"stablehlo.return {sum_var} : {mlir_scalar}")
        self._indent -= 1
        self._emit("}")
        self._indent -= 1
        return var

    def _gen_transpose(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)

        if not isinstance(arg_type, ArrayType) or len(arg_type.dims) != 2:
            raise MaomiError("codegen: transpose requires 2D array", "<codegen>", expr.span.line_start, expr.span.col_start)

        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.transpose {arg}, dims = [1, 0] "
            f": ({_mlir_type(arg_type)}) -> {_mlir_type(result_type)}"
        )
        return var

    # -- Struct codegen --

    def _gen_struct_literal(self, expr: StructLiteral, env: dict[str, str]) -> str:
        result_type = self._type_of(expr)
        field_vals = [self._gen_expr(fv, env) for _, fv in expr.fields]
        var = self._fresh()
        vals_str = ", ".join(field_vals)
        self._emit(f"{var} = stablehlo.tuple {vals_str} : {_mlir_type(result_type)}")
        return var

    def _gen_field_access(self, expr: FieldAccess, env: dict[str, str]) -> str:
        obj = self._gen_expr(expr.object, env)
        obj_type = self._type_of(expr.object)
        result_type = self._type_of(expr)
        if not isinstance(obj_type, StructType):
            raise MaomiError("codegen: field access on non-struct", "<codegen>", 0, 0)
        field_idx = next(i for i, (fn, _) in enumerate(obj_type.fields) if fn == expr.field)
        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.get_tuple_element {obj}[{field_idx}] "
            f": ({_mlir_type(obj_type)}) -> {_mlir_type(result_type)}"
        )
        return var

    def _gen_with(self, expr: WithExpr, env: dict[str, str]) -> str:
        base = self._gen_expr(expr.base, env)
        base_type = self._type_of(expr.base)
        if not isinstance(base_type, StructType):
            raise MaomiError("codegen: 'with' on non-struct", "<codegen>", 0, 0)
        return self._gen_with_struct(base, base_type, expr.updates, env)

    def _gen_with_struct(self, base_ssa: str, stype: StructType,
                          updates: list[tuple[list[str], 'Expr']],
                          env: dict[str, str]) -> str:
        """Reconstruct a struct tuple with some fields updated."""
        # Group updates by top-level field
        top_updates: dict[str, list[tuple[list[str], 'Expr']]] = {}
        for path, value_expr in updates:
            top = path[0]
            rest = path[1:]
            if top not in top_updates:
                top_updates[top] = []
            top_updates[top].append((rest, value_expr))

        # Build field values for the new tuple
        field_vals = []
        for i, (field_name, field_type) in enumerate(stype.fields):
            if field_name in top_updates:
                field_updates = top_updates[field_name]
                if any(len(rest) == 0 for rest, _ in field_updates):
                    # Direct replacement: path is just [field_name]
                    _, value_expr = next((rest, ve) for rest, ve in field_updates if len(rest) == 0)
                    field_vals.append(self._gen_expr(value_expr, env))
                else:
                    # Nested update: extract current field, recurse
                    if not isinstance(field_type, StructType):
                        raise MaomiError(f"codegen: nested 'with' on non-struct field '{field_name}'", "<codegen>", 0, 0)
                    extracted = self._fresh()
                    self._emit(
                        f"{extracted} = stablehlo.get_tuple_element {base_ssa}[{i}] "
                        f": ({_mlir_type(stype)}) -> {_mlir_type(field_type)}"
                    )
                    field_vals.append(self._gen_with_struct(extracted, field_type, field_updates, env))
            else:
                # Unchanged field: extract from base
                extracted = self._fresh()
                self._emit(
                    f"{extracted} = stablehlo.get_tuple_element {base_ssa}[{i}] "
                    f": ({_mlir_type(stype)}) -> {_mlir_type(field_type)}"
                )
                field_vals.append(extracted)

        var = self._fresh()
        vals_str = ", ".join(field_vals)
        self._emit(f"{var} = stablehlo.tuple {vals_str} : {_mlir_type(stype)}")
        return var

    # -- Index codegen --

    def _gen_index(self, expr: IndexExpr, env: dict[str, str]) -> str:
        base_ssa = self._gen_expr(expr.base, env)
        base_type = self._type_of(expr.base)
        result_type = self._type_of(expr)

        if not isinstance(base_type, ArrayType):
            raise MaomiError("codegen: indexing non-array", "<codegen>", expr.span.line_start, expr.span.col_start)

        # Build per-dimension start indices and slice sizes
        start_ssas: list[str] = []
        slice_sizes: list[int] = []
        squeezed_axes: list[int] = []  # axes to remove (single-indexed)
        all_static = True

        for i, ic in enumerate(expr.indices):
            dim = base_type.dims[i]
            if ic.kind == "single":
                idx_ssa = self._gen_expr(ic.value, env)
                start_ssas.append(idx_ssa)
                slice_sizes.append(1)
                squeezed_axes.append(i)
                if not isinstance(ic.value, IntLiteral):
                    all_static = False
            elif ic.kind == "slice":
                start_val = ic.start.value  # IntLiteral guaranteed by type checker
                end_val = ic.end.value
                s = self._gen_literal(start_val, ScalarType("i32"))
                start_ssas.append(s)
                slice_sizes.append(end_val - start_val)
            elif ic.kind == "full":
                s = self._gen_literal(0, ScalarType("i32"))
                start_ssas.append(s)
                if isinstance(dim, int):
                    slice_sizes.append(dim)
                else:
                    raise MaomiError(f"codegen: symbolic dim '{dim}' in index", "<codegen>", 0, 0)

        # Trailing unindexed axes
        for i in range(len(expr.indices), len(base_type.dims)):
            dim = base_type.dims[i]
            s = self._gen_literal(0, ScalarType("i32"))
            start_ssas.append(s)
            if isinstance(dim, int):
                slice_sizes.append(dim)
            else:
                raise MaomiError(f"codegen: symbolic dim '{dim}' in index", "<codegen>", 0, 0)

        # Intermediate type: same rank as base, but with size-1 for single-indexed dims
        inter_dims = tuple(slice_sizes)
        inter_type = ArrayType(base_type.base, inter_dims)
        mlir_base = _mlir_type(base_type)
        mlir_inter = _mlir_type(inter_type)

        # Check if all starts are static IntLiterals → use stablehlo.slice
        if all_static and all(isinstance(ic.value, IntLiteral) for ic in expr.indices if ic.kind == "single"):
            # Build static slice: stablehlo.slice %x [start:start+size:1, ...]
            starts = []
            limits = []
            for i in range(len(base_type.dims)):
                if i < len(expr.indices):
                    ic = expr.indices[i]
                    if ic.kind == "single":
                        starts.append(ic.value.value)
                        limits.append(ic.value.value + 1)
                    elif ic.kind == "slice":
                        starts.append(ic.start.value)
                        limits.append(ic.end.value)
                    else:  # full
                        starts.append(0)
                        limits.append(base_type.dims[i])
                else:
                    starts.append(0)
                    limits.append(base_type.dims[i])

            strides = [1] * len(base_type.dims)
            starts_str = ", ".join(str(s) for s in starts)
            limits_str = ", ".join(str(s) for s in limits)
            strides_str = ", ".join(str(s) for s in strides)

            sliced = self._fresh()
            self._emit(
                f"{sliced} = stablehlo.slice {base_ssa} "
                f"[{starts_str}] [{limits_str}] [{strides_str}] "
                f": ({mlir_base}) -> {mlir_inter}"
            )
        else:
            # Dynamic path: stablehlo.dynamic_slice
            sizes_str = ", ".join(str(s) for s in slice_sizes)
            starts_str = ", ".join(start_ssas)
            start_types = ", ".join("tensor<i32>" for _ in start_ssas)

            sliced = self._fresh()
            self._emit(
                f"{sliced} = stablehlo.dynamic_slice {base_ssa}, {starts_str}, "
                f"sizes = [{sizes_str}] "
                f": ({mlir_base}, {start_types}) -> {mlir_inter}"
            )

        # Reshape to remove squeezed dimensions
        if squeezed_axes:
            result = self._fresh()
            self._emit(f"{result} = stablehlo.reshape {sliced} : ({mlir_inter}) -> {_mlir_type(result_type)}")
            return result
        return sliced

    def _gen_index_grad(self, expr: _IndexGrad, env: dict[str, str]) -> str:
        """Emit backward pass for indexing: zeros + dynamic_update_slice."""
        base_type = self._type_of(expr.base_expr)
        adj_ssa = self._gen_expr(expr.adj, env)
        adj_type = self._type_of(expr.adj)

        if not isinstance(base_type, ArrayType):
            raise MaomiError("codegen: _IndexGrad base must be array", "<codegen>", 0, 0)

        mlir_base = _mlir_type(base_type)

        # Create zero tensor of base shape
        zeros = self._fresh()
        self._emit(f"{zeros} = stablehlo.constant dense<0.000000e+00> : {mlir_base}")

        # Build start indices (same logic as forward)
        start_ssas: list[str] = []
        slice_sizes: list[int] = []
        squeezed_axes: list[int] = []

        for i, ic in enumerate(expr.indices):
            dim = base_type.dims[i]
            if ic.kind == "single":
                idx_ssa = self._gen_expr(ic.value, env)
                start_ssas.append(idx_ssa)
                slice_sizes.append(1)
                squeezed_axes.append(i)
            elif ic.kind == "slice":
                s = self._gen_literal(ic.start.value, ScalarType("i32"))
                start_ssas.append(s)
                slice_sizes.append(ic.end.value - ic.start.value)
            elif ic.kind == "full":
                s = self._gen_literal(0, ScalarType("i32"))
                start_ssas.append(s)
                slice_sizes.append(dim)

        for i in range(len(expr.indices), len(base_type.dims)):
            s = self._gen_literal(0, ScalarType("i32"))
            start_ssas.append(s)
            slice_sizes.append(base_type.dims[i])

        # Reshape adj to re-insert squeezed dims
        update_dims = tuple(slice_sizes)
        update_type = ArrayType(base_type.base, update_dims)
        mlir_update = _mlir_type(update_type)

        if squeezed_axes:
            reshaped = self._fresh()
            self._emit(f"{reshaped} = stablehlo.reshape {adj_ssa} : ({_mlir_type(adj_type)}) -> {mlir_update}")
            adj_ssa = reshaped

        # Emit dynamic_update_slice
        starts_str = ", ".join(start_ssas)
        start_types = ", ".join("tensor<i32>" for _ in start_ssas)

        result = self._fresh()
        self._emit(
            f"{result} = stablehlo.dynamic_update_slice {zeros}, {adj_ssa}, {starts_str} "
            f": ({mlir_base}, {mlir_update}, {start_types}) -> {mlir_base}"
        )
        return result

    # -- Scan helpers --

    def _slice_element(self, seq_ssa: str, idx_ssa: str, seq_type: ArrayType, elem_type: MaomiType) -> str:
        """Slice a single element from a 1D+ array at a dynamic index, returning scalar or lower-rank."""
        mlir_seq = _mlir_type(seq_type)
        slice_sizes = [1] + [d for d in seq_type.dims[1:]]
        sizes_str = ", ".join(str(s) for s in slice_sizes)
        sliced_dims = tuple(slice_sizes)
        sliced_type = ArrayType(seq_type.base, sliced_dims)
        mlir_sliced = _mlir_type(sliced_type)

        sliced = self._fresh()
        self._emit(
            f"{sliced} = stablehlo.dynamic_slice {seq_ssa}, {idx_ssa}, "
            f"sizes = [{sizes_str}] : ({mlir_seq}, tensor<i32>) -> {mlir_sliced}"
        )
        # Reshape to remove the leading 1 dim
        result = self._fresh()
        mlir_elem = _mlir_type(elem_type)
        self._emit(f"{result} = stablehlo.reshape {sliced} : ({mlir_sliced}) -> {mlir_elem}")
        return result

    def _update_element(self, arr_ssa: str, elem_ssa: str, idx_ssa: str,
                         arr_type: ArrayType, elem_type: MaomiType) -> str:
        """Update a single element in an array at a dynamic index."""
        mlir_arr = _mlir_type(arr_type)
        # Reshape elem to have leading dim 1
        if isinstance(elem_type, ScalarType):
            update_dims = (1,)
        else:
            update_dims = (1,) + elem_type.dims
        update_type = ArrayType(arr_type.base, update_dims)
        mlir_update = _mlir_type(update_type)

        reshaped = self._fresh()
        self._emit(f"{reshaped} = stablehlo.reshape {elem_ssa} : ({_mlir_type(elem_type)}) -> {mlir_update}")

        result = self._fresh()
        self._emit(
            f"{result} = stablehlo.dynamic_update_slice {arr_ssa}, {reshaped}, {idx_ssa} "
            f": ({mlir_arr}, {mlir_update}, tensor<i32>) -> {mlir_arr}"
        )
        return result

    # -- Scan codegen --

    def _gen_scan(self, expr: ScanExpr, env: dict[str, str]) -> str:
        init_val = self._gen_expr(expr.init, env)
        seq_vals = [self._gen_expr(s, env) for s in expr.sequences]

        init_type = self._type_of(expr.init)
        seq_types = [self._type_of(s) for s in expr.sequences]
        result_type = self._type_of(expr)

        for st in seq_types:
            if not isinstance(st, ArrayType):
                raise MaomiError("codegen: scan sequence must be array", "<codegen>", 0, 0)

        seq_len = seq_types[0].dims[0]
        if isinstance(seq_len, str):
            raise MaomiError(f"codegen: scan requires concrete sequence length, got '{seq_len}'", "<codegen>", 0, 0)

        # Element types (first dim stripped)
        elem_types: list[MaomiType] = []
        for st in seq_types:
            if len(st.dims) == 1:
                elem_types.append(ScalarType(st.base))
            else:
                elem_types.append(ArrayType(st.base, st.dims[1:]))

        mlir_init = _mlir_type(init_type)
        mlir_result = _mlir_type(result_type)
        mlir_seqs = [_mlir_type(st) for st in seq_types]
        mlir_i32 = "tensor<i32>"

        # Create initial values outside the while
        counter_var = self._fresh()
        if expr.reverse:
            self._emit(f"{counter_var} = stablehlo.constant dense<{seq_len - 1}> : {mlir_i32}")
        else:
            self._emit(f"{counter_var} = stablehlo.constant dense<0> : {mlir_i32}")

        output_var = self._fresh()
        self._emit(f"{output_var} = stablehlo.constant dense<0.000000e+00> : {mlir_result}")

        # Build while arg names and types
        n_seqs = len(seq_vals)
        uid = self._counter
        ctr_name = f"iterC{uid}"
        carry_name = f"iterK{uid}"
        out_name = f"iterO{uid}"
        seq_names = [f"iterS{uid}_{i}" for i in range(n_seqs)]

        arg_names = [ctr_name, carry_name, out_name] + seq_names
        init_vals_list = [counter_var, init_val, output_var] + seq_vals
        arg_types = [mlir_i32, mlir_init, mlir_result] + mlir_seqs
        n_args = len(arg_names)

        # Emit custom while format: stablehlo.while(%name = %init, ...) : types
        while_result = self._fresh()
        header_parts = ", ".join(
            f"%{arg_names[i]} = {init_vals_list[i]}" for i in range(n_args)
        )
        types_str = ", ".join(arg_types)
        self._emit(f"{while_result}:{n_args} = stablehlo.while({header_parts}) : {types_str}")

        # Cond region
        self._emit("cond {")
        self._indent += 1
        limit_v = self._fresh()
        if expr.reverse:
            self._emit(f"{limit_v} = stablehlo.constant dense<0> : {mlir_i32}")
            c_cmp = self._fresh()
            self._emit(f"{c_cmp} = stablehlo.compare GE, %{ctr_name}, {limit_v}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        else:
            self._emit(f"{limit_v} = stablehlo.constant dense<{seq_len}> : {mlir_i32}")
            c_cmp = self._fresh()
            self._emit(f"{c_cmp} = stablehlo.compare LT, %{ctr_name}, {limit_v}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        self._emit(f"stablehlo.return {c_cmp} : tensor<i1>")
        self._indent -= 1

        # Body region
        self._emit("} do {")
        self._indent += 1

        # Slice elements from each sequence
        body_env = dict(env)
        body_env[expr.carry_var] = f"%{carry_name}"
        for i, (ev, st, et) in enumerate(zip(expr.elem_vars, seq_types, elem_types)):
            b_elem = self._slice_element(f"%{seq_names[i]}", f"%{ctr_name}", st, et)
            body_env[ev] = b_elem

        new_carry = self._gen_block(expr.body, body_env)

        # Update output
        new_output = self._update_element(f"%{out_name}", new_carry, f"%{ctr_name}", result_type, init_type)

        # Update counter
        one_v = self._fresh()
        self._emit(f"{one_v} = stablehlo.constant dense<1> : {mlir_i32}")
        new_counter = self._fresh()
        if expr.reverse:
            self._emit(f"{new_counter} = stablehlo.subtract %{ctr_name}, {one_v} : {mlir_i32}")
        else:
            self._emit(f"{new_counter} = stablehlo.add %{ctr_name}, {one_v} : {mlir_i32}")

        # Return new values
        new_vals = [new_counter, new_carry, new_output] + [f"%{sn}" for sn in seq_names]
        vals_str = ", ".join(new_vals)
        self._emit(f"stablehlo.return {vals_str} : {types_str}")
        self._indent -= 1
        self._emit("}")

        # Result is the output buffer (index 2 in while results)
        return f"{while_result}#2"

    # -- Scan gradient codegen --

    def _gen_scan_grad(self, expr: _ScanGrad, env: dict[str, str]) -> str:
        """Emit a reverse while loop for the backward pass of scan."""
        fwd_result = self._gen_expr(expr.forward_result, env)
        init_val = self._gen_expr(expr.init, env)
        seq_vals = [self._gen_expr(s, env) for s in expr.sequences]
        adj_val = self._gen_expr(expr.adj, env)

        init_type = self._type_of(expr.init)
        fwd_type = self._type_of(expr.forward_result)
        adj_type = self._type_of(expr.adj)
        seq_types = [self._type_of(s) for s in expr.sequences]

        if not isinstance(fwd_type, ArrayType):
            raise MaomiError("codegen: _ScanGrad forward_result must be array", "<codegen>", 0, 0)

        # Broadcast scalar adj to array if needed
        if isinstance(adj_type, ScalarType) and isinstance(fwd_type, ArrayType):
            adj_val = self._maybe_broadcast(adj_val, adj_type, fwd_type)
            adj_type = fwd_type

        seq_len = fwd_type.dims[0]
        if isinstance(seq_len, str):
            raise MaomiError(f"codegen: _ScanGrad requires concrete length, got '{seq_len}'", "<codegen>", 0, 0)

        elem_types: list[MaomiType] = []
        for st in seq_types:
            if not isinstance(st, ArrayType):
                raise MaomiError("codegen: _ScanGrad sequence must be array", "<codegen>", 0, 0)
            if len(st.dims) == 1:
                elem_types.append(ScalarType(st.base))
            else:
                elem_types.append(ArrayType(st.base, st.dims[1:]))

        n_seqs = len(expr.sequences)
        mlir_i32 = "tensor<i32>"
        mlir_init = _mlir_type(init_type)
        mlir_fwd = _mlir_type(fwd_type)
        mlir_adj = _mlir_type(adj_type)
        mlir_seqs = [_mlir_type(st) for st in seq_types]

        # Initialize outside the while
        counter_var = self._fresh()
        self._emit(f"{counter_var} = stablehlo.constant dense<{seq_len - 1}> : {mlir_i32}")

        adj_carry_init = self._fresh()
        self._emit(f"{adj_carry_init} = stablehlo.constant dense<0.000000e+00> : {mlir_init}")

        adj_seq_inits = []
        for ms in mlir_seqs:
            v = self._fresh()
            self._emit(f"{v} = stablehlo.constant dense<0.000000e+00> : {ms}")
            adj_seq_inits.append(v)

        # Build while arg names and types
        # State: counter, adj_carry, adj_seq_0..N-1, fwd_carries, seq_0..N-1, adj_array, init_val
        uid = self._counter
        gc_name = f"gC{uid}"
        gac_name = f"gAC{uid}"
        gas_names = [f"gAS{uid}_{i}" for i in range(n_seqs)]
        gfwd_name = f"gFwd{uid}"
        gos_names = [f"gOS{uid}_{i}" for i in range(n_seqs)]
        gadj_name = f"gAdj{uid}"
        ginit_name = f"gInit{uid}"

        arg_names = [gc_name, gac_name]
        init_vals = [counter_var, adj_carry_init]
        arg_types_list = [mlir_i32, mlir_init]

        for i in range(n_seqs):
            arg_names.append(gas_names[i])
            init_vals.append(adj_seq_inits[i])
            arg_types_list.append(mlir_seqs[i])

        arg_names.append(gfwd_name)
        init_vals.append(fwd_result)
        arg_types_list.append(mlir_fwd)

        for i in range(n_seqs):
            arg_names.append(gos_names[i])
            init_vals.append(seq_vals[i])
            arg_types_list.append(mlir_seqs[i])

        arg_names.append(gadj_name)
        init_vals.append(adj_val)
        arg_types_list.append(mlir_adj)

        arg_names.append(ginit_name)
        init_vals.append(init_val)
        arg_types_list.append(mlir_init)

        n_args = len(arg_names)
        types_str = ", ".join(arg_types_list)

        # Emit custom while format
        while_result = self._fresh()
        header_parts = ", ".join(
            f"%{arg_names[i]} = {init_vals[i]}" for i in range(n_args)
        )
        self._emit(f"{while_result}:{n_args} = stablehlo.while({header_parts}) : {types_str}")

        # Cond region: counter >= 0
        self._emit("cond {")
        self._indent += 1
        zero_c = self._fresh()
        self._emit(f"{zero_c} = stablehlo.constant dense<0> : {mlir_i32}")
        c_cmp = self._fresh()
        self._emit(f"{c_cmp} = stablehlo.compare GE, %{gc_name}, {zero_c}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        self._emit(f"stablehlo.return {c_cmp} : tensor<i1>")
        self._indent -= 1

        # Body region
        self._emit("} do {")
        self._indent += 1

        # Slice adj_t from adj_array
        adj_t = self._slice_element(f"%{gadj_name}", f"%{gc_name}", adj_type, init_type)

        # adj_total = adj_carry + adj_t
        adj_total = self._fresh()
        self._emit(f"{adj_total} = stablehlo.add %{gac_name}, {adj_t} : {mlir_init}")

        # Compute prev_carry = select(counter > 0, fwd_carries[counter-1], init)
        one_b = self._fresh()
        self._emit(f"{one_b} = stablehlo.constant dense<1> : {mlir_i32}")
        zero_b = self._fresh()
        self._emit(f"{zero_b} = stablehlo.constant dense<0> : {mlir_i32}")

        prev_idx = self._fresh()
        self._emit(f"{prev_idx} = stablehlo.subtract %{gc_name}, {one_b} : {mlir_i32}")
        clamped_idx = self._fresh()
        self._emit(f"{clamped_idx} = stablehlo.clamp {zero_b}, {prev_idx}, %{gc_name} : ({mlir_i32}, {mlir_i32}, {mlir_i32}) -> {mlir_i32}")

        fwd_prev = self._slice_element(f"%{gfwd_name}", clamped_idx, fwd_type, init_type)

        gt_zero = self._fresh()
        self._emit(f"{gt_zero} = stablehlo.compare GT, %{gc_name}, {zero_b}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")

        prev_carry = self._fresh()
        self._emit(f"{prev_carry} = stablehlo.select {gt_zero}, {fwd_prev}, %{ginit_name} : (tensor<i1>, {mlir_init}, {mlir_init}) -> {mlir_init}")

        # Slice elements from original sequences
        b_elems = []
        for i in range(n_seqs):
            elem = self._slice_element(f"%{gos_names[i]}", f"%{gc_name}", seq_types[i], elem_types[i])
            b_elems.append(elem)

        # Evaluate derivative expressions with substituted values
        deriv_env = dict(env)
        deriv_env[expr.carry_var] = prev_carry
        for ev, elem_ssa in zip(expr.elem_vars, b_elems):
            deriv_env[ev] = elem_ssa

        d_carry_val = self._gen_expr(expr.d_body_d_carry, deriv_env)

        # new_adj_carry = adj_total * d_carry_val
        new_adj_carry = self._fresh()
        self._emit(f"{new_adj_carry} = stablehlo.multiply {adj_total}, {d_carry_val} : {mlir_init}")

        # For each sequence, compute adj_elem and accumulate
        new_adj_seqs = []
        for i in range(n_seqs):
            d_elem_val = self._gen_expr(expr.d_body_d_elems[i], deriv_env)
            adj_elem = self._fresh()
            mlir_et = _mlir_type(elem_types[i])
            self._emit(f"{adj_elem} = stablehlo.multiply {adj_total}, {d_elem_val} : {mlir_et}")
            new_adj_seq = self._update_element(f"%{gas_names[i]}", adj_elem, f"%{gc_name}", seq_types[i], elem_types[i])
            new_adj_seqs.append(new_adj_seq)

        # Decrement counter
        new_counter = self._fresh()
        self._emit(f"{new_counter} = stablehlo.subtract %{gc_name}, {one_b} : {mlir_i32}")

        # Return new values
        new_vals = [new_counter, new_adj_carry] + new_adj_seqs
        new_vals += [f"%{gfwd_name}"] + [f"%{gos_names[i]}" for i in range(n_seqs)]
        new_vals += [f"%{gadj_name}", f"%{ginit_name}"]
        vals_str = ", ".join(new_vals)
        self._emit(f"stablehlo.return {vals_str} : {types_str}")
        self._indent -= 1
        self._emit("}")

        # Extract result based on wrt
        if expr.wrt == "__init__":
            return f"{while_result}#1"  # adj_carry
        else:
            for i, seq_expr in enumerate(expr.sequences):
                if isinstance(seq_expr, Identifier) and seq_expr.name == expr.wrt:
                    return f"{while_result}#{2 + i}"  # adj_seq_i
            return f"{while_result}#2"  # fallback: first adj_seq

    # -- Map codegen --

    def _gen_map(self, expr: MapExpr, env: dict[str, str]) -> str:
        seq_val = self._gen_expr(expr.sequence, env)
        seq_type = self._type_of(expr.sequence)

        if not isinstance(seq_type, ArrayType):
            raise MaomiError("codegen: map sequence must be array", "<codegen>", 0, 0)

        batch_dim = seq_type.dims[0]

        # Lift all body expression types to include the batch dimension
        self._lift_body_types(expr.body, batch_dim)

        body_env = dict(env)
        body_env[expr.elem_var] = seq_val
        return self._gen_block(expr.body, body_env)

    def _lift_body_types(self, block, batch_dim):
        """Prepend batch_dim to all types in a block for map codegen."""
        from .ast_nodes import Block, LetStmt, ExprStmt
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                self._lift_expr_type(stmt.value, batch_dim)
            elif isinstance(stmt, ExprStmt):
                self._lift_expr_type(stmt.expr, batch_dim)
        if block.expr is not None:
            self._lift_expr_type(block.expr, batch_dim)

    def _lift_expr_type(self, expr, batch_dim):
        """Recursively lift an expression's type to include batch_dim.
        Literals stay scalar — they'll be broadcast by _maybe_broadcast."""
        # Don't lift literals — they remain scalar and broadcast naturally
        if isinstance(expr, (IntLiteral, FloatLiteral, BoolLiteral)):
            return

        t = self.type_map.get(id(expr))
        if t is not None:
            if isinstance(t, ScalarType):
                self.type_map[id(expr)] = ArrayType(t.base, (batch_dim,))
            elif isinstance(t, ArrayType):
                self.type_map[id(expr)] = ArrayType(t.base, (batch_dim,) + t.dims)

        # Recurse into sub-expressions
        match expr:
            case BinOp(left=left, right=right):
                self._lift_expr_type(left, batch_dim)
                self._lift_expr_type(right, batch_dim)
            case UnaryOp(operand=operand):
                self._lift_expr_type(operand, batch_dim)
            case CallExpr(args=args):
                for a in args:
                    self._lift_expr_type(a, batch_dim)
            case IfExpr():
                self._lift_expr_type(expr.condition, batch_dim)
                self._lift_body_types(expr.then_block, batch_dim)
                self._lift_body_types(expr.else_block, batch_dim)
            case IndexExpr(base=base, indices=indices):
                self._lift_expr_type(base, batch_dim)
                for ic in indices:
                    if ic.value is not None:
                        self._lift_expr_type(ic.value, batch_dim)
                    if ic.start is not None:
                        self._lift_expr_type(ic.start, batch_dim)
                    if ic.end is not None:
                        self._lift_expr_type(ic.end, batch_dim)
            case _:
                pass

    # -- Broadcasting helpers --

    def _maybe_broadcast(self, ssa: str, from_type: MaomiType, to_type: MaomiType) -> str:
        """Insert broadcast_in_dim if from_type needs to be broadcast to to_type."""
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
            if len(from_type.dims) < len(to_type.dims):
                offset = len(to_type.dims) - len(from_type.dims)
                dims = list(range(offset, len(to_type.dims)))
                dims_str = ", ".join(str(d) for d in dims)
                var = self._fresh()
                self._emit(
                    f"{var} = stablehlo.broadcast_in_dim {ssa}, "
                    f"dims = [{dims_str}] : ({_mlir_type(from_type)}) -> {_mlir_type(to_type)}"
                )
                return var

        return ssa


def _base_of_type(t: MaomiType) -> str:
    if isinstance(t, ScalarType):
        return t.base
    return t.base


def _types_equal(a: MaomiType, b: MaomiType) -> bool:
    if isinstance(a, ScalarType) and isinstance(b, ScalarType):
        return a.base == b.base
    if isinstance(a, ArrayType) and isinstance(b, ArrayType):
        return a.base == b.base and a.dims == b.dims
    return False
