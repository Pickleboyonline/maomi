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
    WhileExpr,
    MapExpr,
    CastExpr,
    FoldExpr,
    StructLiteral,
    FieldAccess,
    WithExpr,
    IndexExpr,
    IndexComponent,
    _ScanGrad,
    _WhileGrad,
    _IndexGrad,
    _GatherGrad,
    _Conv2dGrad,
    _MaxPoolGrad,
    _AvgPoolGrad,
    _FoldGrad,
    _BroadcastExpr,
    _ReduceSum,
    Expr,
)
from .types import MaomiType, ScalarType, ArrayType, StructType
from .errors import MaomiError


def _block_references_var(block: Block, var_name: str) -> bool:
    """Check if a block's expression references a variable name."""
    if block.expr is None:
        return False
    return _expr_references_var(block.expr, var_name)


def _expr_references_var(expr: Expr, var_name: str) -> bool:
    """Recursively check if an expression references a variable by name."""
    match expr:
        case Identifier(name=name):
            return name == var_name
        case UnaryOp(operand=operand):
            return _expr_references_var(operand, var_name)
        case BinOp(left=left, right=right):
            return _expr_references_var(left, var_name) or _expr_references_var(right, var_name)
        case CallExpr(args=args):
            return any(_expr_references_var(a, var_name) for a in args)
        case IfExpr(condition=c, then_block=tb, else_block=eb):
            return (_expr_references_var(c, var_name)
                    or _block_references_var(tb, var_name)
                    or _block_references_var(eb, var_name))
        case _:
            return False


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
                    f"codegen: unresolved symbolic dimension '{d}' in type '{t}'. "
                    f"XLA requires concrete shapes to compile — symbolic dimensions are checked "
                    f"during type checking but must be resolved to integers before code generation. "
                    f"Ensure all function parameters have concrete dimensions at the entry point.",
                    "<codegen>", 0, 0,
                )
        shape = "x".join(str(d) for d in t.dims)
        return f"tensor<{shape}x{_MLIR_ETYPE[t.base]}>"
    if isinstance(t, StructType):
        field_types = ", ".join(_mlir_type(ft) for _, ft in t.fields)
        return f"tuple<{field_types}>"
    raise MaomiError("codegen: unknown type", "<codegen>", 0, 0)


def _callback_layout(t: MaomiType) -> str:
    """Return MLIR operand layout attribute for a type (row-major)."""
    if isinstance(t, ScalarType):
        return "dense<> : tensor<0xindex>"
    if isinstance(t, ArrayType):
        ndim = len(t.dims)
        if ndim == 1:
            return "dense<0> : tensor<1xindex>"
        order = ", ".join(str(i) for i in reversed(range(ndim)))
        return f"dense<[{order}]> : tensor<{ndim}xindex>"
    return "dense<> : tensor<0xindex>"


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


def _collect_refs_expr(expr: Expr, refs: set[str]):
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
        case IndexExpr(base=base, indices=indices):
            _collect_refs_expr(base, refs)
            for ic in indices:
                if ic.value is not None:
                    _collect_refs_expr(ic.value, refs)
                if ic.start is not None:
                    _collect_refs_expr(ic.start, refs)
                if ic.end is not None:
                    _collect_refs_expr(ic.end, refs)
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
        case _ReduceSum(expr=e):
            _collect_refs_expr(e, refs)


class StableHLOCodegen:
    def __init__(self, program: Program, type_map: dict[int, MaomiType] | None = None):
        self.program = program
        self.type_map = type_map or {}
        self._counter = 0
        self._lines: list[str] = []
        self._indent = 0
        self._batch_depth = 0
        self._batch_dims: list[int] = []
        self._batched_fns: dict[tuple[str, tuple[int, ...]], str] = {}
        self._callback_count: int = 0

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
        # Key type alias
        if ta.base == "Key":
            return ArrayType("i32", (4,))
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
            case WhileExpr():
                return self._gen_while(expr, env)
            case MapExpr():
                return self._gen_map(expr, env)
            case CastExpr():
                return self._gen_cast(expr, env)
            case FoldExpr():
                return self._gen_fold(expr, env)
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
            case _WhileGrad():
                return self._gen_while_grad(expr, env)
            case _IndexGrad():
                return self._gen_index_grad(expr, env)
            case _GatherGrad():
                return self._gen_gather_grad(expr, env)
            case _Conv2dGrad():
                return self._gen_conv2d_grad(expr, env)
            case _MaxPoolGrad():
                return self._gen_max_pool_grad(expr, env)
            case _AvgPoolGrad():
                return self._gen_avg_pool_grad(expr, env)
            case _FoldGrad():
                return self._gen_fold_grad(expr, env)
            case _BroadcastExpr():
                return self._gen_broadcast_expr(expr, env)
            case _ReduceSum():
                return self._gen_reduce_sum_axes(expr, env)
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

        bd = self._batch_depth
        if bd > 0:
            batch_dims_tuple = tuple(self._batch_dims)
            l_batched = len(lt.dims) > bd and lt.dims[:bd] == batch_dims_tuple
            r_batched = len(rt.dims) > bd and rt.dims[:bd] == batch_dims_tuple

            if not l_batched:
                lhs = self._broadcast_to_batched(lhs, lt)
                lt = ArrayType(lt.base, batch_dims_tuple + lt.dims)
            if not r_batched:
                rhs = self._broadcast_to_batched(rhs, rt)
                rt = ArrayType(rt.base, batch_dims_tuple + rt.dims)

            batch_str = ", ".join(str(i) for i in range(bd))
            l_contract = len(lt.dims) - 1
            r_contract = bd

            var = self._fresh()
            self._emit(
                f"{var} = stablehlo.dot_general {lhs}, {rhs}, "
                f"batching_dims = [{batch_str}] x [{batch_str}], "
                f"contracting_dims = [{l_contract}] x [{r_contract}], "
                f"precision = [DEFAULT, DEFAULT] "
                f": ({_mlir_type(lt)}, {_mlir_type(rt)}) -> {_mlir_type(result_type)}"
            )
            return var

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

    def _gen_where(self, expr: CallExpr, env: dict[str, str]) -> str:
        cond = self._gen_expr(expr.args[0], env)
        x = self._gen_expr(expr.args[1], env)
        y = self._gen_expr(expr.args[2], env)

        cond_type = self._type_of(expr.args[0])
        x_type = self._type_of(expr.args[1])
        y_type = self._type_of(expr.args[2])
        result_type = self._type_of(expr)
        mlir_result = _mlir_type(result_type)

        # Broadcast x and y to result type
        x = self._maybe_broadcast(x, x_type, result_type)
        y = self._maybe_broadcast(y, y_type, result_type)

        # Broadcast cond to result shape (as bool array)
        if isinstance(cond_type, ScalarType) and isinstance(result_type, ArrayType):
            bool_array_type = ArrayType("bool", result_type.dims)
            broadcast_cond = self._fresh()
            self._emit(
                f"{broadcast_cond} = stablehlo.broadcast_in_dim {cond}, "
                f"dims = [] : ({_mlir_type(cond_type)}) -> {_mlir_type(bool_array_type)}"
            )
            cond = broadcast_cond
            cond_type = bool_array_type
        elif isinstance(cond_type, ArrayType) and isinstance(result_type, ArrayType) and cond_type.dims != result_type.dims:
            bool_array_type = ArrayType("bool", result_type.dims)
            cond = self._maybe_broadcast(cond, cond_type, bool_array_type)
            cond_type = bool_array_type

        cond_mlir = _mlir_type(cond_type)
        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.select {cond}, {x}, {y} "
            f": ({cond_mlir}, {mlir_result}, {mlir_result}) -> {mlir_result}"
        )
        return var

    _CALLBACK_BUILTINS = {"callback"}
    _RNG_BUILTINS = {"rng_key", "rng_split", "rng_uniform", "rng_normal"}

    def _gen_callback(self, expr: CallExpr, env: dict[str, str]) -> str:
        """Emit stablehlo.custom_call targeting JAX's FFI callback handler."""
        arg_ssas = [self._gen_expr(a, env) for a in expr.args]
        arg_types = [self._type_of(a) for a in expr.args]

        idx = self._callback_count
        self._callback_count += 1

        if arg_ssas:
            operands = ", ".join(arg_ssas)
            mlir_types = ", ".join(_mlir_type(t) for t in arg_types)
            layouts = ", ".join(_callback_layout(t) for t in arg_types)
            self._emit(
                f'"stablehlo.custom_call"({operands}) '
                f'{{call_target_name = "xla_ffi_python_cpu_callback", '
                f'has_side_effect = true, backend_config = "", '
                f'api_version = 1 : i32, '
                f'mhlo.backend_config = {{index = {idx} : ui64}}, '
                f'operand_layouts = [{layouts}], '
                f'result_layouts = []}} : ({mlir_types}) -> ()'
            )
        else:
            self._emit(
                f'"stablehlo.custom_call"() '
                f'{{call_target_name = "xla_ffi_python_cpu_callback", '
                f'has_side_effect = true, backend_config = "", '
                f'api_version = 1 : i32, '
                f'mhlo.backend_config = {{index = {idx} : ui64}}, '
                f'operand_layouts = [], '
                f'result_layouts = []}} : () -> ()'
            )
        return ""

    def _gen_call(self, expr: CallExpr, env: dict[str, str]) -> str:
        # Callback: emit stablehlo.custom_call to fire Python callback via FFI
        if expr.callee in self._CALLBACK_BUILTINS:
            return self._gen_callback(expr, env)

        # iota(N) → stablehlo.iota
        if expr.callee == "iota":
            return self._gen_iota(expr, env)

        # RNG builtins
        if expr.callee in self._RNG_BUILTINS:
            return self._gen_rng(expr, env)

        # Handle builtins
        if expr.callee in _BUILTIN_OPS:
            return self._gen_elementwise_builtin(expr, env)
        if expr.callee == "mean":
            return self._gen_mean(expr, env)
        if expr.callee == "sum":
            return self._gen_sum(expr, env)
        if expr.callee in ("max", "min"):
            return self._gen_max_min(expr, env)
        if expr.callee in ("argmax", "argmin"):
            return self._gen_argmax(expr, env)
        if expr.callee == "transpose":
            return self._gen_transpose(expr, env)
        if expr.callee == "reshape":
            return self._gen_reshape(expr, env)
        if expr.callee == "concat":
            return self._gen_concat(expr, env)
        if expr.callee == "stop_gradient":
            return self._gen_expr(expr.args[0], env)
        if expr.callee == "where":
            return self._gen_where(expr, env)
        if expr.callee == "conv2d":
            return self._gen_conv2d(expr, env)
        if expr.callee == "max_pool":
            return self._gen_max_pool(expr, env)
        if expr.callee == "avg_pool":
            return self._gen_avg_pool(expr, env)

        # User-defined function call
        if self._batch_depth > 0:
            return self._gen_batched_call(expr, env)

        args = [self._gen_expr(a, env) for a in expr.args]
        arg_types = [self._type_of(a) for a in expr.args]
        result_type = self._type_of(expr)

        args_str = ", ".join(args)
        types_str = ", ".join(_mlir_type(t) for t in arg_types)
        var = self._fresh()
        self._emit(f"{var} = func.call @{expr.callee}({args_str}) : ({types_str}) -> {_mlir_type(result_type)}")
        return var

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
        import copy
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

    def _gen_elementwise_builtin(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        result_type = self._type_of(expr)
        mlir_t = _mlir_type(result_type)
        op = _BUILTIN_OPS[expr.callee]
        var = self._fresh()
        self._emit(f"{var} = {op} {arg} : {mlir_t}")
        return var

    def _gen_iota(self, expr: CallExpr, env: dict[str, str]) -> str:
        result_type = self._type_of(expr)
        mlir_t = _mlir_type(result_type)
        var = self._fresh()
        self._emit(f"{var} = stablehlo.iota dim = 0 : {mlir_t}")
        return var

    def _gen_mean(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)

        if not isinstance(arg_type, ArrayType):
            return arg  # mean of scalar is itself

        # Axis-specific or all-dims reduction
        if len(expr.args) == 2:
            axis = expr.args[1].value
            bd = self._batch_depth
            actual_axis = bd + axis
            # Sum along specific axis
            sum_var = self._gen_reduce_sum_single_axis(arg, arg_type, result_type, actual_axis)
            # Divide by axis size
            axis_size = arg_type.dims[actual_axis]
            if isinstance(axis_size, str):
                raise MaomiError(f"codegen: cannot compute mean with symbolic dim '{axis_size}'", "<codegen>", 0, 0)
            count_scalar = self._fresh()
            scalar_mlir = _mlir_type(ScalarType(arg_type.base))
            self._emit(f"{count_scalar} = stablehlo.constant dense<{float(axis_size):e}> : {scalar_mlir}")
            count_var = self._maybe_broadcast(count_scalar, ScalarType(arg_type.base), result_type)
            mlir_result = _mlir_type(result_type)
            var = self._fresh()
            self._emit(f"{var} = stablehlo.divide {sum_var}, {count_var} : {mlir_result}")
            return var

        # All-dims reduction
        sum_var = self._gen_reduce_sum(arg, arg_type, result_type)

        # Compute element count (skip batch dims)
        bd = self._batch_depth
        numel = 1
        for d in arg_type.dims[bd:]:
            if isinstance(d, int):
                numel *= d
            else:
                raise MaomiError(f"codegen: cannot compute mean with symbolic dim '{d}'", "<codegen>", 0, 0)

        # Divide by count — constant must match result type
        mlir_result = _mlir_type(result_type)
        count_scalar = self._fresh()
        scalar_mlir = _mlir_type(ScalarType(arg_type.base))
        self._emit(f"{count_scalar} = stablehlo.constant dense<{float(numel):e}> : {scalar_mlir}")
        count_var = self._maybe_broadcast(count_scalar, ScalarType(arg_type.base), result_type)

        var = self._fresh()
        self._emit(f"{var} = stablehlo.divide {sum_var}, {count_var} : {mlir_result}")
        return var

    def _gen_sum(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)

        if not isinstance(arg_type, ArrayType):
            return arg

        if len(expr.args) == 2:
            axis = expr.args[1].value
            bd = self._batch_depth
            actual_axis = bd + axis
            return self._gen_reduce_sum_single_axis(arg, arg_type, result_type, actual_axis)

        return self._gen_reduce_sum(arg, arg_type, result_type)

    def _gen_reduce_sum(self, arg: str, arg_type: ArrayType, result_type: MaomiType) -> str:
        """Generate a sum reduction. When batched, skip batch dims."""
        bd = self._batch_depth
        ndims = len(arg_type.dims)
        reduce_dims = list(range(bd, ndims))
        dims_str = ", ".join(str(i) for i in reduce_dims)

        # Initial value (0)
        init_var = self._fresh()
        scalar_type = ScalarType(arg_type.base)
        mlir_scalar = _mlir_type(scalar_type)
        self._emit(f"{init_var} = stablehlo.constant dense<0.000000e+00> : {mlir_scalar}")

        mlir_result = _mlir_type(result_type)
        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.reduce({arg} init: {init_var}) "
            f"across dimensions = [{dims_str}] "
            f": ({_mlir_type(arg_type)}, {mlir_scalar}) -> {mlir_result}"
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

    def _gen_reduce_sum_single_axis(self, arg: str, arg_type: ArrayType, result_type: MaomiType, axis: int) -> str:
        """Generate a sum reduction along a single axis."""
        init_var = self._fresh()
        scalar_type = ScalarType(arg_type.base)
        mlir_scalar = _mlir_type(scalar_type)
        self._emit(f"{init_var} = stablehlo.constant dense<0.000000e+00> : {mlir_scalar}")

        mlir_result = _mlir_type(result_type)
        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.reduce({arg} init: {init_var}) "
            f"across dimensions = [{axis}] "
            f": ({_mlir_type(arg_type)}, {mlir_scalar}) -> {mlir_result}"
        )
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

    def _gen_max_min(self, expr: CallExpr, env: dict[str, str]) -> str:
        """Generate reduce-max or reduce-min."""
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)

        if not isinstance(arg_type, ArrayType):
            return arg

        is_max = expr.callee == "max"

        if len(expr.args) == 2:
            axis = expr.args[1].value
            bd = self._batch_depth
            actual_axis = bd + axis
            return self._gen_reduce_max_min_single_axis(arg, arg_type, result_type, actual_axis, is_max)

        return self._gen_reduce_max_min(arg, arg_type, result_type, is_max)

    def _gen_reduce_max_min(self, arg: str, arg_type: ArrayType, result_type: MaomiType, is_max: bool) -> str:
        """Generate a max/min reduction over all non-batch dims."""
        bd = self._batch_depth
        ndims = len(arg_type.dims)
        reduce_dims = list(range(bd, ndims))
        dims_str = ", ".join(str(i) for i in reduce_dims)

        init_var = self._fresh()
        scalar_type = ScalarType(arg_type.base)
        mlir_scalar = _mlir_type(scalar_type)
        init_val = self._reduce_init_value(arg_type.base, is_max)
        self._emit(f"{init_var} = stablehlo.constant dense<{init_val}> : {mlir_scalar}")

        combiner = "stablehlo.maximum" if is_max else "stablehlo.minimum"

        mlir_result = _mlir_type(result_type)
        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.reduce({arg} init: {init_var}) "
            f"across dimensions = [{dims_str}] "
            f": ({_mlir_type(arg_type)}, {mlir_scalar}) -> {mlir_result}"
        )
        self._indent += 1
        a_var = self._fresh()
        b_var = self._fresh()
        self._emit(f"reducer({a_var}: {mlir_scalar}, {b_var}: {mlir_scalar}) {{")
        self._indent += 1
        r_var = self._fresh()
        self._emit(f"{r_var} = {combiner} {a_var}, {b_var} : {mlir_scalar}")
        self._emit(f"stablehlo.return {r_var} : {mlir_scalar}")
        self._indent -= 1
        self._emit("}")
        self._indent -= 1
        return var

    def _gen_reduce_max_min_single_axis(self, arg: str, arg_type: ArrayType, result_type: MaomiType, axis: int, is_max: bool) -> str:
        """Generate a max/min reduction along a single axis."""
        init_var = self._fresh()
        scalar_type = ScalarType(arg_type.base)
        mlir_scalar = _mlir_type(scalar_type)
        init_val = self._reduce_init_value(arg_type.base, is_max)
        self._emit(f"{init_var} = stablehlo.constant dense<{init_val}> : {mlir_scalar}")

        combiner = "stablehlo.maximum" if is_max else "stablehlo.minimum"

        mlir_result = _mlir_type(result_type)
        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.reduce({arg} init: {init_var}) "
            f"across dimensions = [{axis}] "
            f": ({_mlir_type(arg_type)}, {mlir_scalar}) -> {mlir_result}"
        )
        self._indent += 1
        a_var = self._fresh()
        b_var = self._fresh()
        self._emit(f"reducer({a_var}: {mlir_scalar}, {b_var}: {mlir_scalar}) {{")
        self._indent += 1
        r_var = self._fresh()
        self._emit(f"{r_var} = {combiner} {a_var}, {b_var} : {mlir_scalar}")
        self._emit(f"stablehlo.return {r_var} : {mlir_scalar}")
        self._indent -= 1
        self._emit("}")
        self._indent -= 1
        return var

    @staticmethod
    def _reduce_init_value(base: str, is_max: bool) -> str:
        """Return the StableHLO init value literal for reduce-max/min."""
        if base in ("f32", "f64"):
            return "0xFF800000" if is_max else "0x7F800000"  # -inf / +inf
        elif base == "i32":
            return "-2147483647" if is_max else "2147483647"
        elif base == "i64":
            return "-9223372036854775807" if is_max else "9223372036854775807"
        return "0xFF800000" if is_max else "0x7F800000"

    def _gen_argmax(self, expr: CallExpr, env: dict[str, str]) -> str:
        """Generate argmax/argmin via variadic reduce with (value, index) pairs."""
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)
        is_max = expr.callee == "argmax"

        if not isinstance(arg_type, ArrayType):
            raise MaomiError("codegen: argmax/argmin requires array", "<codegen>", expr.span.line_start, expr.span.col_start)

        bd = self._batch_depth

        if len(expr.args) == 2:
            axis = expr.args[1].value + bd
        else:
            # All-dims: reshape to 1D first, then reduce axis 0
            total = 1
            for d in arg_type.dims:
                total *= d
            flat_type = ArrayType(arg_type.base, (total,))
            flat_var = self._fresh()
            self._emit(
                f"{flat_var} = stablehlo.reshape {arg} "
                f": ({_mlir_type(arg_type)}) -> {_mlir_type(flat_type)}"
            )
            arg = flat_var
            arg_type = flat_type
            axis = bd  # axis 0 (or bd if batched)

        # Generate iota for indices along the reduction axis
        iota_type = ArrayType("i32", arg_type.dims)
        iota_var = self._fresh()
        self._emit(f"{iota_var} = stablehlo.iota dim = {axis} : {_mlir_type(iota_type)}")

        # Init values
        init_val_str = self._reduce_init_value(arg_type.base, is_max)
        scalar_val_type = ScalarType(arg_type.base)
        scalar_idx_type = ScalarType("i32")
        mlir_val_scalar = _mlir_type(scalar_val_type)
        mlir_idx_scalar = _mlir_type(scalar_idx_type)

        init_val = self._fresh()
        self._emit(f"{init_val} = stablehlo.constant dense<{init_val_str}> : {mlir_val_scalar}")
        init_idx = self._fresh()
        self._emit(f"{init_idx} = stablehlo.constant dense<0> : {mlir_idx_scalar}")

        # Compute result types for the reduce
        mlir_result = _mlir_type(result_type)
        # Value result has same shape but keeps original base type
        if isinstance(result_type, ScalarType):
            val_result_type = ScalarType(arg_type.base)
        else:
            val_result_type = ArrayType(arg_type.base, result_type.dims)
        mlir_val_result = _mlir_type(val_result_type)

        # Variadic reduce
        result_var = self._fresh()
        cmp = "GT" if is_max else "LT"
        cmp_kind = "FLOAT" if arg_type.base in ("f32", "f64") else "SIGNED"

        self._emit(
            f"{result_var}:2 = stablehlo.reduce({arg} init: {init_val}, {iota_var} init: {init_idx}) "
            f"across dimensions = [{axis}] "
            f": ({_mlir_type(arg_type)}, {_mlir_type(iota_type)}, {mlir_val_scalar}, {mlir_idx_scalar}) "
            f"-> ({mlir_val_result}, {mlir_result})"
        )
        self._indent += 1
        a_val = self._fresh()
        b_val = self._fresh()
        a_idx = self._fresh()
        b_idx = self._fresh()
        self._emit(f"reducer({a_val}: {mlir_val_scalar}, {b_val}: {mlir_val_scalar}, "
                   f"{a_idx}: {mlir_idx_scalar}, {b_idx}: {mlir_idx_scalar}) {{")
        self._indent += 1
        cmp_var = self._fresh()
        self._emit(f"{cmp_var} = stablehlo.compare {cmp}, {a_val}, {b_val}, {cmp_kind} "
                   f": ({mlir_val_scalar}, {mlir_val_scalar}) -> tensor<i1>")
        sel_val = self._fresh()
        self._emit(f"{sel_val} = stablehlo.select {cmp_var}, {a_val}, {b_val} "
                   f": (tensor<i1>, {mlir_val_scalar}, {mlir_val_scalar}) -> {mlir_val_scalar}")
        sel_idx = self._fresh()
        self._emit(f"{sel_idx} = stablehlo.select {cmp_var}, {a_idx}, {b_idx} "
                   f": (tensor<i1>, {mlir_idx_scalar}, {mlir_idx_scalar}) -> {mlir_idx_scalar}")
        self._emit(f"stablehlo.return {sel_val}, {sel_idx} : {mlir_val_scalar}, {mlir_idx_scalar}")
        self._indent -= 1
        self._emit("}")
        self._indent -= 1

        # Return indices only
        return f"{result_var}#1"

    def _gen_transpose(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)

        bd = self._batch_depth
        expected_rank = 2 + bd
        if not isinstance(arg_type, ArrayType) or len(arg_type.dims) != expected_rank:
            raise MaomiError("codegen: transpose requires 2D array", "<codegen>", expr.span.line_start, expr.span.col_start)

        perm = list(range(bd)) + [bd + 1, bd]
        perm_str = ", ".join(str(p) for p in perm)

        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.transpose {arg}, dims = [{perm_str}] "
            f": ({_mlir_type(arg_type)}) -> {_mlir_type(result_type)}"
        )
        return var

    def _gen_reshape(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)
        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.reshape {arg} "
            f": ({_mlir_type(arg_type)}) -> {_mlir_type(result_type)}"
        )
        return var

    def _gen_concat(self, expr: CallExpr, env: dict[str, str]) -> str:
        # Detect axis: if last arg is IntLiteral with i32 type, it's the axis
        if (isinstance(expr.args[-1], IntLiteral)
                and isinstance(self._type_of(expr.args[-1]), ScalarType)):
            axis = expr.args[-1].value
            array_args = expr.args[:-1]
        else:
            axis = 0
            array_args = expr.args

        arg_ssas = [self._gen_expr(a, env) for a in array_args]
        arg_types = [self._type_of(a) for a in array_args]
        result_type = self._type_of(expr)

        args_str = ", ".join(arg_ssas)
        types_str = ", ".join(_mlir_type(t) for t in arg_types)
        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.concatenate {args_str}, dim = {axis} "
            f": ({types_str}) -> {_mlir_type(result_type)}"
        )
        return var

    # -- Conv/Pool codegen --

    def _extract_conv2d_params(self, expr: CallExpr) -> tuple[int, int, int, int]:
        """Extract (stride_h, stride_w, pad_h, pad_w) from conv2d args."""
        nargs = len(expr.args)
        if nargs == 2:
            return (1, 1, 0, 0)
        elif nargs == 4:
            return (expr.args[2].value, expr.args[2].value,
                    expr.args[3].value, expr.args[3].value)
        else:
            return (expr.args[2].value, expr.args[3].value,
                    expr.args[4].value, expr.args[5].value)

    def _gen_conv2d(self, expr: CallExpr, env: dict[str, str]) -> str:
        lhs = self._gen_expr(expr.args[0], env)
        rhs = self._gen_expr(expr.args[1], env)
        lhs_type = self._type_of(expr.args[0])
        rhs_type = self._type_of(expr.args[1])
        result_type = self._type_of(expr)

        sh, sw, ph, pw = self._extract_conv2d_params(expr)

        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.convolution({lhs}, {rhs}) "
            f"dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], "
            f"window = {{stride = [{sh}, {sw}], pad = [[{ph}, {ph}], [{pw}, {pw}]], "
            f"lhs_dilate = [1, 1], rhs_dilate = [1, 1]}} "
            f"{{batch_group_count = 1 : i64, feature_group_count = 1 : i64}} "
            f": ({_mlir_type(lhs_type)}, {_mlir_type(rhs_type)}) -> {_mlir_type(result_type)}"
        )
        return var

    def _gen_reduce_window(self, input_ssa: str, input_type: ArrayType,
                           result_type: MaomiType, reducer_op: str,
                           init_value: str, window_dims: list[int],
                           window_strides: list[int],
                           padding: list[tuple[int, int]]) -> str:
        """Emit stablehlo.reduce_window with the given reducer."""
        scalar_type = ScalarType(input_type.base)
        mlir_scalar = _mlir_type(scalar_type)
        ndims = len(window_dims)

        init_var = self._fresh()
        self._emit(f"{init_var} = stablehlo.constant dense<{init_value}> : {mlir_scalar}")

        dims_str = ", ".join(str(d) for d in window_dims)
        strides_str = ", ".join(str(s) for s in window_strides)
        base_dilations_str = ", ".join("1" for _ in window_dims)

        # Build padding as dense tensor attribute
        # Check if all padding is zero
        all_zero_pad = all(lo == 0 and hi == 0 for lo, hi in padding)
        if all_zero_pad:
            pad_attr = f"dense<0> : tensor<{ndims}x2xi64>"
        else:
            pad_rows = ", ".join(f"{lo}, {hi}" for lo, hi in padding)
            pad_attr = f"dense<[[{pad_rows}]]> : tensor<{ndims}x2xi64>"

        var = self._fresh()
        self._emit(
            f'{var} = "stablehlo.reduce_window"({input_ssa}, {init_var}) '
            f"<{{base_dilations = array<i64: {base_dilations_str}>, "
            f"padding = {pad_attr}, "
            f"window_dilations = array<i64: {base_dilations_str}>, "
            f"window_dimensions = array<i64: {dims_str}>, "
            f"window_strides = array<i64: {strides_str}>}}> ({{"
        )
        # Emit reducer body
        self._indent += 1
        a_var = self._fresh()
        b_var = self._fresh()
        self._emit(f"^bb0({a_var}: {mlir_scalar}, {b_var}: {mlir_scalar}):")
        self._indent += 1
        r_var = self._fresh()
        self._emit(f"{r_var} = {reducer_op} {a_var}, {b_var} : {mlir_scalar}")
        self._emit(f"stablehlo.return {r_var} : {mlir_scalar}")
        self._indent -= 1
        self._indent -= 1
        self._emit(
            f"}}) : ({_mlir_type(input_type)}, {mlir_scalar}) -> {_mlir_type(result_type)}"
        )
        return var

    def _gen_max_pool(self, expr: CallExpr, env: dict[str, str]) -> str:
        input_ssa = self._gen_expr(expr.args[0], env)
        input_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)
        assert isinstance(input_type, ArrayType)

        wh, ww = expr.args[1].value, expr.args[2].value
        sh, sw = expr.args[3].value, expr.args[4].value

        base = input_type.base
        if base in ("f32", "f64"):
            init = "0xFF800000" if base == "f32" else "0xFFF0000000000000"
        else:
            init = str(-(2**31)) if base == "i32" else str(-(2**63))

        return self._gen_reduce_window(
            input_ssa, input_type, result_type,
            "stablehlo.maximum", init,
            [1, 1, wh, ww], [1, 1, sh, sw],
            [(0, 0), (0, 0), (0, 0), (0, 0)],
        )

    def _gen_avg_pool(self, expr: CallExpr, env: dict[str, str]) -> str:
        input_ssa = self._gen_expr(expr.args[0], env)
        input_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)
        assert isinstance(input_type, ArrayType)

        wh, ww = expr.args[1].value, expr.args[2].value
        sh, sw = expr.args[3].value, expr.args[4].value

        # Sum pool
        sum_var = self._gen_reduce_window(
            input_ssa, input_type, result_type,
            "stablehlo.add", "0.000000e+00",
            [1, 1, wh, ww], [1, 1, sh, sw],
            [(0, 0), (0, 0), (0, 0), (0, 0)],
        )

        # Divide by window size
        count = float(wh * ww)
        count_var = self._fresh()
        mlir_result = _mlir_type(result_type)
        self._emit(f"{count_var} = stablehlo.constant dense<{count:e}> : {mlir_result}")

        var = self._fresh()
        self._emit(f"{var} = stablehlo.divide {sum_var}, {count_var} : {mlir_result}")
        return var

    # -- Cast codegen --

    def _gen_cast(self, expr: CastExpr, env: dict[str, str]) -> str:
        val = self._gen_expr(expr.expr, env)
        src_type = self._type_of(expr.expr)
        dst_type = self._type_of(expr)
        if _types_equal(src_type, dst_type):
            return val
        var = self._fresh()
        self._emit(f"{var} = stablehlo.convert {val} : ({_mlir_type(src_type)}) -> {_mlir_type(dst_type)}")
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

    def _normalize_index(self, idx_ssa: str, dim: int) -> str:
        """Emit runtime normalization for potentially negative index: select(i < 0, i + dim, i)."""
        mlir_i32 = "tensor<i32>"
        mlir_i1 = "tensor<i1>"
        zero = self._gen_literal(0, ScalarType("i32"))
        dim_ssa = self._gen_literal(dim, ScalarType("i32"))
        is_neg = self._fresh()
        self._emit(
            f"{is_neg} = stablehlo.compare LT, {idx_ssa}, {zero}, SIGNED "
            f": ({mlir_i32}, {mlir_i32}) -> {mlir_i1}"
        )
        wrapped = self._fresh()
        self._emit(f"{wrapped} = stablehlo.add {idx_ssa}, {dim_ssa} : {mlir_i32}")
        result = self._fresh()
        self._emit(
            f"{result} = stablehlo.select {is_neg}, {wrapped}, {idx_ssa} "
            f": ({mlir_i1}, {mlir_i32}, {mlir_i32}) -> {mlir_i32}"
        )
        return result

    # -- RNG codegen --

    def _gen_rng(self, expr: CallExpr, env: dict[str, str]) -> str:
        callee = expr.callee
        if callee == "rng_key":
            return self._gen_rng_key(expr, env)
        elif callee == "rng_split":
            return self._gen_rng_split(expr, env)
        elif callee == "rng_uniform":
            return self._gen_rng_uniform(expr, env)
        elif callee == "rng_normal":
            return self._gen_rng_normal(expr, env)
        raise MaomiError(f"codegen: unknown RNG builtin '{callee}'", "<codegen>", 0, 0)

    def _gen_key_to_u64(self, key_ssa: str) -> str:
        """Convert i32[4] key to ui64[2] for rng_bit_generator.

        i32[4] -> bitcast -> ui32[4] -> reshape -> 2x2xui32 -> bitcast -> 2xui64
        """
        key_u32 = self._fresh()
        self._emit(f"{key_u32} = stablehlo.bitcast_convert {key_ssa} : (tensor<4xi32>) -> tensor<4xui32>")
        key_2x2 = self._fresh()
        self._emit(f"{key_2x2} = stablehlo.reshape {key_u32} : (tensor<4xui32>) -> tensor<2x2xui32>")
        key_u64 = self._fresh()
        self._emit(f"{key_u64} = stablehlo.bitcast_convert {key_2x2} : (tensor<2x2xui32>) -> tensor<2xui64>")
        return key_u64

    def _gen_rng_bits(self, key_u64: str, shape: tuple[int, ...]) -> tuple[str, str]:
        """Call rng_bit_generator, return (out_key_u64, bits_ui32)."""
        shape_str = "x".join(str(d) for d in shape)
        out_key = self._fresh()
        bits = self._fresh()
        self._emit(
            f"{out_key}, {bits} = stablehlo.rng_bit_generator {key_u64}, algorithm = DEFAULT "
            f": (tensor<2xui64>) -> (tensor<2xui64>, tensor<{shape_str}xui32>)"
        )
        return out_key, bits

    def _gen_bits_to_uniform(self, bits_ssa: str, shape: tuple[int, ...]) -> str:
        """Convert ui32 random bits to f32 uniform [0, 1).

        bits >> 9 | 0x3F800000 -> bitcast to f32 -> subtract 1.0
        """
        shape_str = "x".join(str(d) for d in shape)
        ui32_type = f"tensor<{shape_str}xui32>"
        f32_type = f"tensor<{shape_str}xf32>"

        # Broadcast constants to shape
        nine_s = self._fresh()
        self._emit(f"{nine_s} = stablehlo.constant dense<9> : tensor<ui32>")
        nine = self._fresh()
        dims = "" if not shape else "dims = []"
        self._emit(f"{nine} = stablehlo.broadcast_in_dim {nine_s}, {dims} : (tensor<ui32>) -> {ui32_type}")

        shifted = self._fresh()
        self._emit(f"{shifted} = stablehlo.shift_right_logical {bits_ssa}, {nine} : {ui32_type}")

        # 0x3F800000 = 1065353216 = bit pattern of 1.0f
        one_bits_s = self._fresh()
        self._emit(f"{one_bits_s} = stablehlo.constant dense<1065353216> : tensor<ui32>")
        one_bits = self._fresh()
        self._emit(f"{one_bits} = stablehlo.broadcast_in_dim {one_bits_s}, {dims} : (tensor<ui32>) -> {ui32_type}")

        mantissa = self._fresh()
        self._emit(f"{mantissa} = stablehlo.or {shifted}, {one_bits} : {ui32_type}")

        as_float = self._fresh()
        self._emit(f"{as_float} = stablehlo.bitcast_convert {mantissa} : ({ui32_type}) -> {f32_type}")

        one_f_s = self._fresh()
        self._emit(f"{one_f_s} = stablehlo.constant dense<1.000000e+00> : tensor<f32>")
        one_f = self._fresh()
        self._emit(f"{one_f} = stablehlo.broadcast_in_dim {one_f_s}, {dims} : (tensor<f32>) -> {f32_type}")

        result = self._fresh()
        self._emit(f"{result} = stablehlo.subtract {as_float}, {one_f} : {f32_type}")
        return result

    def _gen_rng_key(self, expr: CallExpr, env: dict[str, str]) -> str:
        """rng_key(seed) -> i32[4]: pack [0, 0, 0, seed]."""
        seed_ssa = self._gen_expr(expr.args[0], env)

        zeros = self._fresh()
        self._emit(f"{zeros} = stablehlo.constant dense<0> : tensor<3xi32>")
        seed_1d = self._fresh()
        self._emit(f"{seed_1d} = stablehlo.reshape {seed_ssa} : (tensor<i32>) -> tensor<1xi32>")
        key = self._fresh()
        self._emit(
            f"{key} = stablehlo.concatenate {zeros}, {seed_1d}, dim = 0 "
            f": (tensor<3xi32>, tensor<1xi32>) -> tensor<4xi32>"
        )
        return key

    def _gen_rng_split(self, expr: CallExpr, env: dict[str, str]) -> str:
        """rng_split(key, n) -> i32[n, 4]: generate n subkeys."""
        key_ssa = self._gen_expr(expr.args[0], env)
        n = expr.args[1].value  # IntLiteral, validated by type checker

        key_u64 = self._gen_key_to_u64(key_ssa)
        _, bits = self._gen_rng_bits(key_u64, (n * 4,))

        # Reshape flat ui32[n*4] -> ui32[n, 4]
        reshaped_u = self._fresh()
        self._emit(f"{reshaped_u} = stablehlo.reshape {bits} : (tensor<{n * 4}xui32>) -> tensor<{n}x4xui32>")

        # Bitcast back to i32[n, 4]
        result = self._fresh()
        self._emit(f"{result} = stablehlo.bitcast_convert {reshaped_u} : (tensor<{n}x4xui32>) -> tensor<{n}x4xi32>")
        return result

    def _gen_rng_uniform(self, expr: CallExpr, env: dict[str, str]) -> str:
        """rng_uniform(key, low, high, d1, ...) -> f32[d1, ...]."""
        key_ssa = self._gen_expr(expr.args[0], env)
        low_ssa = self._gen_expr(expr.args[1], env)
        high_ssa = self._gen_expr(expr.args[2], env)

        result_type = self._type_of(expr)
        assert isinstance(result_type, ArrayType)
        shape = tuple(result_type.dims)
        shape_str = "x".join(str(d) for d in shape)
        f32_type = f"tensor<{shape_str}xf32>"

        # Generate uniform [0, 1)
        key_u64 = self._gen_key_to_u64(key_ssa)
        _, bits = self._gen_rng_bits(key_u64, shape)
        base_uniform = self._gen_bits_to_uniform(bits, shape)

        # Scale to [low, high): result = base_uniform * (high - low) + low
        dims = "dims = []"
        low_bc = self._fresh()
        self._emit(f"{low_bc} = stablehlo.broadcast_in_dim {low_ssa}, {dims} : (tensor<f32>) -> {f32_type}")
        high_bc = self._fresh()
        self._emit(f"{high_bc} = stablehlo.broadcast_in_dim {high_ssa}, {dims} : (tensor<f32>) -> {f32_type}")

        range_val = self._fresh()
        self._emit(f"{range_val} = stablehlo.subtract {high_bc}, {low_bc} : {f32_type}")
        scaled = self._fresh()
        self._emit(f"{scaled} = stablehlo.multiply {base_uniform}, {range_val} : {f32_type}")
        result = self._fresh()
        self._emit(f"{result} = stablehlo.add {scaled}, {low_bc} : {f32_type}")
        return result

    def _gen_rng_normal(self, expr: CallExpr, env: dict[str, str]) -> str:
        """rng_normal(key, mean, std, d1, ...) -> f32[d1, ...].

        Uses Box-Muller: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
        Generates 2x elements, splits into pairs, applies transform.
        """
        key_ssa = self._gen_expr(expr.args[0], env)
        mean_ssa = self._gen_expr(expr.args[1], env)
        std_ssa = self._gen_expr(expr.args[2], env)

        result_type = self._type_of(expr)
        assert isinstance(result_type, ArrayType)
        shape = tuple(result_type.dims)
        numel = 1
        for d in shape:
            numel *= d

        shape_str = "x".join(str(d) for d in shape)
        f32_type = f"tensor<{shape_str}xf32>"
        flat_f32 = f"tensor<{numel}xf32>"

        # Generate 2*numel uniform values as a flat array
        key_u64 = self._gen_key_to_u64(key_ssa)
        _, bits = self._gen_rng_bits(key_u64, (2 * numel,))
        uniform_2n = self._gen_bits_to_uniform(bits, (2 * numel,))

        flat_2n = f"tensor<{2 * numel}xf32>"

        # Split into u1 and u2
        u1 = self._fresh()
        self._emit(f"{u1} = stablehlo.slice {uniform_2n} [0:{numel}] : ({flat_2n}) -> {flat_f32}")
        u2 = self._fresh()
        self._emit(f"{u2} = stablehlo.slice {uniform_2n} [{numel}:{2 * numel}] : ({flat_2n}) -> {flat_f32}")

        # Clamp u1 away from 0 (log safety)
        eps_s = self._fresh()
        self._emit(f"{eps_s} = stablehlo.constant dense<1.000000e-07> : tensor<f32>")
        eps = self._fresh()
        self._emit(f"{eps} = stablehlo.broadcast_in_dim {eps_s}, dims = [] : (tensor<f32>) -> {flat_f32}")
        u1_safe = self._fresh()
        self._emit(f"{u1_safe} = stablehlo.maximum {u1}, {eps} : {flat_f32}")

        # Box-Muller: z = sqrt(-2 * ln(u1)) * cos(2*pi*u2)
        log_u1 = self._fresh()
        self._emit(f"{log_u1} = stablehlo.log {u1_safe} : {flat_f32}")

        neg2_s = self._fresh()
        self._emit(f"{neg2_s} = stablehlo.constant dense<-2.000000e+00> : tensor<f32>")
        neg2 = self._fresh()
        self._emit(f"{neg2} = stablehlo.broadcast_in_dim {neg2_s}, dims = [] : (tensor<f32>) -> {flat_f32}")

        neg2_log = self._fresh()
        self._emit(f"{neg2_log} = stablehlo.multiply {neg2}, {log_u1} : {flat_f32}")
        radius = self._fresh()
        self._emit(f"{radius} = stablehlo.sqrt {neg2_log} : {flat_f32}")

        two_pi_s = self._fresh()
        self._emit(f"{two_pi_s} = stablehlo.constant dense<6.28318530e+00> : tensor<f32>")
        two_pi = self._fresh()
        self._emit(f"{two_pi} = stablehlo.broadcast_in_dim {two_pi_s}, dims = [] : (tensor<f32>) -> {flat_f32}")

        angle = self._fresh()
        self._emit(f"{angle} = stablehlo.multiply {two_pi}, {u2} : {flat_f32}")
        cos_angle = self._fresh()
        self._emit(f"{cos_angle} = stablehlo.cosine {angle} : {flat_f32}")

        z_flat = self._fresh()
        self._emit(f"{z_flat} = stablehlo.multiply {radius}, {cos_angle} : {flat_f32}")

        # Scale: result = z * std + mean
        std_bc = self._fresh()
        self._emit(f"{std_bc} = stablehlo.broadcast_in_dim {std_ssa}, dims = [] : (tensor<f32>) -> {flat_f32}")
        mean_bc = self._fresh()
        self._emit(f"{mean_bc} = stablehlo.broadcast_in_dim {mean_ssa}, dims = [] : (tensor<f32>) -> {flat_f32}")

        scaled = self._fresh()
        self._emit(f"{scaled} = stablehlo.multiply {z_flat}, {std_bc} : {flat_f32}")
        shifted = self._fresh()
        self._emit(f"{shifted} = stablehlo.add {scaled}, {mean_bc} : {flat_f32}")

        # Reshape from flat to target shape
        if len(shape) == 1:
            return shifted
        result = self._fresh()
        self._emit(f"{result} = stablehlo.reshape {shifted} : ({flat_f32}) -> {f32_type}")
        return result

    def _gen_index(self, expr: IndexExpr, env: dict[str, str]) -> str:
        base_type = self._type_of(expr.base)

        if not isinstance(base_type, ArrayType):
            raise MaomiError("codegen: indexing non-array", "<codegen>", expr.span.line_start, expr.span.col_start)

        # Check for array-based indexing (gather) — dispatch before scalar path
        for ic in expr.indices:
            if ic.kind == "single":
                idx_type = self._type_of(ic.value)
                if isinstance(idx_type, ArrayType):
                    return self._gen_gather(expr, env)

        base_ssa = self._gen_expr(expr.base, env)
        result_type = self._type_of(expr)

        # Build per-dimension start indices and slice sizes
        start_ssas: list[str] = []
        slice_sizes: list[int] = []
        squeezed_axes: list[int] = []  # axes to remove (single-indexed)
        all_static = True

        for i, ic in enumerate(expr.indices):
            dim = base_type.dims[i]
            if ic.kind == "single":
                idx_ssa = self._gen_expr(ic.value, env)
                # Runtime normalization for dynamic indices (may be negative)
                if not isinstance(ic.value, IntLiteral):
                    idx_ssa = self._normalize_index(idx_ssa, dim)
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
            # Custom assembly: [start:limit:stride, ...] or [start:limit, ...] when stride=1
            parts = []
            for s, l, st in zip(starts, limits, strides):
                if st == 1:
                    parts.append(f"{s}:{l}")
                else:
                    parts.append(f"{s}:{l}:{st}")
            slice_spec = ", ".join(parts)

            sliced = self._fresh()
            self._emit(
                f"{sliced} = stablehlo.slice {base_ssa} "
                f"[{slice_spec}] "
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
                # Runtime normalization for dynamic indices (may be negative)
                if not isinstance(ic.value, IntLiteral):
                    idx_ssa = self._normalize_index(idx_ssa, dim)
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

    # -- Gather / Scatter codegen --

    def _gen_gather(self, expr: IndexExpr, env: dict[str, str]) -> str:
        """Emit stablehlo.gather for array-based indexing (e.g., table[ids])."""
        base_ssa = self._gen_expr(expr.base, env)
        base_type = self._type_of(expr.base)
        result_type = self._type_of(expr)
        rank = len(base_type.dims)

        # Find the array-indexed axis
        gather_axis = 0
        indices_expr = None
        for i, ic in enumerate(expr.indices):
            if ic.kind == "single":
                idx_type = self._type_of(ic.value)
                if isinstance(idx_type, ArrayType):
                    gather_axis = i
                    indices_expr = ic.value
                    break

        indices_ssa = self._gen_expr(indices_expr, env)
        indices_type = self._type_of(indices_expr)
        B = indices_type.dims[0]

        # Reshape indices: i32[B] → i32[B, 1] (add index_vector_dim)
        reshaped_type = ArrayType(indices_type.base, (B, 1))
        reshaped_indices = self._fresh()
        self._emit(
            f"{reshaped_indices} = stablehlo.reshape {indices_ssa} "
            f": ({_mlir_type(indices_type)}) -> {_mlir_type(reshaped_type)}"
        )

        # Dimension numbers
        offset_dims = list(range(0, gather_axis)) + list(range(gather_axis + 1, rank))
        collapsed_slice_dims = [gather_axis]
        start_index_map = [gather_axis]

        # Slice sizes: 1 for gathered axis, full for others
        slice_sizes = [d for d in base_type.dims]
        slice_sizes[gather_axis] = 1

        # Format dimension number lists
        od_str = ", ".join(str(d) for d in offset_dims)
        cd_str = ", ".join(str(d) for d in collapsed_slice_dims)
        sim_str = ", ".join(str(d) for d in start_index_map)
        ss_str = ", ".join(str(s) for s in slice_sizes)

        # Build gather dimension_numbers attribute — omit offset_dims if empty
        dnums_parts = []
        if offset_dims:
            dnums_parts.append(f"offset_dims = [{od_str}]")
        dnums_parts.append(f"collapsed_slice_dims = [{cd_str}]")
        dnums_parts.append(f"start_index_map = [{sim_str}]")
        dnums_parts.append("index_vector_dim = 1")
        dnums_str = ", ".join(dnums_parts)

        result = self._fresh()
        self._emit(
            f'{result} = "stablehlo.gather"({base_ssa}, {reshaped_indices}) '
            f'<{{dimension_numbers = #stablehlo.gather<{dnums_str}>, '
            f'indices_are_sorted = false, '
            f'slice_sizes = array<i64: {ss_str}>}}> '
            f': ({_mlir_type(base_type)}, {_mlir_type(reshaped_type)}) -> {_mlir_type(result_type)}'
        )
        return result

    def _gen_gather_grad(self, expr: _GatherGrad, env: dict[str, str]) -> str:
        """Emit stablehlo.scatter (add-combiner) for gather gradient."""
        base_type = self._type_of(expr.base_expr)
        adj_ssa = self._gen_expr(expr.adj, env)
        adj_type = self._type_of(expr.adj)
        indices_ssa = self._gen_expr(expr.indices, env)
        indices_type = self._type_of(expr.indices)
        k = expr.gather_axis
        rank = len(base_type.dims)
        B = indices_type.dims[0]

        # Expected updates shape = gather output shape (base with dims[k] → B)
        updates_dims = list(base_type.dims)
        updates_dims[k] = B
        updates_type = ArrayType(base_type.base, tuple(updates_dims))

        # Broadcast adjoint if it's scalar or lower rank than expected
        if adj_type != updates_type:
            broadcast_dims: list[int] = []
            if isinstance(adj_type, ArrayType):
                # Map existing dims to output dims
                broadcast_dims = list(range(len(adj_type.dims)))
            broadcast_str = ", ".join(str(d) for d in broadcast_dims)
            broadcasted = self._fresh()
            self._emit(
                f"{broadcasted} = stablehlo.broadcast_in_dim {adj_ssa}, dims = [{broadcast_str}] "
                f": ({_mlir_type(adj_type)}) -> {_mlir_type(updates_type)}"
            )
            adj_ssa = broadcasted
            adj_type = updates_type

        mlir_base = _mlir_type(base_type)

        # Create zero tensor of base shape
        zeros = self._fresh()
        self._emit(f"{zeros} = stablehlo.constant dense<0.000000e+00> : {mlir_base}")

        # Reshape indices: i32[B] → i32[B, 1]
        B = indices_type.dims[0]
        reshaped_idx_type = ArrayType(indices_type.base, (B, 1))
        reshaped_indices = self._fresh()
        self._emit(
            f"{reshaped_indices} = stablehlo.reshape {indices_ssa} "
            f": ({_mlir_type(indices_type)}) -> {_mlir_type(reshaped_idx_type)}"
        )

        # Scatter dimension numbers (mirror of gather)
        update_window_dims = list(range(0, k)) + list(range(k + 1, rank))
        inserted_window_dims = [k]
        scatter_dims_to_operand_dims = [k]

        uwd_str = ", ".join(str(d) for d in update_window_dims)
        iwd_str = ", ".join(str(d) for d in inserted_window_dims)
        sdtod_str = ", ".join(str(d) for d in scatter_dims_to_operand_dims)

        # Build scatter dimension_numbers attribute — omit update_window_dims if empty
        sdnums_parts = []
        if update_window_dims:
            sdnums_parts.append(f"update_window_dims = [{uwd_str}]")
        sdnums_parts.append(f"inserted_window_dims = [{iwd_str}]")
        sdnums_parts.append(f"scatter_dims_to_operand_dims = [{sdtod_str}]")
        sdnums_parts.append("index_vector_dim = 1")
        sdnums_str = ", ".join(sdnums_parts)

        # Element type for combiner region — use fresh names to avoid SSA conflicts
        etype = _MLIR_ETYPE[base_type.base]
        lhs = self._fresh()
        rhs = self._fresh()
        add_var = self._fresh()

        result = self._fresh()
        self._emit(
            f'{result} = "stablehlo.scatter"({zeros}, {reshaped_indices}, {adj_ssa}) '
            f'<{{scatter_dimension_numbers = #stablehlo.scatter<{sdnums_str}>, '
            f'indices_are_sorted = false, unique_indices = false}}> '
            f'({{')
        self._emit(f'  ^bb0({lhs}: tensor<{etype}>, {rhs}: tensor<{etype}>):')
        self._emit(f'    {add_var} = stablehlo.add {lhs}, {rhs} : tensor<{etype}>')
        self._emit(f'    stablehlo.return {add_var} : tensor<{etype}>')
        self._emit(
            f'}}) : ({mlir_base}, {_mlir_type(reshaped_idx_type)}, {_mlir_type(adj_type)}) -> {mlir_base}'
        )
        return result

    # -- Conv/Pool backward codegen --

    @staticmethod
    def _dilate_dim(d: int, dilation: int) -> int:
        """max(0, 1 + dilation * (d - 1)) — same as JAX's core.dilate_dim."""
        return max(0, 1 + dilation * (d - 1))

    @staticmethod
    def _conv_vjp_lhs_padding(
        in_spatial: tuple[int, ...], kernel_spatial: tuple[int, ...],
        strides: tuple[int, ...], out_spatial: tuple[int, ...],
        padding: tuple[tuple[int, int], ...],
        lhs_dilation: tuple[int, ...], rhs_dilation: tuple[int, ...],
    ) -> list[tuple[int, int]]:
        """Compute VJP padding for input gradient (from JAX convolution.py:1007-1016)."""
        result = []
        for i in range(len(in_spatial)):
            ld = StableHLOCodegen._dilate_dim(in_spatial[i], lhs_dilation[i])
            rd = StableHLOCodegen._dilate_dim(kernel_spatial[i], rhs_dilation[i])
            od = StableHLOCodegen._dilate_dim(out_spatial[i], strides[i])
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
        """Compute VJP padding for kernel gradient (from JAX convolution.py:1019-1032)."""
        result = []
        for i in range(len(in_spatial)):
            ld = StableHLOCodegen._dilate_dim(in_spatial[i], lhs_dilation[i])
            rd = StableHLOCodegen._dilate_dim(kernel_spatial[i], rhs_dilation[i])
            od = StableHLOCodegen._dilate_dim(out_spatial[i], strides[i])
            pad_lo = padding[i][0]
            pads_from_lhs = od - ld
            pads_from_rhs = rd - pad_lo - 1
            pad_hi = pads_from_lhs + pads_from_rhs
            result.append((pad_lo, pad_hi))
        return result

    def _gen_conv2d_grad(self, expr: _Conv2dGrad, env: dict[str, str]) -> str:
        adj_ssa = self._gen_expr(expr.adj, env)
        adj_type = self._type_of(expr.adj)
        input_type = self._type_of(expr.input_expr)
        kernel_type = self._type_of(expr.kernel_expr)
        assert isinstance(input_type, ArrayType) and isinstance(kernel_type, ArrayType)
        assert isinstance(adj_type, ArrayType)

        N, Ci, H, W = input_type.dims
        Co, Ki, Kh, Kw = kernel_type.dims
        _, _, OH, OW = adj_type.dims
        sh, sw = expr.strides
        ph, pw = expr.padding
        fwd_padding = ((ph, ph), (pw, pw))

        if expr.wrt == "lhs":
            # grad w.r.t. input: conv(adj, reverse(kernel)) with VJP padding
            # reverse kernel along spatial dims [2, 3]
            kernel_ssa = self._gen_expr(expr.kernel_expr, env)
            rev_kernel = self._fresh()
            self._emit(
                f"{rev_kernel} = stablehlo.reverse {kernel_ssa}, dims = [2, 3] "
                f": {_mlir_type(kernel_type)}"
            )

            vjp_pad = self._conv_vjp_lhs_padding(
                (H, W), (Kh, Kw), (sh, sw), (OH, OW),
                fwd_padding, (1, 1), (1, 1),
            )

            # Transposed dimension numbers: swap lhs_spec <-> out_spec, transpose rhs_spec
            # Forward: [b,f,0,1]x[o,i,0,1]->[b,f,0,1]
            # Backward (lhs): [b,f,0,1]x[i,o,0,1]->[b,f,0,1]
            # (rhs spec transposed: swap o<->i)
            result_type = input_type  # gradient has same shape as input
            var = self._fresh()
            self._emit(
                f"{var} = stablehlo.convolution({adj_ssa}, {rev_kernel}) "
                f"dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1], "
                f"window = {{stride = [1, 1], pad = [[{vjp_pad[0][0]}, {vjp_pad[0][1]}], "
                f"[{vjp_pad[1][0]}, {vjp_pad[1][1]}]], "
                f"lhs_dilate = [{sh}, {sw}], rhs_dilate = [1, 1]}} "
                f"{{batch_group_count = 1 : i64, feature_group_count = 1 : i64}} "
                f": ({_mlir_type(adj_type)}, {_mlir_type(kernel_type)}) -> {_mlir_type(result_type)}"
            )
            return var

        else:  # wrt == "rhs"
            # grad w.r.t. kernel: conv(input, adj) with transposed roles
            input_ssa = self._gen_expr(expr.input_expr, env)

            vjp_pad = self._conv_vjp_rhs_padding(
                (H, W), (Kh, Kw), (sh, sw), (OH, OW),
                fwd_padding, (1, 1), (1, 1),
            )

            # Transposed dim numbers for kernel gradient (from JAX):
            # lhs (input): [f, b, 0, 1]
            # rhs (adj):   [i, o, 0, 1]  (StableHLO rhs always uses i/o labels)
            # out (kernel grad): [f, b, 0, 1]
            result_type = kernel_type  # gradient has same shape as kernel
            var = self._fresh()
            self._emit(
                f"{var} = stablehlo.convolution({input_ssa}, {adj_ssa}) "
                f"dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[f, b, 0, 1], "
                f"window = {{stride = [1, 1], pad = [[{vjp_pad[0][0]}, {vjp_pad[0][1]}], "
                f"[{vjp_pad[1][0]}, {vjp_pad[1][1]}]], "
                f"lhs_dilate = [1, 1], rhs_dilate = [{sh}, {sw}]}} "
                f"{{batch_group_count = 1 : i64, feature_group_count = 1 : i64}} "
                f": ({_mlir_type(input_type)}, {_mlir_type(adj_type)}) -> {_mlir_type(result_type)}"
            )
            return var

    def _gen_max_pool_grad(self, expr: _MaxPoolGrad, env: dict[str, str]) -> str:
        """Emit select_and_scatter for max_pool backward pass."""
        input_ssa = self._gen_expr(expr.input_expr, env)
        adj_ssa = self._gen_expr(expr.adj, env)
        input_type = self._type_of(expr.input_expr)
        adj_type = self._type_of(expr.adj)
        assert isinstance(input_type, ArrayType)

        wh, ww = expr.window
        sh, sw = expr.strides
        result_type = input_type  # gradient has same shape as input
        scalar_type = ScalarType(input_type.base)
        mlir_scalar = _mlir_type(scalar_type)

        # Init value (0.0)
        init_var = self._fresh()
        self._emit(f"{init_var} = stablehlo.constant dense<0.000000e+00> : {mlir_scalar}")

        var = self._fresh()
        self._emit(
            f"{var} = \"stablehlo.select_and_scatter\"({input_ssa}, {adj_ssa}, {init_var}) "
            f"({{")
        # Select region (ge comparator)
        self._indent += 1
        sa = self._fresh()
        sb = self._fresh()
        self._emit(f"^bb0({sa}: {mlir_scalar}, {sb}: {mlir_scalar}):")
        self._indent += 1
        cmp_var = self._fresh()
        self._emit(f"{cmp_var} = stablehlo.compare GE, {sa}, {sb} : ({mlir_scalar}, {mlir_scalar}) -> tensor<i1>")
        self._emit(f"stablehlo.return {cmp_var} : tensor<i1>")
        self._indent -= 1
        self._indent -= 1
        self._emit("}, {")
        # Scatter region (add)
        self._indent += 1
        sc = self._fresh()
        sd = self._fresh()
        self._emit(f"^bb0({sc}: {mlir_scalar}, {sd}: {mlir_scalar}):")
        self._indent += 1
        add_var = self._fresh()
        self._emit(f"{add_var} = stablehlo.add {sc}, {sd} : {mlir_scalar}")
        self._emit(f"stablehlo.return {add_var} : {mlir_scalar}")
        self._indent -= 1
        self._indent -= 1
        self._emit("}) {")
        self._indent += 1
        self._emit(f"window_dimensions = array<i64: 1, 1, {wh}, {ww}>,")
        self._emit(f"window_strides = array<i64: 1, 1, {sh}, {sw}>,")
        self._emit(f"padding = dense<0> : tensor<4x2xi64>")
        self._indent -= 1
        self._emit(
            f"}} : ({_mlir_type(input_type)}, {_mlir_type(adj_type)}, {mlir_scalar}) "
            f"-> {_mlir_type(result_type)}"
        )
        return var

    def _gen_avg_pool_grad(self, expr: _AvgPoolGrad, env: dict[str, str]) -> str:
        """Emit backward pass for avg_pool: scale, pad, reduce_window_sum."""
        adj_ssa = self._gen_expr(expr.adj, env)
        adj_type = self._type_of(expr.adj)
        input_type = self._type_of(expr.input_expr)
        assert isinstance(input_type, ArrayType) and isinstance(adj_type, ArrayType)

        wh, ww = expr.window
        sh, sw = expr.strides
        N, C, H, W = input_type.dims
        _, _, OH, OW = adj_type.dims

        # 1. Scale adjoint by 1/(wh*ww)
        count = float(wh * ww)
        scale_var = self._fresh()
        mlir_adj = _mlir_type(adj_type)
        self._emit(f"{scale_var} = stablehlo.constant dense<{1.0/count:e}> : {mlir_adj}")
        scaled = self._fresh()
        self._emit(f"{scaled} = stablehlo.multiply {adj_ssa}, {scale_var} : {mlir_adj}")

        # 2. Compute VJP padding
        vjp_pad = self._conv_vjp_lhs_padding(
            (1, 1, H, W), (1, 1, wh, ww), (1, 1, sh, sw), (1, 1, OH, OW),
            ((0, 0), (0, 0), (0, 0), (0, 0)), (1, 1, 1, 1), (1, 1, 1, 1),
        )

        # 3. Pad the scaled adjoint (edge padding + interior padding of stride-1)
        scalar_type = ScalarType(input_type.base)
        mlir_scalar = _mlir_type(scalar_type)
        zero_var = self._fresh()
        self._emit(f"{zero_var} = stablehlo.constant dense<0.000000e+00> : {mlir_scalar}")

        # Build padding config: [low, high, interior] for each dim
        # Interior padding inserts stride-1 zeros between elements
        interior = [0, 0, sh - 1, sw - 1]
        # Compute padded type
        padded_dims = []
        for i, d in enumerate(adj_type.dims):
            assert isinstance(d, int)
            lo, hi = vjp_pad[i]
            padded_d = lo + d + (d - 1) * interior[i] + hi
            padded_dims.append(padded_d)
        padded_type = ArrayType(input_type.base, tuple(padded_dims))

        padded = self._fresh()
        low_str = ", ".join(str(vjp_pad[i][0]) for i in range(4))
        high_str = ", ".join(str(vjp_pad[i][1]) for i in range(4))
        interior_str = ", ".join(str(interior[i]) for i in range(4))
        self._emit(
            f"{padded} = stablehlo.pad {scaled}, {zero_var}, "
            f"low = [{low_str}], high = [{high_str}], interior = [{interior_str}] "
            f": ({mlir_adj}, {mlir_scalar}) -> {_mlir_type(padded_type)}"
        )

        # 4. reduce_window sum with window=original window, stride=1
        result_type = input_type
        return self._gen_reduce_window(
            padded, padded_type, result_type,
            "stablehlo.add", "0.000000e+00",
            [1, 1, wh, ww], [1, 1, 1, 1],
            [(0, 0), (0, 0), (0, 0), (0, 0)],
        )

    def _gen_broadcast_expr(self, expr: _BroadcastExpr, env: dict[str, str]) -> str:
        """Emit broadcast_in_dim to broadcast scalar/lower-rank to target shape."""
        inner_ssa = self._gen_expr(expr.expr, env)
        inner_type = self._type_of(expr.expr)
        target_type = ArrayType(
            inner_type.base if hasattr(inner_type, 'base') else 'f32',
            expr.target_dims,
        )
        mlir_target = _mlir_type(target_type)
        var = self._fresh()
        if isinstance(inner_type, ScalarType):
            # Scalar → array: no broadcast_dimensions
            self._emit(
                f"{var} = stablehlo.broadcast_in_dim {inner_ssa}, "
                f"dims = [] : (tensor<{inner_type.base}>) -> {mlir_target}"
            )
        else:
            # Lower-rank array → higher-rank: compute broadcast dims
            assert isinstance(inner_type, ArrayType)
            if expr.broadcast_dims is not None:
                # Explicit dim mapping (e.g. from axis-specific reduction backprop)
                broadcast_dims = list(expr.broadcast_dims)
            else:
                # Right-align dimensions (numpy-style broadcasting)
                ndim_target = len(expr.target_dims)
                ndim_inner = len(inner_type.dims)
                broadcast_dims = list(range(ndim_target - ndim_inner, ndim_target))
            dims_str = ", ".join(str(d) for d in broadcast_dims)
            mlir_inner = _mlir_type(inner_type)
            self._emit(
                f"{var} = stablehlo.broadcast_in_dim {inner_ssa}, "
                f"dims = [{dims_str}] : ({mlir_inner}) -> {mlir_target}"
            )
        return var

    def _gen_reduce_sum_axes(self, expr: _ReduceSum, env: dict[str, str]) -> str:
        """Emit stablehlo.reduce (sum) over specific axes only."""
        arg_ssa = self._gen_expr(expr.expr, env)
        arg_type = self._type_of(expr.expr)
        result_type = self._type_of(expr)

        if not isinstance(arg_type, ArrayType):
            return arg_ssa

        dims_str = ", ".join(str(a) for a in expr.axes)

        init_var = self._fresh()
        scalar_type = ScalarType(arg_type.base)
        mlir_scalar = _mlir_type(scalar_type)
        self._emit(f"{init_var} = stablehlo.constant dense<0.000000e+00> : {mlir_scalar}")

        mlir_result = _mlir_type(result_type)
        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.reduce({arg_ssa} init: {init_var}) "
            f"across dimensions = [{dims_str}] "
            f": ({_mlir_type(arg_type)}, {mlir_scalar}) -> {mlir_result}"
        )
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

    # -- Scan helpers --

    def _slice_element(self, seq_ssa: str, idx_ssa: str, seq_type: ArrayType, elem_type: MaomiType, iter_axis: int = 0) -> str:
        """Slice a single element from a 1D+ array at a dynamic index along iter_axis."""
        mlir_seq = _mlir_type(seq_type)

        # Build slice sizes: all dims kept except iter_axis which becomes 1
        slice_sizes = list(seq_type.dims)
        slice_sizes[iter_axis] = 1
        sizes_str = ", ".join(str(s) for s in slice_sizes)
        sliced_type = ArrayType(seq_type.base, tuple(slice_sizes))
        mlir_sliced = _mlir_type(sliced_type)

        # Build start indices: 0 for all dims except iter_axis which is idx_ssa
        start_ssas = []
        for i in range(len(seq_type.dims)):
            if i == iter_axis:
                start_ssas.append(idx_ssa)
            else:
                z = self._fresh()
                self._emit(f"{z} = stablehlo.constant dense<0> : tensor<i32>")
                start_ssas.append(z)
        starts_str = ", ".join(start_ssas)
        start_types = ", ".join("tensor<i32>" for _ in start_ssas)

        sliced = self._fresh()
        self._emit(
            f"{sliced} = stablehlo.dynamic_slice {seq_ssa}, {starts_str}, "
            f"sizes = [{sizes_str}] : ({mlir_seq}, {start_types}) -> {mlir_sliced}"
        )
        # Reshape to remove the iter_axis dim (size 1)
        result = self._fresh()
        mlir_elem = _mlir_type(elem_type)
        self._emit(f"{result} = stablehlo.reshape {sliced} : ({mlir_sliced}) -> {mlir_elem}")
        return result

    def _update_element(self, arr_ssa: str, elem_ssa: str, idx_ssa: str,
                         arr_type: ArrayType, elem_type: MaomiType, iter_axis: int = 0) -> str:
        """Update a single element in an array at a dynamic index along iter_axis."""
        mlir_arr = _mlir_type(arr_type)
        # Reshape elem to insert size-1 dim at iter_axis
        if isinstance(elem_type, ScalarType):
            update_dims = (1,)
        elif isinstance(elem_type, ArrayType):
            update_dims = elem_type.dims[:iter_axis] + (1,) + elem_type.dims[iter_axis:]
        else:
            update_dims = (1,)
        update_type = ArrayType(arr_type.base, update_dims)
        mlir_update = _mlir_type(update_type)

        reshaped = self._fresh()
        self._emit(f"{reshaped} = stablehlo.reshape {elem_ssa} : ({_mlir_type(elem_type)}) -> {mlir_update}")

        # Build start indices: 0 for all dims except iter_axis which is idx_ssa
        start_ssas = []
        for i in range(len(arr_type.dims)):
            if i == iter_axis:
                start_ssas.append(idx_ssa)
            else:
                z = self._fresh()
                self._emit(f"{z} = stablehlo.constant dense<0> : tensor<i32>")
                start_ssas.append(z)
        starts_str = ", ".join(start_ssas)
        start_types = ", ".join("tensor<i32>" for _ in start_ssas)

        result = self._fresh()
        self._emit(
            f"{result} = stablehlo.dynamic_update_slice {arr_ssa}, {reshaped}, {starts_str} "
            f": ({mlir_arr}, {mlir_update}, {start_types}) -> {mlir_arr}"
        )
        return result

    # -- Scan codegen --

    # -- While loop codegen --

    def _gen_while(self, expr: WhileExpr, env: dict[str, str]) -> str:
        init_val = self._gen_expr(expr.init, env)
        state_type = self._type_of(expr.init)
        mlir_state = _mlir_type(state_type)

        uid = self._counter
        state_name = f"whileS{uid}"

        while_result = self._fresh()
        header = f"%{state_name} = {init_val}"
        self._emit(f"{while_result}:1 = stablehlo.while({header}) : {mlir_state}")

        # Cond region
        self._emit("cond {")
        self._indent += 1
        cond_env = dict(env)
        cond_env[expr.state_var] = f"%{state_name}"
        cond_val = self._gen_block(expr.cond, cond_env)
        self._emit(f"stablehlo.return {cond_val} : tensor<i1>")
        self._indent -= 1

        # Body region
        self._emit("} do {")
        self._indent += 1
        body_env = dict(env)
        body_env[expr.state_var] = f"%{state_name}"
        new_state = self._gen_block(expr.body, body_env)
        self._emit(f"stablehlo.return {new_state} : {mlir_state}")
        self._indent -= 1
        self._emit("}")

        return f"{while_result}#0"

    def _gen_while_grad(self, expr: _WhileGrad, env: dict[str, str]) -> str:
        """Emit augmented forward + reverse while for bounded while backward pass."""
        # Generate the forward result (the original while loop)
        fwd_val = self._gen_expr(expr.forward_result, env)

        state_type = self._type_of(expr.init)
        mlir_state = _mlir_type(state_type)
        mlir_i32 = "tensor<i32>"
        mlir_bool = "tensor<i1>"
        max_iters = expr.max_iters

        # Build trajectory type: tensor<max_iters x ...state_shape>
        if isinstance(state_type, ScalarType):
            traj_type = ArrayType(state_type.base, (max_iters,))
        elif isinstance(state_type, ArrayType):
            traj_type = ArrayType(state_type.base, (max_iters,) + state_type.dims)
        else:
            raise MaomiError("codegen: while grad only supports scalar/array state", "<codegen>", 0, 0)
        mlir_traj = _mlir_type(traj_type)

        # ── Forward augmented while: save trajectory ──
        init_val = self._gen_expr(expr.init, env)

        # Pre-allocate trajectory buffer (zeros)
        traj_init = self._fresh()
        self._emit(f"{traj_init} = stablehlo.constant dense<0.000000e+00> : {mlir_traj}")

        # Step counter = 0
        step_init = self._fresh()
        self._emit(f"{step_init} = stablehlo.constant dense<0> : {mlir_i32}")

        # running = true
        run_init = self._fresh()
        self._emit(f"{run_init} = stablehlo.constant dense<true> : {mlir_bool}")

        uid = self._counter
        s_name = f"fwdS{uid}"
        step_name = f"fwdStep{uid}"
        traj_name = f"fwdTraj{uid}"
        run_name = f"fwdRun{uid}"

        fwd_result = self._fresh()
        header_parts = ", ".join([
            f"%{s_name} = {init_val}",
            f"%{step_name} = {step_init}",
            f"%{traj_name} = {traj_init}",
            f"%{run_name} = {run_init}",
        ])
        types_str = f"{mlir_state}, {mlir_i32}, {mlir_traj}, {mlir_bool}"
        self._emit(f"{fwd_result}:4 = stablehlo.while({header_parts}) : {types_str}")

        # Cond: running && step < max_iters
        self._emit("cond {")
        self._indent += 1
        max_v = self._fresh()
        self._emit(f"{max_v} = stablehlo.constant dense<{max_iters}> : {mlir_i32}")
        lt_v = self._fresh()
        self._emit(f"{lt_v} = stablehlo.compare LT, %{step_name}, {max_v}, SIGNED : ({mlir_i32}, {mlir_i32}) -> {mlir_bool}")
        and_v = self._fresh()
        self._emit(f"{and_v} = stablehlo.and %{run_name}, {lt_v} : {mlir_bool}")
        self._emit(f"stablehlo.return {and_v} : {mlir_bool}")
        self._indent -= 1

        # Body: save state to trajectory, run body, check cond on new state
        self._emit("} do {")
        self._indent += 1

        # Save current state to trajectory at position step
        if isinstance(state_type, ScalarType):
            reshaped = self._fresh()
            self._emit(f"{reshaped} = stablehlo.reshape %{s_name} : ({mlir_state}) -> tensor<1x{_MLIR_ETYPE[state_type.base]}>")
            new_traj = self._fresh()
            self._emit(f"{new_traj} = stablehlo.dynamic_update_slice %{traj_name}, {reshaped}, %{step_name} : ({mlir_traj}, tensor<1x{_MLIR_ETYPE[state_type.base]}>, {mlir_i32}) -> {mlir_traj}")
        elif isinstance(state_type, ArrayType):
            reshaped = self._fresh()
            inner_shape = "x".join(str(d) for d in state_type.dims)
            etype = _MLIR_ETYPE[state_type.base]
            self._emit(f"{reshaped} = stablehlo.reshape %{s_name} : ({mlir_state}) -> tensor<1x{inner_shape}x{etype}>")
            # Start indices: step for dim 0, then zeros for remaining dims
            zeros = [self._fresh() for _ in state_type.dims]
            for z in zeros:
                self._emit(f"{z} = stablehlo.constant dense<0> : {mlir_i32}")
            start_indices = f"%{step_name}, " + ", ".join(zeros)
            slice_shape = f"1x{inner_shape}x{etype}"
            new_traj = self._fresh()
            self._emit(f"{new_traj} = stablehlo.dynamic_update_slice %{traj_name}, {reshaped}, {start_indices} : ({mlir_traj}, tensor<{slice_shape}>, {', '.join([mlir_i32] * (1 + len(state_type.dims)))}) -> {mlir_traj}")

        # Run body
        body_env = dict(env)
        body_env[expr.state_var] = f"%{s_name}"
        new_state = self._gen_block(expr.body, body_env)

        # Check condition on new state
        cond_env = dict(env)
        cond_env[expr.state_var] = new_state
        still_running = self._gen_block(expr.cond, cond_env)

        # Increment step
        one_v = self._fresh()
        self._emit(f"{one_v} = stablehlo.constant dense<1> : {mlir_i32}")
        new_step = self._fresh()
        self._emit(f"{new_step} = stablehlo.add %{step_name}, {one_v} : {mlir_i32}")

        vals_str = f"{new_state}, {new_step}, {new_traj}, {still_running}"
        self._emit(f"stablehlo.return {vals_str} : {types_str}")
        self._indent -= 1
        self._emit("}")

        # Extract: num_iters = fwd_result#1, trajectory = fwd_result#2
        fwd_num_iters = f"{fwd_result}#1"
        fwd_trajectory = f"{fwd_result}#2"

        # ── Backward while: reverse through trajectory ──
        adj_val = self._gen_expr(expr.adj, env)

        # Start step = num_iters - 1
        one_bwd = self._fresh()
        self._emit(f"{one_bwd} = stablehlo.constant dense<1> : {mlir_i32}")
        start_step = self._fresh()
        self._emit(f"{start_step} = stablehlo.subtract {fwd_num_iters}, {one_bwd} : {mlir_i32}")

        uid2 = self._counter
        badj_name = f"bwdAdj{uid2}"
        bstep_name = f"bwdStep{uid2}"

        bwd_result = self._fresh()
        bwd_header = f"%{badj_name} = {adj_val}, %{bstep_name} = {start_step}"
        bwd_types = f"{mlir_state}, {mlir_i32}"
        self._emit(f"{bwd_result}:2 = stablehlo.while({bwd_header}) : {bwd_types}")

        # Cond: step >= 0
        self._emit("cond {")
        self._indent += 1
        zero_v = self._fresh()
        self._emit(f"{zero_v} = stablehlo.constant dense<0> : {mlir_i32}")
        ge_v = self._fresh()
        self._emit(f"{ge_v} = stablehlo.compare GE, %{bstep_name}, {zero_v}, SIGNED : ({mlir_i32}, {mlir_i32}) -> {mlir_bool}")
        self._emit(f"stablehlo.return {ge_v} : {mlir_bool}")
        self._indent -= 1

        # Body: read saved state, evaluate derivative, multiply
        self._emit("} do {")
        self._indent += 1

        # Read saved state from trajectory at step
        if isinstance(state_type, ScalarType):
            etype = _MLIR_ETYPE[state_type.base]
            sliced = self._fresh()
            self._emit(f"{sliced} = stablehlo.dynamic_slice {fwd_trajectory}, %{bstep_name}, sizes = [1] : ({mlir_traj}, {mlir_i32}) -> tensor<1x{etype}>")
            saved_state = self._fresh()
            self._emit(f"{saved_state} = stablehlo.reshape {sliced} : (tensor<1x{etype}>) -> {mlir_state}")
        elif isinstance(state_type, ArrayType):
            inner_shape = "x".join(str(d) for d in state_type.dims)
            etype = _MLIR_ETYPE[state_type.base]
            zeros_bwd = [self._fresh() for _ in state_type.dims]
            for z in zeros_bwd:
                self._emit(f"{z} = stablehlo.constant dense<0> : {mlir_i32}")
            start_indices_bwd = f"%{bstep_name}, " + ", ".join(zeros_bwd)
            n_idx = 1 + len(state_type.dims)
            slice_sizes = f"1x{inner_shape}x{etype}"
            sliced = self._fresh()
            self._emit(f"{sliced} = stablehlo.dynamic_slice {fwd_trajectory}, {start_indices_bwd}, sizes = [1, {', '.join(str(d) for d in state_type.dims)}] : ({mlir_traj}, {', '.join([mlir_i32] * n_idx)}) -> tensor<{slice_sizes}>")
            saved_state = self._fresh()
            self._emit(f"{saved_state} = stablehlo.reshape {sliced} : (tensor<{slice_sizes}>) -> {mlir_state}")

        # Evaluate d_body_d_state with state_var = saved_state
        deriv_env = dict(env)
        deriv_env[expr.state_var] = saved_state
        d_val = self._gen_expr(expr.d_body_d_state, deriv_env)

        # new_adj = adj * d_val
        new_adj = self._fresh()
        self._emit(f"{new_adj} = stablehlo.multiply %{badj_name}, {d_val} : {mlir_state}")

        # Decrement step
        one_bwd2 = self._fresh()
        self._emit(f"{one_bwd2} = stablehlo.constant dense<1> : {mlir_i32}")
        new_bstep = self._fresh()
        self._emit(f"{new_bstep} = stablehlo.subtract %{bstep_name}, {one_bwd2} : {mlir_i32}")

        self._emit(f"stablehlo.return {new_adj}, {new_bstep} : {bwd_types}")
        self._indent -= 1
        self._emit("}")

        # Result is the accumulated adjoint (index 0)
        return f"{bwd_result}#0"

    def _gen_scan(self, expr: ScanExpr, env: dict[str, str]) -> str:
        init_val = self._gen_expr(expr.init, env)
        seq_vals = [self._gen_expr(s, env) for s in expr.sequences]

        init_type = self._type_of(expr.init)
        seq_types = [self._type_of(s) for s in expr.sequences]
        result_type = self._type_of(expr)

        for st in seq_types:
            if not isinstance(st, ArrayType):
                raise MaomiError("codegen: scan sequence must be array", "<codegen>", 0, 0)

        bd = self._batch_depth
        iter_axis = bd  # batch dims occupy 0..bd-1, iteration is along dim bd

        seq_len = seq_types[0].dims[iter_axis]
        if isinstance(seq_len, str):
            raise MaomiError(f"codegen: scan requires concrete sequence length, got '{seq_len}'", "<codegen>", 0, 0)

        # Element types: strip dim at iter_axis (keep batch dims before it)
        elem_types: list[MaomiType] = []
        for st in seq_types:
            remaining = st.dims[:iter_axis] + st.dims[iter_axis + 1:]
            if len(remaining) == 0:
                elem_types.append(ScalarType(st.base))
            else:
                elem_types.append(ArrayType(st.base, remaining))

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

        # Slice elements from each sequence (skip if elem_var unused in body)
        body_env = dict(env)
        body_env[expr.carry_var] = f"%{carry_name}"
        for i, (ev, st, et) in enumerate(zip(expr.elem_vars, seq_types, elem_types)):
            if _block_references_var(expr.body, ev):
                b_elem = self._slice_element(f"%{seq_names[i]}", f"%{ctr_name}", st, et, iter_axis=iter_axis)
                body_env[ev] = b_elem

        new_carry = self._gen_block(expr.body, body_env)

        # Update output
        new_output = self._update_element(f"%{out_name}", new_carry, f"%{ctr_name}", result_type, init_type, iter_axis=iter_axis)

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

    # -- Fold codegen --

    def _gen_fold(self, expr: FoldExpr, env: dict[str, str]) -> str:
        init_val = self._gen_expr(expr.init, env)
        seq_vals = [self._gen_expr(s, env) for s in expr.sequences]

        init_type = self._type_of(expr.init)
        seq_types = [self._type_of(s) for s in expr.sequences]

        for st in seq_types:
            if not isinstance(st, ArrayType):
                raise MaomiError("codegen: fold sequence must be array", "<codegen>", 0, 0)

        bd = self._batch_depth
        iter_axis = bd

        seq_len = seq_types[0].dims[iter_axis]
        if isinstance(seq_len, str):
            raise MaomiError(f"codegen: fold requires concrete sequence length, got '{seq_len}'", "<codegen>", 0, 0)

        elem_types: list[MaomiType] = []
        for st in seq_types:
            remaining = st.dims[:iter_axis] + st.dims[iter_axis + 1:]
            if len(remaining) == 0:
                elem_types.append(ScalarType(st.base))
            else:
                elem_types.append(ArrayType(st.base, remaining))

        mlir_init = _mlir_type(init_type)
        mlir_seqs = [_mlir_type(st) for st in seq_types]
        mlir_i32 = "tensor<i32>"

        counter_var = self._fresh()
        self._emit(f"{counter_var} = stablehlo.constant dense<0> : {mlir_i32}")

        n_seqs = len(seq_vals)
        uid = self._counter
        ctr_name = f"foldC{uid}"
        carry_name = f"foldK{uid}"
        seq_names = [f"foldS{uid}_{i}" for i in range(n_seqs)]

        arg_names = [ctr_name, carry_name] + seq_names
        init_vals_list = [counter_var, init_val] + seq_vals
        arg_types = [mlir_i32, mlir_init] + mlir_seqs
        n_args = len(arg_names)

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
        self._emit(f"{limit_v} = stablehlo.constant dense<{seq_len}> : {mlir_i32}")
        c_cmp = self._fresh()
        self._emit(f"{c_cmp} = stablehlo.compare LT, %{ctr_name}, {limit_v}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        self._emit(f"stablehlo.return {c_cmp} : tensor<i1>")
        self._indent -= 1

        # Body region
        self._emit("} do {")
        self._indent += 1

        body_env = dict(env)
        body_env[expr.carry_var] = f"%{carry_name}"
        for i, (ev, st, et) in enumerate(zip(expr.elem_vars, seq_types, elem_types)):
            if _block_references_var(expr.body, ev):
                b_elem = self._slice_element(f"%{seq_names[i]}", f"%{ctr_name}", st, et, iter_axis=iter_axis)
                body_env[ev] = b_elem

        new_carry = self._gen_block(expr.body, body_env)

        # Update counter
        one_v = self._fresh()
        self._emit(f"{one_v} = stablehlo.constant dense<1> : {mlir_i32}")
        new_counter = self._fresh()
        self._emit(f"{new_counter} = stablehlo.add %{ctr_name}, {one_v} : {mlir_i32}")

        # Return new values
        new_vals = [new_counter, new_carry] + [f"%{sn}" for sn in seq_names]
        vals_str = ", ".join(new_vals)
        self._emit(f"stablehlo.return {vals_str} : {types_str}")
        self._indent -= 1
        self._emit("}")

        # Result is the final carry (index 1)
        return f"{while_result}#1"

    # -- Fold gradient codegen --

    def _gen_fold_grad(self, expr: _FoldGrad, env: dict[str, str]) -> str:
        """Emit augmented forward + reverse while loop for fold backward pass.

        Phase A: Run fold body forward, stacking carries into a trajectory buffer.
        Phase B: Build padded adjoint array (zeros with adj at last position).
        Phase C: Reverse while loop identical to _gen_scan_grad.
        """
        init_val = self._gen_expr(expr.init, env)
        seq_vals = [self._gen_expr(s, env) for s in expr.sequences]
        adj_val = self._gen_expr(expr.adj, env)

        init_type = self._type_of(expr.init)
        seq_types = [self._type_of(s) for s in expr.sequences]
        adj_type = self._type_of(expr.adj)

        for st in seq_types:
            if not isinstance(st, ArrayType):
                raise MaomiError("codegen: _FoldGrad sequence must be array", "<codegen>", 0, 0)

        seq_len = seq_types[0].dims[0]
        if isinstance(seq_len, str):
            raise MaomiError(f"codegen: _FoldGrad requires concrete length, got '{seq_len}'", "<codegen>", 0, 0)

        elem_types: list[MaomiType] = []
        for st in seq_types:
            if len(st.dims) == 1:
                elem_types.append(ScalarType(st.base))
            else:
                elem_types.append(ArrayType(st.base, st.dims[1:]))

        # Trajectory type: stack init_type along a new leading dim of seq_len
        if isinstance(init_type, ScalarType):
            traj_type = ArrayType(init_type.base, (seq_len,))
        elif isinstance(init_type, ArrayType):
            traj_type = ArrayType(init_type.base, (seq_len,) + init_type.dims)
        else:
            raise MaomiError("codegen: _FoldGrad init must be scalar or array", "<codegen>", 0, 0)

        n_seqs = len(expr.sequences)
        mlir_i32 = "tensor<i32>"
        mlir_init = _mlir_type(init_type)
        mlir_traj = _mlir_type(traj_type)
        mlir_seqs = [_mlir_type(st) for st in seq_types]

        # ---- Phase A: Augmented forward (build trajectory) ----

        fwd_counter = self._fresh()
        self._emit(f"{fwd_counter} = stablehlo.constant dense<0> : {mlir_i32}")

        fwd_traj_init = self._fresh()
        self._emit(f"{fwd_traj_init} = stablehlo.constant dense<0.000000e+00> : {mlir_traj}")

        uid_fwd = self._counter
        fwd_ctr_name = f"fgFC{uid_fwd}"
        fwd_carry_name = f"fgFK{uid_fwd}"
        fwd_traj_name = f"fgFT{uid_fwd}"
        fwd_seq_names = [f"fgFS{uid_fwd}_{i}" for i in range(n_seqs)]

        fwd_arg_names = [fwd_ctr_name, fwd_carry_name, fwd_traj_name] + fwd_seq_names
        fwd_init_vals = [fwd_counter, init_val, fwd_traj_init] + seq_vals
        fwd_arg_types = [mlir_i32, mlir_init, mlir_traj] + mlir_seqs
        fwd_n_args = len(fwd_arg_names)

        fwd_while = self._fresh()
        fwd_header = ", ".join(
            f"%{fwd_arg_names[i]} = {fwd_init_vals[i]}" for i in range(fwd_n_args)
        )
        fwd_types_str = ", ".join(fwd_arg_types)
        self._emit(f"{fwd_while}:{fwd_n_args} = stablehlo.while({fwd_header}) : {fwd_types_str}")

        # Cond: counter < seq_len
        self._emit("cond {")
        self._indent += 1
        fwd_limit = self._fresh()
        self._emit(f"{fwd_limit} = stablehlo.constant dense<{seq_len}> : {mlir_i32}")
        fwd_cmp = self._fresh()
        self._emit(f"{fwd_cmp} = stablehlo.compare LT, %{fwd_ctr_name}, {fwd_limit}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        self._emit(f"stablehlo.return {fwd_cmp} : tensor<i1>")
        self._indent -= 1

        # Body: compute new_carry from fold body, stack into trajectory
        self._emit("} do {")
        self._indent += 1

        fwd_body_env = dict(env)
        fwd_body_env[expr.carry_var] = f"%{fwd_carry_name}"
        for i, (ev, st, et) in enumerate(zip(expr.elem_vars, seq_types, elem_types)):
            if _block_references_var(expr.body, ev):
                b_elem = self._slice_element(f"%{fwd_seq_names[i]}", f"%{fwd_ctr_name}", st, et)
                fwd_body_env[ev] = b_elem

        new_carry = self._gen_block(expr.body, fwd_body_env)

        # Stack new_carry into trajectory at counter position
        new_traj = self._update_element(f"%{fwd_traj_name}", new_carry, f"%{fwd_ctr_name}", traj_type, init_type)

        # Increment counter
        fwd_one = self._fresh()
        self._emit(f"{fwd_one} = stablehlo.constant dense<1> : {mlir_i32}")
        fwd_new_ctr = self._fresh()
        self._emit(f"{fwd_new_ctr} = stablehlo.add %{fwd_ctr_name}, {fwd_one} : {mlir_i32}")

        fwd_new_vals = [fwd_new_ctr, new_carry, new_traj] + [f"%{sn}" for sn in fwd_seq_names]
        fwd_vals_str = ", ".join(fwd_new_vals)
        self._emit(f"stablehlo.return {fwd_vals_str} : {fwd_types_str}")
        self._indent -= 1
        self._emit("}")

        # Extract trajectory from augmented forward result (index 2)
        fwd_trajectory = f"{fwd_while}#2"

        # ---- Phase B: Padded adjoint array ----
        # Build adj_array = zeros(traj_type) with adj placed at position seq_len - 1

        adj_zeros = self._fresh()
        self._emit(f"{adj_zeros} = stablehlo.constant dense<0.000000e+00> : {mlir_traj}")

        last_idx = self._fresh()
        self._emit(f"{last_idx} = stablehlo.constant dense<{seq_len - 1}> : {mlir_i32}")

        # Broadcast scalar adj to init_type if needed
        if isinstance(adj_type, ScalarType) and isinstance(init_type, ArrayType):
            adj_val = self._maybe_broadcast(adj_val, adj_type, init_type)

        adj_array = self._update_element(adj_zeros, adj_val, last_idx, traj_type, init_type)

        # ---- Phase C: Reverse while loop (same structure as _gen_scan_grad) ----

        rev_counter = self._fresh()
        self._emit(f"{rev_counter} = stablehlo.constant dense<{seq_len - 1}> : {mlir_i32}")

        adj_carry_init = self._fresh()
        self._emit(f"{adj_carry_init} = stablehlo.constant dense<0.000000e+00> : {mlir_init}")

        adj_seq_inits = []
        for ms in mlir_seqs:
            v = self._fresh()
            self._emit(f"{v} = stablehlo.constant dense<0.000000e+00> : {ms}")
            adj_seq_inits.append(v)

        uid_rev = self._counter
        gc_name = f"fgRC{uid_rev}"
        gac_name = f"fgRAC{uid_rev}"
        gas_names = [f"fgRAS{uid_rev}_{i}" for i in range(n_seqs)]
        gfwd_name = f"fgRFwd{uid_rev}"
        gos_names = [f"fgROS{uid_rev}_{i}" for i in range(n_seqs)]
        gadj_name = f"fgRAdj{uid_rev}"
        ginit_name = f"fgRInit{uid_rev}"

        arg_names = [gc_name, gac_name]
        init_vals = [rev_counter, adj_carry_init]
        arg_types_list = [mlir_i32, mlir_init]

        for i in range(n_seqs):
            arg_names.append(gas_names[i])
            init_vals.append(adj_seq_inits[i])
            arg_types_list.append(mlir_seqs[i])

        arg_names.append(gfwd_name)
        init_vals.append(fwd_trajectory)
        arg_types_list.append(mlir_traj)

        for i in range(n_seqs):
            arg_names.append(gos_names[i])
            init_vals.append(seq_vals[i])
            arg_types_list.append(mlir_seqs[i])

        arg_names.append(gadj_name)
        init_vals.append(adj_array)
        arg_types_list.append(mlir_traj)

        arg_names.append(ginit_name)
        init_vals.append(init_val)
        arg_types_list.append(mlir_init)

        n_args = len(arg_names)
        types_str = ", ".join(arg_types_list)

        while_result = self._fresh()
        header_parts = ", ".join(
            f"%{arg_names[i]} = {init_vals[i]}" for i in range(n_args)
        )
        self._emit(f"{while_result}:{n_args} = stablehlo.while({header_parts}) : {types_str}")

        # Cond: counter >= 0
        self._emit("cond {")
        self._indent += 1
        zero_c = self._fresh()
        self._emit(f"{zero_c} = stablehlo.constant dense<0> : {mlir_i32}")
        c_cmp = self._fresh()
        self._emit(f"{c_cmp} = stablehlo.compare GE, %{gc_name}, {zero_c}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        self._emit(f"stablehlo.return {c_cmp} : tensor<i1>")
        self._indent -= 1

        # Body
        self._emit("} do {")
        self._indent += 1

        # Slice adj_t from adj_array
        adj_t = self._slice_element(f"%{gadj_name}", f"%{gc_name}", traj_type, init_type)

        # adj_total = adj_carry + adj_t
        adj_total = self._fresh()
        self._emit(f"{adj_total} = stablehlo.add %{gac_name}, {adj_t} : {mlir_init}")

        # prev_carry = select(counter > 0, trajectory[counter-1], init)
        one_b = self._fresh()
        self._emit(f"{one_b} = stablehlo.constant dense<1> : {mlir_i32}")
        zero_b = self._fresh()
        self._emit(f"{zero_b} = stablehlo.constant dense<0> : {mlir_i32}")

        prev_idx = self._fresh()
        self._emit(f"{prev_idx} = stablehlo.subtract %{gc_name}, {one_b} : {mlir_i32}")
        clamped_idx = self._fresh()
        self._emit(f"{clamped_idx} = stablehlo.clamp {zero_b}, {prev_idx}, %{gc_name} : ({mlir_i32}, {mlir_i32}, {mlir_i32}) -> {mlir_i32}")

        fwd_prev = self._slice_element(f"%{gfwd_name}", clamped_idx, traj_type, init_type)

        gt_zero = self._fresh()
        self._emit(f"{gt_zero} = stablehlo.compare GT, %{gc_name}, {zero_b}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")

        prev_carry = self._fresh()
        self._emit(f"{prev_carry} = stablehlo.select {gt_zero}, {fwd_prev}, %{ginit_name} : (tensor<i1>, {mlir_init}, {mlir_init}) -> {mlir_init}")

        # Slice elements from original sequences
        b_elems = []
        for i in range(n_seqs):
            elem = self._slice_element(f"%{gos_names[i]}", f"%{gc_name}", seq_types[i], elem_types[i])
            b_elems.append(elem)

        # Evaluate derivative expressions
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
