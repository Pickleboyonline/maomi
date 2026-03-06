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
    _ScanGrad,
    Expr,
)
from .types import MaomiType, ScalarType, ArrayType
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
            case _ScanGrad():
                return self._gen_scan_grad(expr, env)
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

    # -- Scan helpers --

    def _slice_element(self, seq_ssa: str, idx_ssa: str, seq_type: ArrayType, elem_type: MaomiType) -> str:
        """Slice a single element from a 1D+ array at a dynamic index, returning scalar or lower-rank."""
        mlir_seq = _mlir_type(seq_type)
        mlir_i32 = "tensor<i32>"
        # Slice with size 1 along first dim, full size on remaining dims
        slice_sizes = [1] + [d for d in seq_type.dims[1:]]
        sizes_str = ", ".join(str(s) for s in slice_sizes)
        sliced_dims = tuple(slice_sizes)
        sliced_type = ArrayType(seq_type.base, sliced_dims)
        mlir_sliced = _mlir_type(sliced_type)

        sliced = self._fresh()
        self._emit(
            f'{sliced} = "stablehlo.dynamic_slice"({seq_ssa}, {idx_ssa}) '
            f"{{slice_sizes = array<i64: {sizes_str}>}} "
            f": ({mlir_seq}, {mlir_i32}) -> {mlir_sliced}"
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
        mlir_i32 = "tensor<i32>"
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
            f": ({mlir_arr}, {mlir_update}, {mlir_i32}) -> {mlir_arr}"
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

        # Create initial values
        counter_var = self._fresh()
        if expr.reverse:
            self._emit(f"{counter_var} = stablehlo.constant dense<{seq_len - 1}> : {mlir_i32}")
        else:
            self._emit(f"{counter_var} = stablehlo.constant dense<0> : {mlir_i32}")

        output_var = self._fresh()
        self._emit(f"{output_var} = stablehlo.constant dense<0.000000e+00> : {mlir_result}")

        # Limit/zero for condition (defined outside while so it's accessible inside)
        limit_var = self._fresh()
        if expr.reverse:
            self._emit(f"{limit_var} = stablehlo.constant dense<0> : {mlir_i32}")
        else:
            self._emit(f"{limit_var} = stablehlo.constant dense<{seq_len}> : {mlir_i32}")

        one_var = self._fresh()
        self._emit(f"{one_var} = stablehlo.constant dense<1> : {mlir_i32}")

        # Build while arg names and types (unique per while loop)
        n_seqs = len(seq_vals)
        uid = self._counter  # unique prefix to avoid name collisions
        ctr_name = f"_c{uid}"
        carry_name = f"_k{uid}"
        out_name = f"_o{uid}"
        seq_names = [f"_s{uid}_{i}" for i in range(n_seqs)]
        arg_names = [ctr_name, carry_name, out_name] + seq_names
        init_vals = [counter_var, init_val, output_var] + seq_vals
        arg_types = [mlir_i32, mlir_init, mlir_result] + mlir_seqs
        n_args = len(arg_names)

        # While header: %result:N = stablehlo.while(%a = %v, ...) : types
        while_result = self._fresh()
        while_args = ", ".join(f"%{arg_names[i]} = {init_vals[i]}" for i in range(n_args))
        types_str = ", ".join(arg_types)
        self._emit(f"{while_result}:{n_args} = stablehlo.while({while_args}) : {types_str}")

        # Condition region
        self._indent += 1
        self._emit("cond {")
        self._indent += 1
        if expr.reverse:
            c_cmp = self._fresh()
            self._emit(f"{c_cmp} = stablehlo.compare GE, %{ctr_name}, {limit_var} : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        else:
            c_cmp = self._fresh()
            self._emit(f"{c_cmp} = stablehlo.compare LT, %{ctr_name}, {limit_var} : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        self._emit(f"stablehlo.return {c_cmp} : tensor<i1>")
        self._indent -= 1
        self._emit("}")

        # Body region
        self._emit("do {")
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
        new_counter = self._fresh()
        if expr.reverse:
            self._emit(f"{new_counter} = stablehlo.subtract %{ctr_name}, {one_var} : {mlir_i32}")
        else:
            self._emit(f"{new_counter} = stablehlo.add %{ctr_name}, {one_var} : {mlir_i32}")

        # Return new values
        new_vals = [new_counter, new_carry, new_output] + [f"%{sn}" for sn in seq_names]
        vals_str = ", ".join(new_vals)
        self._emit(f"stablehlo.return {vals_str} : {types_str}")

        self._indent -= 1
        self._emit("}")
        self._indent -= 1

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

        # Initialize outside the while (so accessible inside regions)
        counter_var = self._fresh()
        self._emit(f"{counter_var} = stablehlo.constant dense<{seq_len - 1}> : {mlir_i32}")

        adj_carry_init = self._fresh()
        self._emit(f"{adj_carry_init} = stablehlo.constant dense<0.000000e+00> : {mlir_init}")

        adj_seq_inits = []
        for ms in mlir_seqs:
            v = self._fresh()
            self._emit(f"{v} = stablehlo.constant dense<0.000000e+00> : {ms}")
            adj_seq_inits.append(v)

        zero_var = self._fresh()
        self._emit(f"{zero_var} = stablehlo.constant dense<0> : {mlir_i32}")
        one_var = self._fresh()
        self._emit(f"{one_var} = stablehlo.constant dense<1> : {mlir_i32}")

        # Build while arg names and types (unique per while loop)
        # State: counter, adj_carry, adj_seq_0..N-1, fwd_carries, seq_0..N-1, adj_array, init_val, zero, one
        uid = self._counter
        gc_name = f"_gc{uid}"
        gac_name = f"_gac{uid}"
        gas_names = [f"_gas{uid}_{i}" for i in range(n_seqs)]
        gfwd_name = f"_gfwd{uid}"
        gos_names = [f"_gos{uid}_{i}" for i in range(n_seqs)]
        gadj_name = f"_gadj{uid}"
        ginit_name = f"_ginit{uid}"
        gzero_name = f"_gz{uid}"
        gone_name = f"_go{uid}"

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

        # Thread zero/one constants as while args to avoid capture issues
        arg_names.append(gzero_name)
        init_vals.append(zero_var)
        arg_types_list.append(mlir_i32)

        arg_names.append(gone_name)
        init_vals.append(one_var)
        arg_types_list.append(mlir_i32)

        n_args = len(arg_names)
        types_str = ", ".join(arg_types_list)

        # Use generic format for while to avoid parser issues with named args
        while_result = self._fresh()
        init_operands = ", ".join(init_vals)
        in_types = ", ".join(arg_types_list)
        out_types = ", ".join(arg_types_list)

        # Cond region needs DIFFERENT names from body region (MLIR requires unique SSA names)
        cond_names = [f"{n}c" for n in arg_names]
        gc_cond = cond_names[0]  # counter in cond
        gz_cond = cond_names[-2]  # zero in cond

        bb_cond = ", ".join(f"%{cond_names[i]}: {arg_types_list[i]}" for i in range(n_args))
        bb_body = ", ".join(f"%{arg_names[i]}: {arg_types_list[i]}" for i in range(n_args))

        z = f"%{gzero_name}"
        o = f"%{gone_name}"

        self._emit(f"{while_result}:{n_args} = \"stablehlo.while\"({init_operands}) ({{")
        self._indent += 1
        self._emit(f"^bb0({bb_cond}):")
        self._indent += 1

        # Condition: counter >= 0
        c_cmp = self._fresh()
        self._emit(f"{c_cmp} = stablehlo.compare GE, %{gc_cond}, %{gz_cond} : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        self._emit(f"stablehlo.return {c_cmp} : tensor<i1>")
        self._indent -= 1
        self._indent -= 1
        self._emit("}, {")
        self._indent += 1
        self._emit(f"^bb0({bb_body}):")
        self._indent += 1

        # Body
        # Slice adj_t from adj_array
        adj_t = self._slice_element(f"%{gadj_name}", f"%{gc_name}", adj_type, init_type)

        # adj_total = adj_carry + adj_t
        adj_total = self._fresh()
        self._emit(f"{adj_total} = stablehlo.add %{gac_name}, {adj_t} : {mlir_init}")

        # Compute prev_carry = select(counter > 0, fwd_carries[counter-1], init)
        prev_idx = self._fresh()
        self._emit(f"{prev_idx} = stablehlo.subtract %{gc_name}, {o} : {mlir_i32}")
        clamped_idx = self._fresh()
        self._emit(f"{clamped_idx} = stablehlo.clamp {z}, {prev_idx}, %{gc_name} : ({mlir_i32}, {mlir_i32}, {mlir_i32}) -> {mlir_i32}")

        fwd_prev = self._slice_element(f"%{gfwd_name}", clamped_idx, fwd_type, init_type)

        gt_zero = self._fresh()
        self._emit(f"{gt_zero} = stablehlo.compare GT, %{gc_name}, {z} : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")

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
        self._emit(f"{new_counter} = stablehlo.subtract %{gc_name}, {o} : {mlir_i32}")

        # Return new values (include zero/one passthrough)
        new_vals = [new_counter, new_adj_carry] + new_adj_seqs
        new_vals += [f"%{gfwd_name}"] + [f"%{gos_names[i]}" for i in range(n_seqs)]
        new_vals += [f"%{gadj_name}", f"%{ginit_name}", z, o]
        vals_str = ", ".join(new_vals)
        self._emit(f"stablehlo.return {vals_str} : {types_str}")

        self._indent -= 1
        self._indent -= 1
        self._emit(f"}}) : ({in_types}) -> ({out_types})")

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
