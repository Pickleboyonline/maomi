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

        # Broadcast if needed
        lhs = self._maybe_broadcast(lhs, lt, result_type)
        rhs = self._maybe_broadcast(rhs, rt, result_type)

        var = self._fresh()

        if op in _COMPARISON_MAP:
            cmp = _COMPARISON_MAP[op]
            # For comparisons, operands have the broadcast shape of their numeric type
            if isinstance(result_type, ArrayType):
                operand_type = ArrayType(_base_of_type(lt), result_type.dims)
            else:
                operand_type = lt if isinstance(lt, ScalarType) else rt
            mlir_operand = _mlir_type(operand_type)
            self._emit(f"{var} = stablehlo.compare GT, {lhs}, {rhs}, compare_type = FLOAT : {mlir_operand}")
            # Fix: use the actual comparison type
            self._lines[-1] = "  " * self._indent + f"{var} = stablehlo.compare {cmp}, {lhs}, {rhs}, compare_type = FLOAT : {mlir_operand}"
            return var

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
        self._emit(f"{var} = stablehlo.select {cond}, {then_val}, {else_val} : {mlir_result}")
        return var

    def _gen_call(self, expr: CallExpr, env: dict[str, str]) -> str:
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
            f"applies stablehlo.add across dimensions = [{dims_str}] "
            f": ({_mlir_type(arg_type)}, {mlir_scalar}) -> {mlir_scalar}"
        )
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

    # -- Scan codegen --

    def _gen_scan(self, expr: ScanExpr, env: dict[str, str]) -> str:
        init_val = self._gen_expr(expr.init, env)
        seq_val = self._gen_expr(expr.sequence, env)

        init_type = self._type_of(expr.init)
        seq_type = self._type_of(expr.sequence)
        result_type = self._type_of(expr)

        if not isinstance(seq_type, ArrayType):
            raise MaomiError("codegen: scan sequence must be array", "<codegen>", 0, 0)

        seq_len = seq_type.dims[0]
        if isinstance(seq_len, str):
            raise MaomiError(f"codegen: scan requires concrete sequence length, got '{seq_len}'", "<codegen>", 0, 0)

        # Element type
        if len(seq_type.dims) == 1:
            elem_type: MaomiType = ScalarType(seq_type.base)
        else:
            elem_type = ArrayType(seq_type.base, seq_type.dims[1:])

        mlir_init = _mlir_type(init_type)
        mlir_result = _mlir_type(result_type)
        mlir_seq = _mlir_type(seq_type)
        mlir_i32 = "tensor<i32>"

        # Create counter = 0
        counter_var = self._fresh()
        self._emit(f"{counter_var} = stablehlo.constant dense<0> : {mlir_i32}")

        # Create output buffer = zeros
        output_var = self._fresh()
        self._emit(f"{output_var} = stablehlo.constant dense<0.000000e+00> : {mlir_result}")

        # While loop: state = (counter, carry, output)
        # We emit this as a sequence of ops that represents the unrolled loop
        # For v0.2, we use stablehlo.while
        state_types = f"{mlir_i32}, {mlir_init}, {mlir_result}, {mlir_seq}"

        # Emit the while op
        limit_var = self._fresh()
        self._emit(f"{limit_var} = stablehlo.constant dense<{seq_len}> : {mlir_i32}")

        result_var = self._fresh()
        one_var = self._fresh()
        self._emit(f"{one_var} = stablehlo.constant dense<1> : {mlir_i32}")

        self._emit(f"// scan loop over {seq_len} steps")
        # For simplicity in v0.2, we'll unroll short sequences or emit a while structure
        # Emit stablehlo.while with tuple state

        tuple_type = f"tuple<{state_types}>"

        # Create initial tuple
        init_tuple = self._fresh()
        self._emit(
            f"{init_tuple} = stablehlo.tuple {counter_var}, {init_val}, {output_var}, {seq_val} "
            f": {tuple_type}"
        )

        # While op
        while_result = self._fresh()
        self._emit(f"{while_result} = stablehlo.while({init_tuple}) : {tuple_type}")

        # Condition region
        self._indent += 1
        self._emit(f"cond {{")
        self._indent += 1
        cond_arg = self._fresh()
        self._emit(f"^bb0({cond_arg}: {tuple_type}):")

        c_counter = self._fresh()
        self._emit(f"{c_counter} = stablehlo.get_tuple_element {cond_arg}[0] : ({tuple_type}) -> {mlir_i32}")
        c_limit = self._fresh()
        self._emit(f"{c_limit} = stablehlo.constant dense<{seq_len}> : {mlir_i32}")
        c_cmp = self._fresh()
        self._emit(f"{c_cmp} = stablehlo.compare LT, {c_counter}, {c_limit}, compare_type = SIGNED : {mlir_i32}")
        self._emit(f"stablehlo.return {c_cmp} : tensor<i1>")
        self._indent -= 1
        self._emit("}")

        # Body region
        self._emit(f"body {{")
        self._indent += 1
        body_arg = self._fresh()
        self._emit(f"^bb0({body_arg}: {tuple_type}):")

        # Extract state
        b_counter = self._fresh()
        self._emit(f"{b_counter} = stablehlo.get_tuple_element {body_arg}[0] : ({tuple_type}) -> {mlir_i32}")
        b_carry = self._fresh()
        self._emit(f"{b_carry} = stablehlo.get_tuple_element {body_arg}[1] : ({tuple_type}) -> {mlir_init}")
        b_output = self._fresh()
        self._emit(f"{b_output} = stablehlo.get_tuple_element {body_arg}[2] : ({tuple_type}) -> {mlir_result}")
        b_seq = self._fresh()
        self._emit(f"{b_seq} = stablehlo.get_tuple_element {body_arg}[3] : ({tuple_type}) -> {mlir_seq}")

        # Slice element from sequence
        b_elem = self._fresh()
        mlir_elem = _mlir_type(elem_type)
        self._emit(
            f"{b_elem} = stablehlo.dynamic_slice {b_seq}, {b_counter} "
            f": ({mlir_seq}, {mlir_i32}) -> {mlir_elem}"
        )

        # Generate body expression
        body_env = dict(env)
        body_env[expr.carry_var] = b_carry
        body_env[expr.elem_var] = b_elem
        new_carry = self._gen_block(expr.body, body_env)

        # Update output: dynamic_update_slice
        new_output = self._fresh()
        self._emit(
            f"{new_output} = stablehlo.dynamic_update_slice {b_output}, {new_carry}, {b_counter} "
            f": ({mlir_result}, {mlir_init}, {mlir_i32}) -> {mlir_result}"
        )

        # Increment counter
        b_one = self._fresh()
        self._emit(f"{b_one} = stablehlo.constant dense<1> : {mlir_i32}")
        new_counter = self._fresh()
        self._emit(f"{new_counter} = stablehlo.add {b_counter}, {b_one} : {mlir_i32}")

        # Return new tuple
        new_tuple = self._fresh()
        self._emit(
            f"{new_tuple} = stablehlo.tuple {new_counter}, {new_carry}, {new_output}, {b_seq} "
            f": {tuple_type}"
        )
        self._emit(f"stablehlo.return {new_tuple} : {tuple_type}")

        self._indent -= 1
        self._emit("}")
        self._indent -= 1

        # Extract result (output buffer) from while result
        final_output = self._fresh()
        self._emit(f"{final_output} = stablehlo.get_tuple_element {while_result}[2] : ({tuple_type}) -> {mlir_result}")
        return final_output

    # -- Map codegen --

    def _gen_map(self, expr: MapExpr, env: dict[str, str]) -> str:
        seq_val = self._gen_expr(expr.sequence, env)
        seq_type = self._type_of(expr.sequence)

        # For trivially batchable bodies, just bind elem_var to the full sequence
        # and generate the body — elementwise ops naturally work on batched shapes
        body_env = dict(env)
        body_env[expr.elem_var] = seq_val
        return self._gen_block(expr.body, body_env)

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
