from __future__ import annotations

from ...ast_nodes import (
    Program,
    FnDef,
    Block,
    LetStmt,
    ExprStmt,
    IntLiteral,
    FloatLiteral,
    BoolLiteral,
    StringLiteral,
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
    ArrayLiteral,
    StructLiteral,
    FieldAccess,
    WithExpr,
    IndexExpr,
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
from ...types import MaomiType, ScalarType, ArrayType, StructType, FLOAT_BASES
from ...errors import MaomiError
from .utils import (
    _mlir_type,
    _MLIR_ETYPE,
    _COMPARISON_MAP,
    _BUILTIN_OPS,
    _callback_layout,
    _base_of_type,
    _types_equal,
)
from ...builtins import ELEMENTWISE as _EW_REGISTRY, COMPLEX as _CX_REGISTRY
from .loops import LoopCodegenMixin
from .conv import ConvCodegenMixin
from .map_codegen import MapCodegenMixin
from .indexing import IndexingCodegenMixin
from .rng import RNGCodegenMixin


class StableHLOCodegen(LoopCodegenMixin, ConvCodegenMixin, MapCodegenMixin,
                        IndexingCodegenMixin, RNGCodegenMixin):

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
        self._callback_labels: dict[int, list[str]] = {}

    def generate(self) -> str:
        self._emit("module {")
        self._indent += 1
        for fn in self.program.functions:
            # Skip generic functions (wildcard or symbolic dims) — only monomorphized copies are generated
            if self._is_generic_fn(fn):
                continue
            self._gen_function(fn)
        self._indent -= 1
        self._emit("}")
        return "\n".join(self._lines)

    @staticmethod
    def _is_typevar_annotation(ta) -> bool:
        """True if type annotation is a single uppercase letter (type variable)."""
        return (len(ta.base) == 1 and ta.base.isupper()
                and ta.base not in ('I',)  # not a base type prefix
                and ta.dims is None and not getattr(ta, 'wildcard', False))

    _CONCRETE_BASES = FLOAT_BASES | {'i32', 'i64', 'bool', 'Key'}

    def _is_generic_fn(self, fn) -> bool:
        """True if function has wildcard, symbolic-dim, or type-variable annotations."""
        struct_names = {sd.name for sd in self.program.struct_defs} if self.program else set()
        for p in fn.params:
            ta = p.type_annotation
            if getattr(ta, 'wildcard', False):
                return True
            if ta.dims is not None and any(isinstance(d.value, str) for d in ta.dims):
                return True
            if (ta.dims is None and not getattr(ta, 'wildcard', False)
                    and ta.base not in self._CONCRETE_BASES
                    and ta.base not in struct_names):
                return True
        rta = fn.return_type
        if getattr(rta, 'wildcard', False):
            return True
        if rta.dims is not None and any(isinstance(d.value, str) for d in rta.dims):
            return True
        if (rta.dims is None and not getattr(rta, 'wildcard', False)
                and rta.base not in self._CONCRETE_BASES
                and rta.base not in struct_names):
            return True
        return False

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
        if getattr(ta, 'wildcard', False):
            raise RuntimeError(f"unresolved wildcard type {ta.base}[..] reached codegen — monomorphization bug")
        if ta.base in ("f32", "f64", "bf16", "i32", "i64", "bool"):
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
            case ArrayLiteral():
                return self._gen_array_literal(expr, env)
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
        if op == "-" and isinstance(result_type, StructType):
            return self._gen_struct_negate(val, result_type)
        mlir_t = _mlir_type(result_type)
        var = self._fresh()
        if op == "-":
            self._emit(f"{var} = stablehlo.negate {val} : {mlir_t}")
        return var

    def _gen_struct_negate(self, val: str, stype: 'StructType') -> str:
        """Negate each field of a struct."""
        field_ssas = []
        for i, (fname, ftype) in enumerate(stype.fields):
            mlir_ft = _mlir_type(ftype)
            extracted = self._fresh()
            self._emit(f"{extracted} = stablehlo.get_tuple_element {val}[{i}] : ({_mlir_type(stype)}) -> {mlir_ft}")
            if isinstance(ftype, StructType):
                field_ssas.append(self._gen_struct_negate(extracted, ftype))
            else:
                negated = self._fresh()
                self._emit(f"{negated} = stablehlo.negate {extracted} : {mlir_ft}")
                field_ssas.append(negated)
        result = self._fresh()
        types_str = ", ".join(_mlir_type(ft) for _, ft in stype.fields)
        self._emit(f"{result} = stablehlo.tuple {', '.join(field_ssas)} : tuple<{types_str}>")
        return result

    def _gen_binop(self, op: str, left: Expr, right: Expr, expr: Expr, env: dict[str, str]) -> str:
        if op == "@":
            return self._gen_matmul(left, right, expr, env)

        lhs = self._gen_expr(left, env)
        rhs = self._gen_expr(right, env)

        lt = self._type_of(left)
        rt = self._type_of(right)
        result_type = self._type_of(expr)

        # Struct arithmetic — field-by-field
        if isinstance(result_type, StructType):
            return self._gen_struct_binop(op, lhs, rhs, lt, rt, result_type)

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

    def _gen_struct_binop(self, op: str, lhs: str, rhs: str,
                          lt: 'MaomiType', rt: 'MaomiType',
                          result_type: 'StructType') -> str:
        """Generate field-by-field arithmetic on struct types."""
        from ...types import StructType as ST, ScalarType as ScT

        op_map = {"+": "stablehlo.add", "-": "stablehlo.subtract",
                  "*": "stablehlo.multiply", "/": "stablehlo.divide",
                  "**": "stablehlo.power"}
        stablehlo_op = op_map[op]

        field_ssas = []
        for i, (fname, ftype) in enumerate(result_type.fields):
            mlir_ft = _mlir_type(ftype)
            # Extract fields from struct operands
            if isinstance(lt, ST):
                lf = self._fresh()
                self._emit(f"{lf} = stablehlo.get_tuple_element {lhs}[{i}] : ({_mlir_type(lt)}) -> {mlir_ft}")
            else:
                lf = lhs  # scalar — will be broadcast per field
            if isinstance(rt, ST):
                rf = self._fresh()
                self._emit(f"{rf} = stablehlo.get_tuple_element {rhs}[{i}] : ({_mlir_type(rt)}) -> {mlir_ft}")
            else:
                rf = rhs  # scalar

            if isinstance(ftype, ST):
                # Recurse for nested struct fields
                inner_lt = dict(lt.fields)[fname] if isinstance(lt, ST) else lt
                inner_rt = dict(rt.fields)[fname] if isinstance(rt, ST) else rt
                field_ssas.append(self._gen_struct_binop(op, lf, rf, inner_lt, inner_rt, ftype))
            else:
                # Leaf numeric field — broadcast scalar if needed
                if not isinstance(lt, ST):
                    lf = self._maybe_broadcast(lf, lt, ftype)
                if not isinstance(rt, ST):
                    rf = self._maybe_broadcast(rf, rt, ftype)
                var = self._fresh()
                self._emit(f"{var} = {stablehlo_op} {lf}, {rf} : {mlir_ft}")
                field_ssas.append(var)

        # Reconstruct the tuple
        mlir_result = _mlir_type(result_type)
        result_var = self._fresh()
        fields_str = ", ".join(field_ssas)
        types_str = ", ".join(_mlir_type(ft) for _, ft in result_type.fields)
        self._emit(f"{result_var} = stablehlo.tuple {fields_str} : tuple<{types_str}>")
        return result_var

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

    def _gen_clip(self, expr: CallExpr, env: dict[str, str]) -> str:
        x = self._gen_expr(expr.args[0], env)
        lo = self._gen_expr(expr.args[1], env)
        hi = self._gen_expr(expr.args[2], env)
        result_type = self._type_of(expr)
        # Broadcast all to result type
        x = self._maybe_broadcast(x, self._type_of(expr.args[0]), result_type)
        lo = self._maybe_broadcast(lo, self._type_of(expr.args[1]), result_type)
        hi = self._maybe_broadcast(hi, self._type_of(expr.args[2]), result_type)
        mlir_t = _mlir_type(result_type)
        var = self._fresh()
        # stablehlo.clamp takes (min, operand, max) order
        self._emit(f"{var} = stablehlo.clamp {lo}, {x}, {hi} : {mlir_t}")
        return var

    _CALLBACK_BUILTINS = {"callback"}
    _RNG_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "rng"}

    def _gen_callback(self, expr: CallExpr, env: dict[str, str]) -> str:
        """Emit stablehlo.custom_call targeting JAX's FFI callback handler."""
        # Separate string labels from tensor args
        tensor_exprs = []
        labels = []
        for a in expr.args:
            if isinstance(a, StringLiteral):
                labels.append(a.value)
            else:
                tensor_exprs.append(a)

        arg_ssas = [self._gen_expr(a, env) for a in tensor_exprs]
        arg_types = [self._type_of(a) for a in tensor_exprs]

        idx = self._callback_count
        self._callback_count += 1
        if labels:
            self._callback_labels[idx] = labels

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

        # one_hot(index, n) → iota + broadcast + compare + convert
        if expr.callee == "one_hot":
            return self._gen_one_hot(expr, env)

        # zeros/ones/full → constant + broadcast
        if expr.callee in ("zeros", "ones", "full"):
            return self._gen_fill(expr, env)

        # RNG builtins
        if expr.callee in self._RNG_BUILTINS:
            return self._gen_rng(expr, env)

        # Handle builtins — elementwise (single op or compound)
        if expr.callee in _BUILTIN_OPS:
            return self._gen_elementwise_builtin(expr, env)
        if expr.callee in _EW_REGISTRY and _EW_REGISTRY[expr.callee].codegen_fn:
            return _EW_REGISTRY[expr.callee].codegen_fn(self, expr, env)
        if expr.callee == "mean":
            return self._gen_mean(expr, env)
        if expr.callee == "sum":
            return self._gen_sum(expr, env)
        if expr.callee in ("max", "min"):
            return self._gen_max_min(expr, env)
        if expr.callee == "logsumexp":
            return self._gen_logsumexp(expr, env)
        if expr.callee in ("argmax", "argmin"):
            return self._gen_argmax(expr, env)
        if expr.callee == "transpose":
            return self._gen_transpose(expr, env)
        if expr.callee == "reshape":
            return self._gen_reshape(expr, env)
        if expr.callee == "concat":
            return self._gen_concat(expr, env)
        if expr.callee == "stack":
            return self._gen_stack(expr, env)
        if expr.callee == "pad":
            return self._gen_pad(expr, env)
        if expr.callee == "expand_dims":
            return self._gen_expand_dims(expr, env)
        if expr.callee == "squeeze":
            return self._gen_squeeze(expr, env)
        if expr.callee == "broadcast_to":
            return self._gen_broadcast_to(expr, env)
        if expr.callee == "stop_gradient":
            return self._gen_expr(expr.args[0], env)
        if expr.callee == "isfinite":
            return self._gen_isfinite(expr, env)
        if expr.callee in ("zeros_like", "ones_like"):
            return self._gen_like(expr, env)
        if expr.callee == "where":
            return self._gen_where(expr, env)
        if expr.callee == "clip":
            return self._gen_clip(expr, env)
        if expr.callee == "conv2d":
            return self._gen_conv2d(expr, env)
        if expr.callee == "max_pool":
            return self._gen_max_pool(expr, env)
        if expr.callee == "avg_pool":
            return self._gen_avg_pool(expr, env)
        if expr.callee in ("maximum", "minimum", "pow"):
            return self._gen_two_arg_elementwise(expr, env)

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

    def _gen_elementwise_builtin(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        result_type = self._type_of(expr)
        op = _BUILTIN_OPS[expr.callee]
        if isinstance(result_type, StructType):
            return self._gen_struct_elementwise(op, arg, result_type)
        mlir_t = _mlir_type(result_type)
        var = self._fresh()
        self._emit(f"{var} = {op} {arg} : {mlir_t}")
        return var

    _TWO_ARG_EW_OPS = {
        "maximum": "stablehlo.maximum",
        "minimum": "stablehlo.minimum",
        "pow": "stablehlo.power",
    }

    def _gen_two_arg_elementwise(self, expr: CallExpr, env: dict[str, str]) -> str:
        """Emit stablehlo for two-arg elementwise ops: maximum, minimum, pow."""
        op = self._TWO_ARG_EW_OPS[expr.callee]
        x = self._gen_expr(expr.args[0], env)
        y = self._gen_expr(expr.args[1], env)
        x_type = self._type_of(expr.args[0])
        y_type = self._type_of(expr.args[1])
        result_type = self._type_of(expr)
        # Broadcast if needed
        x = self._maybe_broadcast(x, x_type, result_type)
        y = self._maybe_broadcast(y, y_type, result_type)
        mlir_t = _mlir_type(result_type)
        var = self._fresh()
        self._emit(f"{var} = {op} {x}, {y} : {mlir_t}")
        return var

    def _gen_struct_elementwise(self, op: str, arg_ssa: str, stype: StructType) -> str:
        """Apply an elementwise builtin to each field of a struct."""
        field_ssas = []
        for i, (fname, ftype) in enumerate(stype.fields):
            mlir_ft = _mlir_type(ftype)
            extracted = self._fresh()
            self._emit(f"{extracted} = stablehlo.get_tuple_element {arg_ssa}[{i}] : ({_mlir_type(stype)}) -> {mlir_ft}")
            if isinstance(ftype, StructType):
                field_ssas.append(self._gen_struct_elementwise(op, extracted, ftype))
            else:
                result = self._fresh()
                self._emit(f"{result} = {op} {extracted} : {mlir_ft}")
                field_ssas.append(result)
        var = self._fresh()
        types_str = ", ".join(_mlir_type(ft) for _, ft in stype.fields)
        self._emit(f"{var} = stablehlo.tuple {', '.join(field_ssas)} : tuple<{types_str}>")
        return var

    def _gen_struct_compound_elementwise(self, name: str, inner_fn, arg_ssa: str, stype: StructType) -> str:
        """Apply a compound elementwise builtin to each field of a struct.

        inner_fn: (codegen, arg_ssa, mlir_type_str) -> result_ssa
        """
        field_ssas = []
        for i, (fname, ftype) in enumerate(stype.fields):
            mlir_ft = _mlir_type(ftype)
            extracted = self._fresh()
            self._emit(f"{extracted} = stablehlo.get_tuple_element {arg_ssa}[{i}] : ({_mlir_type(stype)}) -> {mlir_ft}")
            if isinstance(ftype, StructType):
                field_ssas.append(self._gen_struct_compound_elementwise(name, inner_fn, extracted, ftype))
            else:
                field_ssas.append(inner_fn(self, extracted, mlir_ft))
        var = self._fresh()
        types_str = ", ".join(_mlir_type(ft) for _, ft in stype.fields)
        self._emit(f"{var} = stablehlo.tuple {', '.join(field_ssas)} : tuple<{types_str}>")
        return var

    def _gen_iota(self, expr: CallExpr, env: dict[str, str]) -> str:
        result_type = self._type_of(expr)
        mlir_t = _mlir_type(result_type)
        var = self._fresh()
        self._emit(f"{var} = stablehlo.iota dim = 0 : {mlir_t}")
        return var

    def _gen_one_hot(self, expr: CallExpr, env: dict[str, str]) -> str:
        index = self._gen_expr(expr.args[0], env)
        index_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)
        output_dims = result_type.dims
        ndim_in = len(index_type.dims) if isinstance(index_type, ArrayType) else 0

        # Step 1: iota along the last (new) dimension
        iota_i32_type = ArrayType("i32", output_dims)
        iota_mlir = _mlir_type(iota_i32_type)
        iota_var = self._fresh()
        self._emit(f"{iota_var} = stablehlo.iota dim = {ndim_in} : {iota_mlir}")

        # Step 2: broadcast index to output shape
        dims_str = ", ".join(str(d) for d in range(ndim_in))
        idx_broadcast = self._fresh()
        self._emit(f"{idx_broadcast} = stablehlo.broadcast_in_dim {index}, dims = [{dims_str}] : ({_mlir_type(index_type)}) -> {iota_mlir}")

        # Step 3: compare EQ → bool
        bool_type = ArrayType("bool", output_dims)
        cmp = self._fresh()
        self._emit(f"{cmp} = stablehlo.compare {idx_broadcast}, {iota_var}, EQ : ({iota_mlir}, {iota_mlir}) -> {_mlir_type(bool_type)}")

        # Step 4: convert bool → f32
        result = self._fresh()
        self._emit(f"{result} = stablehlo.convert {cmp} : ({_mlir_type(bool_type)}) -> {_mlir_type(result_type)}")
        return result

    def _gen_fill(self, expr: CallExpr, env: dict[str, str]) -> str:
        result_type = self._type_of(expr)
        mlir_t = _mlir_type(result_type)
        if expr.callee == "zeros":
            scalar = self._fresh()
            self._emit(f"{scalar} = stablehlo.constant dense<0.000000e+00> : tensor<f32>")
        elif expr.callee == "ones":
            scalar = self._fresh()
            self._emit(f"{scalar} = stablehlo.constant dense<1.000000e+00> : tensor<f32>")
        else:  # full
            scalar = self._gen_expr(expr.args[0], env)
        result = self._fresh()
        self._emit(f"{result} = stablehlo.broadcast_in_dim {scalar}, dims = [] : (tensor<f32>) -> {mlir_t}")
        return result

    def _gen_isfinite(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        mlir_arg = _mlir_type(arg_type)
        var = self._fresh()
        self._emit(f"{var} = stablehlo.is_finite {arg} : {mlir_arg}")
        return var

    def _gen_like(self, expr: CallExpr, env: dict[str, str]) -> str:
        # Generate the input arg for side effects (ensure on tape), but we only need its type
        self._gen_expr(expr.args[0], env)
        result_type = self._type_of(expr)
        value = "0.000000e+00" if expr.callee == "zeros_like" else "1.000000e+00"

        if isinstance(result_type, ScalarType):
            var = self._fresh()
            self._emit(f"{var} = stablehlo.constant dense<{value}> : {_mlir_type(result_type)}")
            return var

        # Array: constant scalar + broadcast
        scalar_type = ScalarType(result_type.base)
        scalar = self._fresh()
        self._emit(f"{scalar} = stablehlo.constant dense<{value}> : {_mlir_type(scalar_type)}")
        var = self._fresh()
        self._emit(f"{var} = stablehlo.broadcast_in_dim {scalar}, dims = [] : ({_mlir_type(scalar_type)}) -> {_mlir_type(result_type)}")
        return var

    def _gen_mean(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)

        if not isinstance(arg_type, ArrayType):
            return arg  # mean of scalar is itself

        keepdims = self._has_keepdims(expr)

        # Axis-specific or all-dims reduction
        if len(expr.args) >= 2:
            axis = expr.args[1].value
            bd = self._batch_depth
            actual_axis = bd + axis
            # For keepdims, reduce to intermediate type first
            if keepdims:
                reduced_dims = tuple(d for i, d in enumerate(arg_type.dims) if i != axis)
                reduced_type = ArrayType(arg_type.base, reduced_dims) if reduced_dims else ScalarType(arg_type.base)
            else:
                reduced_type = result_type
            # Sum along specific axis
            sum_var = self._gen_reduce_sum_single_axis(arg, arg_type, reduced_type, actual_axis)
            # Divide by axis size
            axis_size = arg_type.dims[actual_axis]
            if isinstance(axis_size, str):
                raise MaomiError(f"codegen: cannot compute mean with symbolic dim '{axis_size}'", "<codegen>", 0, 0)
            count_scalar = self._fresh()
            scalar_mlir = _mlir_type(ScalarType(arg_type.base))
            self._emit(f"{count_scalar} = stablehlo.constant dense<{float(axis_size):e}> : {scalar_mlir}")
            count_var = self._maybe_broadcast(count_scalar, ScalarType(arg_type.base), reduced_type)
            mlir_reduced = _mlir_type(reduced_type)
            var = self._fresh()
            self._emit(f"{var} = stablehlo.divide {sum_var}, {count_var} : {mlir_reduced}")
            if keepdims:
                return self._keepdims_reshape(var, arg_type, axis, result_type)
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

    def _has_keepdims(self, expr: CallExpr) -> bool:
        return (len(expr.args) == 3
                and isinstance(expr.args[2], BoolLiteral)
                and expr.args[2].value)

    def _keepdims_reshape(self, reduced: str, arg_type: ArrayType, axis: int, result_type) -> str:
        """Reshape reduced result to insert size-1 dim back for keepdims."""
        # Build the reduced type (without keepdims)
        reduced_dims = tuple(d for i, d in enumerate(arg_type.dims) if i != axis)
        if len(reduced_dims) == 0:
            reduced_type = ScalarType(arg_type.base)
        else:
            reduced_type = ArrayType(arg_type.base, reduced_dims)
        var = self._fresh()
        self._emit(f"{var} = stablehlo.reshape {reduced} : ({_mlir_type(reduced_type)}) -> {_mlir_type(result_type)}")
        return var

    def _gen_sum(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)

        if not isinstance(arg_type, ArrayType):
            return arg

        keepdims = self._has_keepdims(expr)

        if len(expr.args) >= 2:
            axis = expr.args[1].value
            bd = self._batch_depth
            actual_axis = bd + axis
            if keepdims:
                # Compute reduced type (without the keepdims dim)
                reduced_dims = tuple(d for i, d in enumerate(arg_type.dims) if i != axis)
                reduced_type = ArrayType(arg_type.base, reduced_dims) if reduced_dims else ScalarType(arg_type.base)
                reduced = self._gen_reduce_sum_single_axis(arg, arg_type, reduced_type, actual_axis)
                return self._keepdims_reshape(reduced, arg_type, axis, result_type)
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
        keepdims = self._has_keepdims(expr)

        if len(expr.args) >= 2:
            axis = expr.args[1].value
            bd = self._batch_depth
            actual_axis = bd + axis
            if keepdims:
                reduced_dims = tuple(d for i, d in enumerate(arg_type.dims) if i != axis)
                reduced_type = ArrayType(arg_type.base, reduced_dims) if reduced_dims else ScalarType(arg_type.base)
                reduced = self._gen_reduce_max_min_single_axis(arg, arg_type, reduced_type, actual_axis, is_max)
                return self._keepdims_reshape(reduced, arg_type, axis, result_type)
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
        if base in FLOAT_BASES:
            if base == "f64":
                return "0xFFF0000000000000" if is_max else "0x7FF0000000000000"
            if base == "bf16":
                return "0xFF80" if is_max else "0x7F80"  # bf16 -inf / +inf
            return "0xFF800000" if is_max else "0x7F800000"  # f32 -inf / +inf
        elif base == "i32":
            return "-2147483647" if is_max else "2147483647"
        elif base == "i64":
            return "-9223372036854775807" if is_max else "9223372036854775807"
        return "0xFF800000" if is_max else "0x7F800000"

    def _gen_logsumexp(self, expr: CallExpr, env: dict[str, str]) -> str:
        """Generate numerically stable logsumexp: m + log(sum(exp(x - m), axis))
        where m = max(x, axis, keepdims=true)."""
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)

        if not isinstance(arg_type, ArrayType):
            return arg  # logsumexp of scalar is itself

        keepdims = self._has_keepdims(expr)
        has_axis = len(expr.args) >= 2
        bd = self._batch_depth
        mlir_arg = _mlir_type(arg_type)

        if has_axis:
            axis = expr.args[1].value
            actual_axis = bd + axis

            # Step 1: m = max(x, axis, keepdims=true)
            # Compute reduced type (without keepdims dim) for the max reduction
            reduced_dims = tuple(d for i, d in enumerate(arg_type.dims) if i != axis)
            reduced_type = ArrayType(arg_type.base, reduced_dims) if reduced_dims else ScalarType(arg_type.base)
            m_reduced = self._gen_reduce_max_min_single_axis(arg, arg_type, reduced_type, actual_axis, True)

            # Reshape m to keepdims shape for broadcasting
            keepdims_dims = tuple(1 if i == axis else d for i, d in enumerate(arg_type.dims))
            keepdims_type = ArrayType(arg_type.base, keepdims_dims)
            m_keepdims = self._fresh()
            self._emit(f"{m_keepdims} = stablehlo.reshape {m_reduced} : ({_mlir_type(reduced_type)}) -> {_mlir_type(keepdims_type)}")

            # Step 2: Broadcast m to input shape and subtract: shifted = x - m
            m_broadcast = self._fresh()
            self._emit(f"{m_broadcast} = stablehlo.broadcast_in_dim {m_keepdims}, dims = [{', '.join(str(i) for i in range(len(arg_type.dims)))}] : ({_mlir_type(keepdims_type)}) -> {mlir_arg}")
            shifted = self._fresh()
            self._emit(f"{shifted} = stablehlo.subtract {arg}, {m_broadcast} : {mlir_arg}")

            # Step 3: exp(shifted)
            exp_shifted = self._fresh()
            self._emit(f"{exp_shifted} = stablehlo.exponential {shifted} : {mlir_arg}")

            # Step 4: sum(exp_shifted, axis)
            sum_exp = self._gen_reduce_sum_single_axis(exp_shifted, arg_type, reduced_type, actual_axis)

            # Step 5: log(sum_exp)
            mlir_reduced = _mlir_type(reduced_type)
            log_sum = self._fresh()
            self._emit(f"{log_sum} = stablehlo.log {sum_exp} : {mlir_reduced}")

            # Step 6: result = log_sum + m_reduced
            result = self._fresh()
            self._emit(f"{result} = stablehlo.add {log_sum}, {m_reduced} : {mlir_reduced}")

            if keepdims:
                return self._keepdims_reshape(result, arg_type, axis, result_type)
            return result
        else:
            # All-dims reduction
            # Step 1: m = max(x) — scalar
            scalar_type = ScalarType(arg_type.base)
            m_scalar = self._gen_reduce_max_min(arg, arg_type, scalar_type, True)

            # Step 2: Broadcast m to input shape and subtract
            m_broadcast = self._fresh()
            self._emit(f"{m_broadcast} = stablehlo.broadcast_in_dim {m_scalar}, dims = [] : ({_mlir_type(scalar_type)}) -> {mlir_arg}")
            shifted = self._fresh()
            self._emit(f"{shifted} = stablehlo.subtract {arg}, {m_broadcast} : {mlir_arg}")

            # Step 3: exp(shifted)
            exp_shifted = self._fresh()
            self._emit(f"{exp_shifted} = stablehlo.exponential {shifted} : {mlir_arg}")

            # Step 4: sum(exp_shifted) — all dims
            sum_exp = self._gen_reduce_sum(exp_shifted, arg_type, result_type)

            # Step 5: log(sum_exp)
            mlir_result = _mlir_type(result_type)
            log_sum = self._fresh()
            self._emit(f"{log_sum} = stablehlo.log {sum_exp} : {mlir_result}")

            # Step 6: result = log_sum + m
            result = self._fresh()
            self._emit(f"{result} = stablehlo.add {log_sum}, {m_scalar} : {mlir_result}")
            return result

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
        cmp_kind = "FLOAT" if arg_type.base in FLOAT_BASES else "SIGNED"

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

        if not isinstance(arg_type, ArrayType):
            raise MaomiError("codegen: transpose requires an array", "<codegen>", expr.span.line_start, expr.span.col_start)

        bd = self._batch_depth
        if len(expr.args) == 1:
            # Shorthand: swap last two dims
            perm = list(range(bd)) + [bd + 1, bd]
        else:
            # General: axes from args, shifted by batch depth
            perm = list(range(bd)) + [bd + a.value for a in expr.args[1:]]

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

    # expand_dims and squeeze both emit stablehlo.reshape, same as _gen_reshape
    _gen_expand_dims = _gen_reshape
    _gen_squeeze = _gen_reshape

    def _gen_broadcast_to(self, expr: CallExpr, env: dict[str, str]) -> str:
        arg = self._gen_expr(expr.args[0], env)
        arg_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)
        mlir_result = _mlir_type(result_type)

        var = self._fresh()
        if isinstance(arg_type, ScalarType):
            # Scalar → array: no broadcast_dimensions
            self._emit(
                f"{var} = stablehlo.broadcast_in_dim {arg}, "
                f"dims = [] : ({_mlir_type(arg_type)}) -> {mlir_result}"
            )
        else:
            # Array → array: compute broadcast_dimensions
            assert isinstance(arg_type, ArrayType) and isinstance(result_type, ArrayType)
            src_rank = len(arg_type.dims)
            dst_rank = len(result_type.dims)
            # Right-align: input dim i maps to output dim (offset + i)
            offset = dst_rank - src_rank
            broadcast_dims = list(range(offset, dst_rank))
            dims_str = ", ".join(str(d) for d in broadcast_dims)
            self._emit(
                f"{var} = stablehlo.broadcast_in_dim {arg}, "
                f"dims = [{dims_str}] : ({_mlir_type(arg_type)}) -> {mlir_result}"
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

    def _gen_stack(self, expr: CallExpr, env: dict[str, str]) -> str:
        axis = expr.args[-1].value  # last arg is axis
        array_args = expr.args[:-1]  # all but last
        result_type = self._type_of(expr)

        # Reshape each array to insert size-1 dim at axis, then concatenate
        reshaped = []
        reshaped_types = []
        for arr_expr in array_args:
            arr = self._gen_expr(arr_expr, env)
            arr_type = self._type_of(arr_expr)
            assert isinstance(arr_type, ArrayType)
            # Insert size-1 dim at axis
            new_dims = list(arr_type.dims)
            new_dims.insert(axis, 1)
            new_type = ArrayType(arr_type.base, tuple(new_dims))
            mlir_new = _mlir_type(new_type)
            reshaped_var = self._fresh()
            self._emit(
                f"{reshaped_var} = stablehlo.reshape {arr} "
                f": ({_mlir_type(arr_type)}) -> {mlir_new}"
            )
            reshaped.append(reshaped_var)
            reshaped_types.append(new_type)

        # Concatenate along axis
        args_str = ", ".join(reshaped)
        types_str = ", ".join(_mlir_type(t) for t in reshaped_types)
        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.concatenate {args_str}, dim = {axis} "
            f": ({types_str}) -> {_mlir_type(result_type)}"
        )
        return var

    def _gen_pad(self, expr: CallExpr, env: dict[str, str]) -> str:
        x = self._gen_expr(expr.args[0], env)
        val = self._gen_expr(expr.args[1], env)
        pad_lo = expr.args[2].value
        pad_hi = expr.args[3].value

        x_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)
        assert isinstance(x_type, ArrayType)
        ndims = len(x_type.dims)

        lo_str = ", ".join([str(pad_lo)] * ndims)
        hi_str = ", ".join([str(pad_hi)] * ndims)
        interior_str = ", ".join(["0"] * ndims)

        mlir_x = _mlir_type(x_type)
        mlir_scalar = _mlir_type(ScalarType(x_type.base))
        mlir_result = _mlir_type(result_type)
        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.pad {x}, {val}, "
            f"low = [{lo_str}], high = [{hi_str}], interior = [{interior_str}] "
            f": ({mlir_x}, {mlir_scalar}) -> {mlir_result}"
        )
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

    def _gen_array_literal(self, expr: ArrayLiteral, env: dict[str, str]) -> str:
        result_type = self._type_of(expr)
        elem_ssas = [self._gen_expr(e, env) for e in expr.elements]
        elem_types = [self._type_of(e) for e in expr.elements]

        # Reshape each element to add a leading dim of 1
        reshaped = []
        for ssa, et in zip(elem_ssas, elem_types):
            if isinstance(et, ScalarType):
                update_type = ArrayType(et.base, (1,))
            elif isinstance(et, ArrayType):
                update_type = ArrayType(et.base, (1,) + et.dims)
            else:
                raise MaomiError("codegen: array literal element must be scalar or array", "<codegen>", 0, 0)
            mlir_from = _mlir_type(et)
            mlir_to = _mlir_type(update_type)
            v = self._fresh()
            self._emit(f"{v} = stablehlo.reshape {ssa} : ({mlir_from}) -> {mlir_to}")
            reshaped.append((v, update_type))

        # Concatenate along dim 0
        args_str = ", ".join(v for v, _ in reshaped)
        types_str = ", ".join(_mlir_type(t) for _, t in reshaped)
        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.concatenate {args_str}, dim = 0 "
            f": ({types_str}) -> {_mlir_type(result_type)}"
        )
        return var

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
