from __future__ import annotations

from dataclasses import dataclass
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
    GradExpr,
    StructLiteral,
    FieldAccess,
    WithExpr,
    IndexExpr,
    IndexComponent,
    TypeAnnotation,
    Expr,
)
from .types import MaomiType, ScalarType, ArrayType, StructType, F32, F64, I32, I64, BOOL
from .errors import MaomiTypeError


# -- Function signatures --


@dataclass
class FnSignature:
    param_names: list[str]
    param_types: list[MaomiType]
    return_type: MaomiType


NUMERIC_BASES = {"f32", "f64", "i32", "i64"}
COMPARISON_OPS = {"==", "!=", "<", ">", "<=", ">="}
ARITHMETIC_OPS = {"+", "-", "*", "/", "**", "@"}


def _is_numeric(t: MaomiType) -> bool:
    if isinstance(t, ScalarType):
        return t.base in NUMERIC_BASES
    if isinstance(t, ArrayType):
        return t.base in NUMERIC_BASES
    return False


def _base_of(t: MaomiType) -> str:
    if isinstance(t, ScalarType):
        return t.base
    return t.base


# -- Built-in functions --

def _make_builtins() -> dict[str, FnSignature]:
    builtins = {}
    # Reduction: array -> scalar
    for name in ("mean", "sum"):
        builtins[name] = FnSignature(["x"], [ArrayType("f32", ("N",))], F32)
    # Elementwise: scalar -> scalar (also works on arrays via type matching)
    for name in ("exp", "log", "tanh", "sqrt", "abs"):
        builtins[name] = FnSignature(["x"], [ScalarType("f32")], F32)
    return builtins


BUILTINS = _make_builtins()
_ELEMENTWISE_BUILTINS = {"exp", "log", "tanh", "sqrt", "abs"}
_CALLBACK_BUILTINS = {"callback"}


# -- Scoped environment --


class TypeEnv:
    def __init__(self, parent: TypeEnv | None = None):
        self.parent = parent
        self.bindings: dict[str, MaomiType] = {}

    def define(self, name: str, typ: MaomiType):
        self.bindings[name] = typ

    def lookup(self, name: str) -> MaomiType | None:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def child(self) -> TypeEnv:
        return TypeEnv(parent=self)


# -- Type checker --


class TypeChecker:
    def __init__(self, filename: str = "<stdin>"):
        self.filename = filename
        self.fn_table: dict[str, FnSignature] = dict(BUILTINS)
        self.struct_defs: dict[str, StructType] = {}
        self.errors: list[MaomiTypeError] = []
        self.type_map: dict[int, MaomiType] = {}  # id(expr) -> inferred type

    def check(self, program: Program) -> list[MaomiTypeError]:
        # Pass 0: register all struct definitions
        for sd in program.struct_defs:
            self._register_struct(sd)

        # Pass 1: register all function signatures
        for fn in program.functions:
            sig = self._resolve_signature(fn)
            if sig:
                self.fn_table[fn.name] = sig

        # Pass 2: check each function body
        for fn in program.functions:
            self._check_fn(fn)

        return self.errors

    def _register_struct(self, sd):
        fields: list[tuple[str, MaomiType]] = []
        for field_name, field_ta in sd.fields:
            t = self._resolve_type_annotation(field_ta)
            if t is None:
                self._error(
                    f"struct '{sd.name}': unknown type for field '{field_name}'",
                    field_ta.span.line_start,
                    field_ta.span.col_start,
                )
                return
            fields.append((field_name, t))
        self.struct_defs[sd.name] = StructType(sd.name, tuple(fields))

    def _error(self, msg: str, line: int, col: int):
        self.errors.append(MaomiTypeError(msg, self.filename, line, col))

    # -- Signature resolution --

    def _resolve_signature(self, fn: FnDef) -> FnSignature | None:
        param_names = []
        param_types = []
        for p in fn.params:
            t = self._resolve_type_annotation(p.type_annotation)
            if t is None:
                return None
            param_names.append(p.name)
            param_types.append(t)
        ret = self._resolve_type_annotation(fn.return_type)
        if ret is None:
            return None
        return FnSignature(param_names, param_types, ret)

    _BASE_TYPES = {"f32", "f64", "i32", "i64", "bool"}

    def _resolve_type_annotation(self, ta: TypeAnnotation) -> MaomiType | None:
        if ta.base in self._BASE_TYPES:
            if ta.dims is None:
                return ScalarType(ta.base)
            dims = tuple(d.value for d in ta.dims)
            return ArrayType(ta.base, dims)
        # Struct type
        if ta.base in self.struct_defs:
            return self.struct_defs[ta.base]
        self._error(
            f"unknown type: '{ta.base}'",
            ta.span.line_start,
            ta.span.col_start,
        )
        return None

    # -- Function body checking --

    def _check_fn(self, fn: FnDef):
        env = TypeEnv()
        for p in fn.params:
            t = self._resolve_type_annotation(p.type_annotation)
            if t:
                env.define(p.name, t)

        body_type = self._check_block(fn.body, env)
        if body_type is None:
            return

        expected = self._resolve_type_annotation(fn.return_type)
        if expected and not self._types_compatible(body_type, expected):
            self._error(
                f"return type mismatch: function '{fn.name}' declares {expected} but body returns {body_type}",
                fn.body.span.line_end,
                fn.body.span.col_end,
            )

    def _check_block(self, block: Block, env: TypeEnv) -> MaomiType | None:
        child_env = env.child()
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                self._check_let(stmt, child_env)
            elif isinstance(stmt, ExprStmt):
                self._infer(stmt.expr, child_env)

        if block.expr is not None:
            return self._infer(block.expr, child_env)
        return None

    def _check_let(self, stmt: LetStmt, env: TypeEnv):
        inferred = self._infer(stmt.value, env)
        if inferred is None:
            return

        if stmt.type_annotation is not None:
            declared = self._resolve_type_annotation(stmt.type_annotation)
            if declared and not self._types_compatible(inferred, declared):
                self._error(
                    f"type mismatch in let binding '{stmt.name}': declared {declared} but got {inferred}",
                    stmt.span.line_start,
                    stmt.span.col_start,
                )
            env.define(stmt.name, declared or inferred)
        else:
            env.define(stmt.name, inferred)

    # -- Expression type inference --

    def _infer(self, expr: Expr, env: TypeEnv) -> MaomiType | None:
        result = self._infer_inner(expr, env)
        if result is not None:
            self.type_map[id(expr)] = result
        return result

    def _infer_inner(self, expr: Expr, env: TypeEnv) -> MaomiType | None:
        match expr:
            case IntLiteral():
                return I32
            case FloatLiteral():
                return F32
            case BoolLiteral():
                return BOOL
            case Identifier(name=name):
                t = env.lookup(name)
                if t is None:
                    self._error(f"undefined variable: '{name}'", expr.span.line_start, expr.span.col_start)
                return t
            case UnaryOp(op=op, operand=operand):
                return self._check_unary(op, operand, expr, env)
            case BinOp(op=op, left=left, right=right):
                return self._check_binop(op, left, right, expr, env)
            case IfExpr():
                return self._check_if(expr, env)
            case CallExpr():
                return self._check_call(expr, env)
            case ScanExpr():
                return self._check_scan(expr, env)
            case MapExpr():
                return self._check_map(expr, env)
            case GradExpr():
                return self._check_grad(expr, env)
            case StructLiteral():
                return self._check_struct_literal(expr, env)
            case FieldAccess():
                return self._check_field_access(expr, env)
            case WithExpr():
                return self._check_with(expr, env)
            case IndexExpr():
                return self._check_index(expr, env)
            case _:
                return None

    def _check_scan(self, expr: ScanExpr, env: TypeEnv) -> MaomiType | None:
        carry_type = self._infer(expr.init, env)

        # Infer and validate all sequences
        seq_types = []
        for seq in expr.sequences:
            st = self._infer(seq, env)
            if st is None:
                return None
            if not isinstance(st, ArrayType):
                self._error(
                    f"scan sequence must be an array, got {st}",
                    seq.span.line_start,
                    seq.span.col_start,
                )
                return None
            seq_types.append(st)

        if carry_type is None:
            return None

        # All sequences must have the same first dimension
        first_dims = [st.dims[0] for st in seq_types]
        for i, fd in enumerate(first_dims[1:], 1):
            if not self._dims_match(fd, first_dims[0]):
                self._error(
                    f"scan sequences must have the same first dimension, got {first_dims[0]} and {fd}",
                    expr.sequences[i].span.line_start,
                    expr.sequences[i].span.col_start,
                )
                return None

        # Check that elem_vars count matches sequences count
        if len(expr.elem_vars) != len(expr.sequences):
            self._error(
                f"scan has {len(expr.elem_vars)} element variables but {len(expr.sequences)} sequences",
                expr.span.line_start,
                expr.span.col_start,
            )
            return None

        # Bind carry and element variables in body scope
        body_env = env.child()
        body_env.define(expr.carry_var, carry_type)
        for ev, st in zip(expr.elem_vars, seq_types):
            if len(st.dims) == 1:
                elem_type: MaomiType = ScalarType(st.base)
            else:
                elem_type = ArrayType(st.base, st.dims[1:])
            body_env.define(ev, elem_type)

        body_type = self._check_block(expr.body, body_env)

        if body_type is None:
            return None

        if not self._types_compatible(body_type, carry_type):
            self._error(
                f"scan body returns {body_type}, but carry has type {carry_type}",
                expr.body.span.line_start,
                expr.body.span.col_start,
            )
            return None

        # Result: stacked carries — [seq_first_dim, ...carry_dims]
        seq_first = first_dims[0]
        if isinstance(carry_type, ScalarType):
            return ArrayType(carry_type.base, (seq_first,))
        else:
            return ArrayType(carry_type.base, (seq_first,) + carry_type.dims)

    def _check_map(self, expr: MapExpr, env: TypeEnv) -> MaomiType | None:
        seq_type = self._infer(expr.sequence, env)

        if seq_type is None:
            return None

        if not isinstance(seq_type, ArrayType):
            self._error(
                f"map sequence must be an array, got {seq_type}",
                expr.sequence.span.line_start,
                expr.sequence.span.col_start,
            )
            return None

        # Element type = sequence with first dim removed
        if len(seq_type.dims) == 1:
            elem_type: MaomiType = ScalarType(seq_type.base)
        else:
            elem_type = ArrayType(seq_type.base, seq_type.dims[1:])

        body_env = env.child()
        body_env.define(expr.elem_var, elem_type)
        body_type = self._check_block(expr.body, body_env)

        if body_type is None:
            return None

        # Result: [seq_first_dim, ...body_dims]
        seq_first = seq_type.dims[0]
        if isinstance(body_type, ScalarType):
            return ArrayType(body_type.base, (seq_first,))
        else:
            return ArrayType(body_type.base, (seq_first,) + body_type.dims)

    def _check_grad(self, expr: GradExpr, env: TypeEnv) -> MaomiType | None:
        wrt_type = env.lookup(expr.wrt)
        if wrt_type is None:
            self._error(
                f"grad: undefined variable '{expr.wrt}'",
                expr.span.line_start,
                expr.span.col_start,
            )
            return None

        expr_type = self._infer(expr.expr, env)
        if expr_type is None:
            return None

        if not isinstance(expr_type, ScalarType):
            self._error(
                f"grad: expression must be scalar, got {expr_type}",
                expr.expr.span.line_start,
                expr.expr.span.col_start,
            )
            return None

        return wrt_type

    def _check_struct_literal(self, expr: StructLiteral, env: TypeEnv) -> MaomiType | None:
        stype = self.struct_defs.get(expr.name)
        if stype is None:
            self._error(f"unknown struct: '{expr.name}'", expr.span.line_start, expr.span.col_start)
            return None

        if len(expr.fields) != len(stype.fields):
            self._error(
                f"struct '{expr.name}' has {len(stype.fields)} fields, got {len(expr.fields)}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        for (given_name, given_expr), (expected_name, expected_type) in zip(expr.fields, stype.fields):
            if given_name != expected_name:
                self._error(
                    f"struct '{expr.name}': expected field '{expected_name}', got '{given_name}'",
                    given_expr.span.line_start, given_expr.span.col_start,
                )
                return None
            given_type = self._infer(given_expr, env)
            if given_type is not None and not self._types_compatible(given_type, expected_type):
                self._error(
                    f"struct '{expr.name}' field '{given_name}': expected {expected_type}, got {given_type}",
                    given_expr.span.line_start, given_expr.span.col_start,
                )

        return stype

    def _check_field_access(self, expr: FieldAccess, env: TypeEnv) -> MaomiType | None:
        obj_type = self._infer(expr.object, env)
        if obj_type is None:
            return None
        if not isinstance(obj_type, StructType):
            self._error(
                f"field access on non-struct type: {obj_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None
        for field_name, field_type in obj_type.fields:
            if field_name == expr.field:
                return field_type
        self._error(
            f"struct '{obj_type.name}' has no field '{expr.field}'",
            expr.span.line_start, expr.span.col_start,
        )
        return None

    def _check_with(self, expr: WithExpr, env: TypeEnv) -> MaomiType | None:
        base_type = self._infer(expr.base, env)
        if base_type is None:
            return None
        if not isinstance(base_type, StructType):
            self._error(
                f"'with' requires a struct, got {base_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        for path, value_expr in expr.updates:
            # Walk the path to find the target field type
            current_type = base_type
            for i, field_name in enumerate(path):
                if not isinstance(current_type, StructType):
                    self._error(
                        f"'with' path '{'.'.join(path[:i+1])}': '{path[i-1]}' is not a struct",
                        value_expr.span.line_start, value_expr.span.col_start,
                    )
                    break
                found = False
                for fn, ft in current_type.fields:
                    if fn == field_name:
                        current_type = ft
                        found = True
                        break
                if not found:
                    self._error(
                        f"struct '{current_type.name}' has no field '{field_name}'",
                        value_expr.span.line_start, value_expr.span.col_start,
                    )
                    break
            else:
                # Path resolved — check value type
                value_type = self._infer(value_expr, env)
                if value_type is not None and not self._types_compatible(value_type, current_type):
                    self._error(
                        f"'with' field '{'.'.join(path)}': expected {current_type}, got {value_type}",
                        value_expr.span.line_start, value_expr.span.col_start,
                    )

        return base_type

    def _check_index(self, expr: IndexExpr, env: TypeEnv) -> MaomiType | None:
        base_type = self._infer(expr.base, env)
        if base_type is None:
            return None

        if not isinstance(base_type, ArrayType):
            self._error(
                f"indexing requires an array, got {base_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        if len(expr.indices) > len(base_type.dims):
            self._error(
                f"too many indices: array has {len(base_type.dims)} dimensions but got {len(expr.indices)} indices",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        result_dims: list[int | str] = []
        has_array_index = False
        for i, ic in enumerate(expr.indices):
            dim = base_type.dims[i]
            if ic.kind == "single":
                idx_type = self._infer(ic.value, env)
                if idx_type is not None:
                    if isinstance(idx_type, ArrayType):
                        # Array-based indexing (gather)
                        if idx_type.base not in ("i32", "i64"):
                            self._error(
                                f"array index must have integer element type, got {idx_type}",
                                ic.span.line_start, ic.span.col_start,
                            )
                            return None
                        if len(idx_type.dims) != 1:
                            self._error(
                                f"array index must be 1-D, got {idx_type}",
                                ic.span.line_start, ic.span.col_start,
                            )
                            return None
                        if has_array_index:
                            self._error(
                                "only one array index is supported per indexing expression",
                                ic.span.line_start, ic.span.col_start,
                            )
                            return None
                        has_array_index = True
                        result_dims.append(idx_type.dims[0])
                        continue
                    elif idx_type != I32:
                        self._error(
                            f"index must be i32 or integer array, got {idx_type}",
                            ic.span.line_start, ic.span.col_start,
                        )
                # Single scalar index removes this dimension
            elif ic.kind == "slice":
                start_type = self._infer(ic.start, env)
                end_type = self._infer(ic.end, env)
                # Slice bounds must be integer literals (static)
                if not isinstance(ic.start, IntLiteral):
                    self._error(
                        "slice start must be an integer literal",
                        ic.start.span.line_start, ic.start.span.col_start,
                    )
                    return None
                if not isinstance(ic.end, IntLiteral):
                    self._error(
                        "slice end must be an integer literal",
                        ic.end.span.line_start, ic.end.span.col_start,
                    )
                    return None
                slice_size = ic.end.value - ic.start.value
                if slice_size <= 0:
                    self._error(
                        f"slice range is empty or negative: {ic.start.value}:{ic.end.value}",
                        ic.span.line_start, ic.span.col_start,
                    )
                    return None
                result_dims.append(slice_size)
            elif ic.kind == "full":
                result_dims.append(dim)

        # Trailing unindexed axes pass through
        for i in range(len(expr.indices), len(base_type.dims)):
            result_dims.append(base_type.dims[i])

        if len(result_dims) == 0:
            return ScalarType(base_type.base)
        return ArrayType(base_type.base, tuple(result_dims))

    def _check_unary(self, op: str, operand: Expr, expr: Expr, env: TypeEnv) -> MaomiType | None:
        t = self._infer(operand, env)
        if t is None:
            return None
        if op == "-" and _is_numeric(t):
            return t
        self._error(f"invalid unary {op} on type {t}", expr.span.line_start, expr.span.col_start)
        return None

    def _check_binop(self, op: str, left: Expr, right: Expr, expr: Expr, env: TypeEnv) -> MaomiType | None:
        lt = self._infer(left, env)
        rt = self._infer(right, env)
        if lt is None or rt is None:
            return None

        if op == "@":
            return self._check_matmul(lt, rt, expr)

        if op in COMPARISON_OPS:
            result_shape = self._broadcast(lt, rt)
            if result_shape is None:
                self._error(
                    f"comparison {op}: mismatched types {lt} and {rt}",
                    expr.span.line_start,
                    expr.span.col_start,
                )
                return None
            # Comparison result is bool with the broadcast shape
            if isinstance(result_shape, ArrayType):
                return ArrayType("bool", result_shape.dims)
            return BOOL

        # Arithmetic
        if not _is_numeric(lt) or not _is_numeric(rt):
            self._error(
                f"operator {op}: expected numeric types, got {lt} and {rt}",
                expr.span.line_start,
                expr.span.col_start,
            )
            return None

        result = self._broadcast(lt, rt)
        if result is None:
            self._error(
                f"operator {op}: mismatched types {lt} and {rt}",
                expr.span.line_start,
                expr.span.col_start,
            )
            return None

        return result

    def _check_matmul(self, lt: MaomiType, rt: MaomiType, expr: Expr) -> MaomiType | None:
        if not isinstance(lt, ArrayType) or not isinstance(rt, ArrayType):
            self._error(
                f"matmul (@): both operands must be arrays, got {lt} and {rt}",
                expr.span.line_start,
                expr.span.col_start,
            )
            return None

        if lt.base != rt.base:
            self._error(
                f"matmul (@): base type mismatch: {lt.base} vs {rt.base}",
                expr.span.line_start,
                expr.span.col_start,
            )
            return None

        if len(lt.dims) < 1 or len(rt.dims) < 1:
            self._error(
                f"matmul (@): operands must have at least 1 dimension",
                expr.span.line_start,
                expr.span.col_start,
            )
            return None

        # Check contracting dimensions
        left_contract = lt.dims[-1]
        right_contract = rt.dims[0]
        if not self._dims_match(left_contract, right_contract):
            self._error(
                f"matmul (@): dimension mismatch: left has {left_contract}, right has {right_contract}",
                expr.span.line_start,
                expr.span.col_start,
            )
            return None

        # Result shape: left batch dims + left non-contract dims + right non-contract dims
        # For 2D: [M, K] @ [K, N] -> [M, N]
        # For general: [..., M, K] @ [K, N] -> [..., M, N]
        result_dims = lt.dims[:-1] + rt.dims[1:]
        if len(result_dims) == 0:
            return ScalarType(lt.base)
        return ArrayType(lt.base, tuple(result_dims))

    def _check_if(self, expr: IfExpr, env: TypeEnv) -> MaomiType | None:
        cond_type = self._infer(expr.condition, env)
        then_type = self._check_block(expr.then_block, env)
        else_type = self._check_block(expr.else_block, env)

        # Accept bool or array-of-bool conditions
        if cond_type is not None and cond_type != BOOL:
            if not (isinstance(cond_type, ArrayType) and cond_type.base == "bool"):
                self._error(
                    f"if condition must be bool, got {cond_type}",
                    expr.condition.span.line_start,
                    expr.condition.span.col_start,
                )

        if then_type is None or else_type is None:
            return then_type or else_type

        result = self._broadcast(then_type, else_type)
        if result is None:
            self._error(
                f"if/else branches have different types: {then_type} vs {else_type}",
                expr.span.line_start,
                expr.span.col_start,
            )
            return None

        return result

    def _check_call(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        # callback is a special builtin: any args, no return value
        if expr.callee in _CALLBACK_BUILTINS:
            for arg in expr.args:
                self._infer(arg, env)
            return None

        # iota(N) — returns i32[N], N must be a positive integer literal
        if expr.callee == "iota":
            if len(expr.args) != 1:
                self._error("iota expects exactly 1 argument", expr.span.line_start, expr.span.col_start)
                return None
            arg = expr.args[0]
            if not isinstance(arg, IntLiteral):
                self._error("iota argument must be an integer literal", expr.span.line_start, expr.span.col_start)
                return None
            if arg.value <= 0:
                self._error("iota argument must be positive", expr.span.line_start, expr.span.col_start)
                return None
            self._infer(arg, env)
            return ArrayType("i32", (arg.value,))

        sig = self.fn_table.get(expr.callee)
        if sig is None:
            self._error(
                f"undefined function: '{expr.callee}'",
                expr.span.line_start,
                expr.span.col_start,
            )
            return None

        if len(expr.args) != len(sig.param_types):
            self._error(
                f"function '{expr.callee}' expects {len(sig.param_types)} arguments, got {len(expr.args)}",
                expr.span.line_start,
                expr.span.col_start,
            )
            return None

        # Unify symbolic dimensions
        substitution: dict[str, int | str] = {}
        arg_types: list[MaomiType | None] = []

        for i, (arg, param_type) in enumerate(zip(expr.args, sig.param_types)):
            arg_type = self._infer(arg, env)
            arg_types.append(arg_type)
            if arg_type is None:
                continue

            if not self._unify_arg(arg_type, param_type, substitution, expr, i):
                continue

        # Apply substitution to return type
        ret = self._apply_substitution(sig.return_type, substitution)

        # For elementwise builtins (scalar->scalar), lift return to array if arg is array
        if expr.callee in _ELEMENTWISE_BUILTINS and isinstance(ret, ScalarType):
            for at in arg_types:
                if isinstance(at, ArrayType) and at.base == ret.base:
                    ret = ArrayType(ret.base, at.dims)
                    break

        return ret

    def _unify_arg(
        self,
        arg_type: MaomiType,
        param_type: MaomiType,
        subst: dict[str, int | str],
        expr: CallExpr,
        arg_index: int,
    ) -> bool:
        # Scalar vs Scalar
        if isinstance(arg_type, ScalarType) and isinstance(param_type, ScalarType):
            if arg_type.base != param_type.base:
                self._error(
                    f"argument {arg_index} of '{expr.callee}': expected {param_type}, got {arg_type}",
                    expr.args[arg_index].span.line_start,
                    expr.args[arg_index].span.col_start,
                )
                return False
            return True

        # Array vs Array
        if isinstance(arg_type, ArrayType) and isinstance(param_type, ArrayType):
            if arg_type.base != param_type.base:
                self._error(
                    f"argument {arg_index} of '{expr.callee}': base type mismatch: {arg_type.base} vs {param_type.base}",
                    expr.args[arg_index].span.line_start,
                    expr.args[arg_index].span.col_start,
                )
                return False

            if len(arg_type.dims) != len(param_type.dims):
                self._error(
                    f"argument {arg_index} of '{expr.callee}': rank mismatch: {len(arg_type.dims)} vs {len(param_type.dims)}",
                    expr.args[arg_index].span.line_start,
                    expr.args[arg_index].span.col_start,
                )
                return False

            for j, (ad, pd) in enumerate(zip(arg_type.dims, param_type.dims)):
                if isinstance(pd, str):
                    # Symbolic param dim — unify
                    if pd in subst:
                        if not self._dims_match(subst[pd], ad):
                            self._error(
                                f"argument {arg_index} of '{expr.callee}': dimension '{pd}' was bound to {subst[pd]} but got {ad}",
                                expr.args[arg_index].span.line_start,
                                expr.args[arg_index].span.col_start,
                            )
                            return False
                    else:
                        subst[pd] = ad
                elif isinstance(ad, str):
                    # Symbolic arg dim — accept (can't check at compile time)
                    pass
                else:
                    # Both concrete
                    if ad != pd:
                        self._error(
                            f"argument {arg_index} of '{expr.callee}': dimension mismatch at axis {j}: {ad} vs {pd}",
                            expr.args[arg_index].span.line_start,
                            expr.args[arg_index].span.col_start,
                        )
                        return False
            return True

        # Scalar arg for array param (e.g. builtin elementwise functions accept scalars too)
        if isinstance(arg_type, ScalarType) and isinstance(param_type, ArrayType):
            if arg_type.base == param_type.base:
                return True  # Accept scalar where array is expected (elementwise builtins)

        # Array arg for scalar param (e.g. elementwise builtins)
        if isinstance(arg_type, ArrayType) and isinstance(param_type, ScalarType):
            if arg_type.base == param_type.base:
                return True

        self._error(
            f"argument {arg_index} of '{expr.callee}': expected {param_type}, got {arg_type}",
            expr.args[arg_index].span.line_start,
            expr.args[arg_index].span.col_start,
        )
        return False

    # -- Substitution --

    def _apply_substitution(self, t: MaomiType, subst: dict[str, int | str]) -> MaomiType:
        if isinstance(t, ScalarType):
            return t
        if isinstance(t, ArrayType):
            new_dims = tuple(subst.get(d, d) if isinstance(d, str) else d for d in t.dims)
            return ArrayType(t.base, new_dims)
        return t

    # -- Broadcasting --

    def _broadcast(self, a: MaomiType, b: MaomiType) -> MaomiType | None:
        """Try to broadcast two types. Returns the result type, or None if incompatible."""
        if _base_of(a) != _base_of(b):
            return None

        # Same type — no broadcast needed
        if self._types_compatible(a, b):
            return a

        # Scalar + Array → Array
        if isinstance(a, ScalarType) and isinstance(b, ArrayType):
            return b
        if isinstance(a, ArrayType) and isinstance(b, ScalarType):
            return a

        # Array + Array with different ranks — trailing dims must match
        if isinstance(a, ArrayType) and isinstance(b, ArrayType):
            if len(a.dims) >= len(b.dims):
                longer, shorter = a, b
            else:
                longer, shorter = b, a
            # Check that shorter dims match the trailing dims of longer
            offset = len(longer.dims) - len(shorter.dims)
            for i, sd in enumerate(shorter.dims):
                ld = longer.dims[offset + i]
                if not self._dims_match(sd, ld):
                    return None
            return longer

        return None

    # -- Type compatibility --

    def _types_compatible(self, a: MaomiType, b: MaomiType) -> bool:
        if isinstance(a, ScalarType) and isinstance(b, ScalarType):
            return a.base == b.base
        if isinstance(a, ArrayType) and isinstance(b, ArrayType):
            if a.base != b.base or len(a.dims) != len(b.dims):
                return False
            return all(self._dims_match(d1, d2) for d1, d2 in zip(a.dims, b.dims))
        if isinstance(a, StructType) and isinstance(b, StructType):
            return a.name == b.name
        return False

    def _dims_match(self, a: int | str, b: int | str) -> bool:
        if isinstance(a, str) or isinstance(b, str):
            return True  # symbolic dims always match
        return a == b
