from __future__ import annotations

import math
from dataclasses import dataclass, field
from .ast_nodes import (
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
    GradExpr,
    ValueAndGradExpr,
    CastExpr,
    FoldExpr,
    ArrayLiteral,
    StructLiteral,
    FieldAccess,
    WithExpr,
    IndexExpr,
    IndexComponent,
    TypeAnnotation,
    Expr,
)
from .types import MaomiType, ScalarType, ArrayType, StructType, WildcardArrayType, StringType, TypeVar, F32, F64, I32, I64, BOOL, STRING, FLOAT_BASES
from .errors import MaomiTypeError


def _substitute_comptime(block: Block, subst: dict[str, IntLiteral]) -> None:
    """Replace Identifier nodes matching comptime param names with IntLiteral values in-place."""
    def walk_expr(expr):
        if expr is None:
            return expr
        if isinstance(expr, Identifier) and expr.name in subst:
            return subst[expr.name]
        if isinstance(expr, BinOp):
            expr.left = walk_expr(expr.left)
            expr.right = walk_expr(expr.right)
        elif isinstance(expr, UnaryOp):
            expr.operand = walk_expr(expr.operand)
        elif isinstance(expr, CallExpr):
            expr.args = [walk_expr(a) for a in expr.args]
            expr.named_args = [(n, walk_expr(v)) for n, v in expr.named_args]
        elif isinstance(expr, IfExpr):
            expr.condition = walk_expr(expr.condition)
            walk_block(expr.then_block)
            walk_block(expr.else_block)
        elif isinstance(expr, GradExpr):
            expr.expr = walk_expr(expr.expr)
        elif isinstance(expr, ValueAndGradExpr):
            expr.expr = walk_expr(expr.expr)
        elif isinstance(expr, CastExpr):
            expr.expr = walk_expr(expr.expr)
        elif isinstance(expr, IndexExpr):
            expr.base = walk_expr(expr.base)
            for ic in expr.indices:
                ic.value = walk_expr(ic.value)
                if ic.end is not None:
                    ic.end = walk_expr(ic.end)
        elif isinstance(expr, StructLiteral):
            expr.fields = [(n, walk_expr(v)) for n, v in expr.fields]
        elif isinstance(expr, FieldAccess):
            expr.expr = walk_expr(expr.expr)
        elif isinstance(expr, WithExpr):
            expr.base = walk_expr(expr.base)
            expr.updates = [(path, walk_expr(v)) for path, v in expr.updates]
        elif isinstance(expr, ScanExpr):
            expr.init = walk_expr(expr.init)
            expr.sequence = walk_expr(expr.sequence)
            walk_block(expr.body)
        elif isinstance(expr, FoldExpr):
            expr.init = walk_expr(expr.init)
            expr.sequence = walk_expr(expr.sequence)
            walk_block(expr.body)
        elif isinstance(expr, MapExpr):
            expr.sequence = walk_expr(expr.sequence)
            walk_block(expr.body)
        elif isinstance(expr, WhileExpr):
            expr.init = walk_expr(expr.init)
            walk_block(expr.condition_block)
            walk_block(expr.body)
        elif isinstance(expr, ArrayLiteral):
            expr.elements = [walk_expr(e) for e in expr.elements]
        return expr

    def walk_block(block):
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                stmt.value = walk_expr(stmt.value)
            elif isinstance(stmt, ExprStmt):
                stmt.expr = walk_expr(stmt.expr)
        if block.expr is not None:
            block.expr = walk_expr(block.expr)

    walk_block(block)


# -- Function signatures --


@dataclass
class FnSignature:
    param_names: list[str]
    param_types: list[MaomiType]
    return_type: MaomiType
    comptime: list[bool] = field(default_factory=list)


NUMERIC_BASES = {"f32", "f64", "bf16", "i32", "i64"}
COMPARISON_OPS = {"==", "!=", "<", ">", "<=", ">="}
ARITHMETIC_OPS = {"+", "-", "*", "/", "**", "@"}


def _try_negative_literal(expr) -> int | None:
    """If expr is UnaryOp('-', IntLiteral(n)), return -n. Otherwise None."""
    if isinstance(expr, UnaryOp) and expr.op == "-" and isinstance(expr.operand, IntLiteral):
        return -expr.operand.value
    return None


def _try_float_literal(expr) -> float | None:
    """Extract float value from FloatLiteral or UnaryOp('-', FloatLiteral). Otherwise None."""
    if isinstance(expr, FloatLiteral):
        return expr.value
    if isinstance(expr, UnaryOp) and expr.op == "-" and isinstance(expr.operand, FloatLiteral):
        return -expr.operand.value
    return None


def _ast_structurally_equal(a, b) -> bool:
    """Check if two AST expressions are structurally identical (ignoring spans)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, Identifier):
        return a.name == b.name
    if isinstance(a, IntLiteral):
        return a.value == b.value
    if isinstance(a, FloatLiteral):
        return a.value == b.value
    if isinstance(a, BinOp):
        return a.op == b.op and _ast_structurally_equal(a.left, b.left) and _ast_structurally_equal(a.right, b.right)
    if isinstance(a, UnaryOp):
        return a.op == b.op and _ast_structurally_equal(a.operand, b.operand)
    return False


def _try_static_slice_size(start, end) -> int | None:
    """Try to determine the static size of a slice from start and end expressions.

    Returns the size if determinable, None otherwise.
    Handles: both literals, end = start + N, start = end - N.
    """
    if isinstance(start, IntLiteral) and isinstance(end, IntLiteral):
        return end.value - start.value
    if isinstance(end, BinOp) and end.op == "+":
        if isinstance(end.right, IntLiteral) and _ast_structurally_equal(end.left, start):
            return end.right.value
        if isinstance(end.left, IntLiteral) and _ast_structurally_equal(end.right, start):
            return end.left.value
    if isinstance(start, BinOp) and start.op == "-":
        if isinstance(start.right, IntLiteral) and _ast_structurally_equal(start.left, end):
            return start.right.value
    return None


def _is_numeric(t: MaomiType) -> bool:
    if isinstance(t, ScalarType):
        return t.base in NUMERIC_BASES
    if isinstance(t, ArrayType):
        return t.base in NUMERIC_BASES
    return False


def _struct_has_float_leaves(t: StructType) -> bool:
    """Check that all leaf fields of a struct are float types."""
    for _, ft in t.fields:
        if isinstance(ft, StructType):
            if not _struct_has_float_leaves(ft):
                return False
        elif isinstance(ft, ScalarType):
            if ft.base not in FLOAT_BASES:
                return False
        elif isinstance(ft, ArrayType):
            if ft.base not in FLOAT_BASES:
                return False
        else:
            return False
    return True


def _struct_has_numeric_leaves(t: StructType) -> bool:
    """Check that all leaf fields of a struct are numeric (recursing into nested structs)."""
    for _, ft in t.fields:
        if isinstance(ft, StructType):
            if not _struct_has_numeric_leaves(ft):
                return False
        elif not _is_numeric(ft):
            return False
    return True


def _base_of(t: MaomiType) -> str:
    if isinstance(t, ScalarType):
        return t.base
    return t.base


# -- Built-in functions (derived from central registry) --

from .builtins import ELEMENTWISE as _EW_REGISTRY, COMPLEX as _CX_REGISTRY, ALL_NAMES as _ALL_BUILTIN_NAMES

BUILTINS = {name: FnSignature(["x"], [ScalarType("f32")], F32) for name in _EW_REGISTRY}
_ELEMENTWISE_BUILTINS = set(_EW_REGISTRY)
_CALLBACK_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "callback"}
_RNG_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "rng"}

# Key type alias — compiler-level alias for i32[4]
KEY_TYPE = ArrayType("i32", (4,))


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
        self._generic_fns: dict[str, FnDef] = {}   # name -> FnDef with wildcard types
        self._program: Program | None = None

    def check(self, program: Program) -> list[MaomiTypeError]:
        self._program = program
        # Pass 0: register all struct definitions
        for sd in program.struct_defs:
            self._register_struct(sd)

        # Pass 1: register all function signatures
        for fn in program.functions:
            sig = self._resolve_signature(fn)
            if sig:
                if self._is_generic_sig(sig):
                    self._generic_fns[fn.name] = fn
                else:
                    self.fn_table[fn.name] = sig

        # Pass 2: check each function body (skip generics — checked per-monomorphization)
        for fn in program.functions:
            if fn.name not in self._generic_fns:
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

    @staticmethod
    def _is_generic_sig(sig: FnSignature) -> bool:
        """True if function has wildcard, symbolic, type-variable, or comptime types needing monomorphization."""
        if sig.comptime and any(sig.comptime):
            return True
        for pt in sig.param_types:
            if isinstance(pt, WildcardArrayType):
                return True
            if isinstance(pt, TypeVar):
                return True
            if isinstance(pt, ArrayType) and any(isinstance(d, str) for d in pt.dims):
                return True
        rt = sig.return_type
        if isinstance(rt, WildcardArrayType):
            return True
        if isinstance(rt, TypeVar):
            return True
        if isinstance(rt, ArrayType) and any(isinstance(d, str) for d in rt.dims):
            return True
        return False

    def _resolve_named_args(self, expr, param_names: list[str]) -> None:
        """Resolve named arguments to positional, mutating expr.args in place."""
        if not expr.named_args:
            return
        for name, value in expr.named_args:
            if name not in param_names:
                self._error(
                    f"unknown parameter name '{name}'",
                    value.span.line_start, value.span.col_start,
                )
                continue
            idx = param_names.index(name)
            if idx < len(expr.args):
                self._error(
                    f"'{name}' already provided as positional argument",
                    value.span.line_start, value.span.col_start,
                )
                continue
            # Extend args list to fit, then place at correct index
            while len(expr.args) <= idx:
                expr.args.append(None)
            expr.args[idx] = value
        expr.named_args = []
        # Remove trailing Nones (optional args not provided)
        while expr.args and expr.args[-1] is None:
            expr.args.pop()

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
        comptime_flags = [p.comptime for p in fn.params]
        return FnSignature(param_names, param_types, ret, comptime=comptime_flags)

    _BASE_TYPES = {"f32", "f64", "bf16", "i32", "i64", "bool"}

    def _resolve_type_annotation(self, ta: TypeAnnotation) -> MaomiType | None:
        if ta.base in self._BASE_TYPES:
            if ta.wildcard:
                return WildcardArrayType(ta.base)
            if ta.dims is None:
                return ScalarType(ta.base)
            dims = tuple(d.value for d in ta.dims)
            return ArrayType(ta.base, dims)
        # Key type alias
        if ta.base == "Key":
            if ta.dims is not None:
                self._error(
                    "Key type does not take dimensions (it is already i32[4])",
                    ta.span.line_start, ta.span.col_start,
                )
                return None
            return KEY_TYPE
        # Struct type
        if ta.base in self.struct_defs:
            return self.struct_defs[ta.base]
        # Type variable: single uppercase letter (A-Z)
        if len(ta.base) == 1 and ta.base.isupper() and ta.dims is None:
            return TypeVar(ta.base)
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
        if expected and isinstance(expected, WildcardArrayType):
            # Wildcard return type: accept any array/scalar with matching base type.
            # The body determines the concrete return type (used by _check_generic_call).
            if isinstance(body_type, (ArrayType, ScalarType)) and body_type.base == expected.base:
                return  # OK — concrete return inferred from body
            self._error(
                f"return type mismatch: function '{fn.name}' declares {expected} but body returns {body_type}",
                fn.body.span.line_end,
                fn.body.span.col_end,
            )
        elif expected and not self._types_compatible(body_type, expected):
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

        if isinstance(inferred, StringType):
            self._error(
                "cannot bind string to a variable",
                stmt.span.line_start,
                stmt.span.col_start,
            )
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
            case StringLiteral():
                return STRING
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
            case WhileExpr():
                return self._check_while(expr, env)
            case MapExpr():
                return self._check_map(expr, env)
            case GradExpr():
                return self._check_grad(expr, env)
            case ValueAndGradExpr():
                return self._check_value_and_grad(expr, env)
            case CastExpr():
                return self._check_cast(expr, env)
            case FoldExpr():
                return self._check_fold(expr, env)
            case ArrayLiteral():
                return self._check_array_literal(expr, env)
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

    def _check_while(self, expr: WhileExpr, env: TypeEnv) -> MaomiType | None:
        state_type = self._infer(expr.init, env)
        if state_type is None:
            return None

        body_env = env.child()
        body_env.define(expr.state_var, state_type)

        cond_type = self._check_block(expr.cond, body_env)
        if cond_type is None:
            return None
        if not (isinstance(cond_type, ScalarType) and cond_type.base == "bool"):
            self._error(
                f"while condition must return bool, got {cond_type}",
                expr.cond.expr.span.line_start if expr.cond.expr else expr.span.line_start,
                expr.cond.expr.span.col_start if expr.cond.expr else expr.span.col_start,
            )
            return None

        body_type = self._check_block(expr.body, body_env)
        if body_type is None:
            return None
        if not self._types_compatible(body_type, state_type):
            self._error(
                f"while body returns {body_type}, but state has type {state_type}",
                expr.body.expr.span.line_start if expr.body.expr else expr.span.line_start,
                expr.body.expr.span.col_start if expr.body.expr else expr.span.col_start,
            )
            return None

        return state_type

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

    def _check_value_and_grad(self, expr: ValueAndGradExpr, env: TypeEnv) -> MaomiType | None:
        wrt_type = env.lookup(expr.wrt)
        if wrt_type is None:
            self._error(
                f"value_and_grad: undefined variable '{expr.wrt}'",
                expr.span.line_start,
                expr.span.col_start,
            )
            return None

        expr_type = self._infer(expr.expr, env)
        if expr_type is None:
            return None

        if not isinstance(expr_type, ScalarType):
            self._error(
                f"value_and_grad: expression must be scalar, got {expr_type}",
                expr.expr.span.line_start,
                expr.expr.span.col_start,
            )
            return None

        return StructType("_ValueAndGrad", (("value", expr_type), ("gradient", wrt_type)))

    _CAST_BASES = {"f32", "f64", "bf16", "i32", "i64", "bool"}

    def _check_cast(self, expr: CastExpr, env: TypeEnv) -> MaomiType | None:
        inner_type = self._infer(expr.expr, env)
        if inner_type is None:
            return None
        if expr.target_type not in self._CAST_BASES:
            self._error(f"cast: unknown target type '{expr.target_type}'", expr.span.line_start, expr.span.col_start)
            return None
        if isinstance(inner_type, StructType):
            self._error("cast: cannot cast struct types", expr.span.line_start, expr.span.col_start)
            return None
        if isinstance(inner_type, ScalarType):
            return ScalarType(expr.target_type)
        if isinstance(inner_type, ArrayType):
            return ArrayType(expr.target_type, inner_type.dims)
        return None

    def _check_fold(self, expr: FoldExpr, env: TypeEnv) -> MaomiType | None:
        carry_type = self._infer(expr.init, env)

        seq_types = []
        for seq in expr.sequences:
            st = self._infer(seq, env)
            if st is None:
                return None
            if not isinstance(st, ArrayType):
                self._error(f"fold sequence must be an array, got {st}", seq.span.line_start, seq.span.col_start)
                return None
            seq_types.append(st)

        if carry_type is None:
            return None

        first_dims = [st.dims[0] for st in seq_types]
        for i, fd in enumerate(first_dims[1:], 1):
            if not self._dims_match(fd, first_dims[0]):
                self._error(
                    f"fold sequences must have the same first dimension, got {first_dims[0]} and {fd}",
                    expr.sequences[i].span.line_start, expr.sequences[i].span.col_start,
                )
                return None

        if len(expr.elem_vars) != len(expr.sequences):
            self._error(
                f"fold has {len(expr.elem_vars)} element variables but {len(expr.sequences)} sequences",
                expr.span.line_start, expr.span.col_start,
            )
            return None

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
                f"fold body returns {body_type}, but carry has type {carry_type}",
                expr.body.span.line_start, expr.body.span.col_start,
            )
            return None

        # fold returns the final carry (not stacked)
        return carry_type

    def _check_array_literal(self, expr: ArrayLiteral, env: TypeEnv) -> MaomiType | None:
        elem_types = [self._infer(e, env) for e in expr.elements]
        if any(t is None for t in elem_types):
            return None
        first = elem_types[0]
        for i, t in enumerate(elem_types[1:], 1):
            if not self._types_compatible(t, first):
                self._error(
                    f"array literal: element {i} has type {t}, expected {first}",
                    expr.elements[i].span.line_start,
                    expr.elements[i].span.col_start,
                )
                return None
        n = len(expr.elements)
        if isinstance(first, ScalarType):
            return ArrayType(first.base, (n,))
        elif isinstance(first, ArrayType):
            return ArrayType(first.base, (n,) + first.dims)
        else:
            self._error(
                f"array literal: unsupported element type {first}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

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
                # Normalize static negative literal: x[-1] on dim=10 → x[9]
                neg = _try_negative_literal(ic.value)
                if neg is not None:
                    if not isinstance(dim, int):
                        self._error(
                            f"negative index requires concrete dimension, got '{dim}'",
                            ic.span.line_start, ic.span.col_start,
                        )
                        return None
                    normalized = dim + neg
                    if normalized < 0:
                        self._error(
                            f"negative index {neg} out of bounds for axis of size {dim}",
                            ic.span.line_start, ic.span.col_start,
                        )
                        return None
                    ic.value = IntLiteral(normalized, ic.value.span)
                    self.type_map[id(ic.value)] = I32
                # Single scalar index removes this dimension
            elif ic.kind == "slice":
                # Fill open-ended bounds from known dimension
                if ic.start is None:
                    ic.start = IntLiteral(0, ic.span)
                    self.type_map[id(ic.start)] = I32
                if ic.end is None:
                    if not isinstance(dim, int):
                        self._error(
                            f"open-ended slice requires concrete dimension, got '{dim}'",
                            ic.span.line_start, ic.span.col_start,
                        )
                        return None
                    ic.end = IntLiteral(dim, ic.span)
                    self.type_map[id(ic.end)] = I32
                # Normalize negative literals in bounds
                neg_s = _try_negative_literal(ic.start)
                if neg_s is not None:
                    if not isinstance(dim, int):
                        self._error(
                            f"negative slice bound requires concrete dimension, got '{dim}'",
                            ic.span.line_start, ic.span.col_start,
                        )
                        return None
                    ic.start = IntLiteral(dim + neg_s, ic.start.span)
                    self.type_map[id(ic.start)] = I32
                neg_e = _try_negative_literal(ic.end)
                if neg_e is not None:
                    if not isinstance(dim, int):
                        self._error(
                            f"negative slice bound requires concrete dimension, got '{dim}'",
                            ic.span.line_start, ic.span.col_start,
                        )
                        return None
                    ic.end = IntLiteral(dim + neg_e, ic.end.span)
                    self.type_map[id(ic.end)] = I32
                start_type = self._infer(ic.start, env)
                end_type = self._infer(ic.end, env)
                # Bounds must be i32
                if start_type != I32:
                    self._error(
                        f"slice start must be i32, got {start_type}",
                        ic.start.span.line_start, ic.start.span.col_start,
                    )
                    return None
                if end_type != I32:
                    self._error(
                        f"slice end must be i32, got {end_type}",
                        ic.end.span.line_start, ic.end.span.col_start,
                    )
                    return None
                # Slice size must be statically determinable
                slice_size = _try_static_slice_size(ic.start, ic.end)
                if slice_size is None:
                    self._error(
                        "slice size must be statically determinable (e.g., x[i:i+3])",
                        ic.span.line_start, ic.span.col_start,
                    )
                    return None
                if slice_size <= 0:
                    self._error(
                        f"slice range is empty or negative (size={slice_size})",
                        ic.span.line_start, ic.span.col_start,
                    )
                    return None
                ic.static_size = slice_size
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
        if op == "-" and isinstance(t, StructType) and _struct_has_numeric_leaves(t):
            return t
        if op == "not":
            if t == BOOL or (isinstance(t, ArrayType) and t.base == "bool"):
                return t
            self._error(f"'not' requires bool operand, got {t}", expr.span.line_start, expr.span.col_start)
            return None
        self._error(f"invalid unary {op} on type {t}", expr.span.line_start, expr.span.col_start)
        return None

    def _check_binop(self, op: str, left: Expr, right: Expr, expr: Expr, env: TypeEnv) -> MaomiType | None:
        lt = self._infer(left, env)
        rt = self._infer(right, env)
        if lt is None or rt is None:
            return None

        if op == "@":
            return self._check_matmul(lt, rt, expr)

        if op in ("and", "or"):
            lt_bool = (lt == BOOL or (isinstance(lt, ArrayType) and lt.base == "bool"))
            rt_bool = (rt == BOOL or (isinstance(rt, ArrayType) and rt.base == "bool"))
            if not lt_bool or not rt_bool:
                self._error(
                    f"'{op}' requires bool operands, got {lt} and {rt}",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            result = self._broadcast(lt, rt)
            if result is None:
                self._error(
                    f"'{op}': mismatched shapes {lt} and {rt}",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            return result

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

        # Struct arithmetic
        if isinstance(lt, StructType) or isinstance(rt, StructType):
            return self._check_struct_binop(op, lt, rt, expr)

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

    def _check_struct_binop(self, op: str, lt: MaomiType, rt: MaomiType, expr: Expr) -> MaomiType | None:
        """Type-check arithmetic on struct types."""
        # struct OP struct (same type)
        if isinstance(lt, StructType) and isinstance(rt, StructType):
            if op not in {"+", "-", "*", "/"}:
                self._error(
                    f"operator {op}: not supported between struct types",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            if lt.name != rt.name:
                self._error(
                    f"operator {op}: mismatched struct types {lt.name} and {rt.name}",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            if not _struct_has_numeric_leaves(lt):
                self._error(
                    f"operator {op}: struct {lt.name} has non-numeric fields",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            return lt

        # scalar * struct / struct * scalar
        if op == "*":
            if isinstance(lt, StructType) and _is_numeric(rt) and isinstance(rt, ScalarType):
                if not _struct_has_numeric_leaves(lt):
                    self._error(f"operator *: struct {lt.name} has non-numeric fields", expr.span.line_start, expr.span.col_start)
                    return None
                return lt
            if _is_numeric(lt) and isinstance(lt, ScalarType) and isinstance(rt, StructType):
                if not _struct_has_numeric_leaves(rt):
                    self._error(f"operator *: struct {rt.name} has non-numeric fields", expr.span.line_start, expr.span.col_start)
                    return None
                return rt

        # struct / scalar
        if op == "/" and isinstance(lt, StructType) and _is_numeric(rt) and isinstance(rt, ScalarType):
            if not _struct_has_numeric_leaves(lt):
                self._error(f"operator /: struct {lt.name} has non-numeric fields", expr.span.line_start, expr.span.col_start)
                return None
            return lt

        # scalar + struct / struct + scalar (broadcast scalar to each field)
        if op in {"+", "-"}:
            if isinstance(lt, StructType) and _is_numeric(rt) and isinstance(rt, ScalarType):
                if not _struct_has_numeric_leaves(lt):
                    self._error(f"operator {op}: struct {lt.name} has non-numeric fields", expr.span.line_start, expr.span.col_start)
                    return None
                return lt
            if _is_numeric(lt) and isinstance(lt, ScalarType) and isinstance(rt, StructType):
                if not _struct_has_numeric_leaves(rt):
                    self._error(f"operator {op}: struct {rt.name} has non-numeric fields", expr.span.line_start, expr.span.col_start)
                    return None
                return rt

        self._error(
            f"operator {op}: unsupported for struct types ({lt} and {rt})",
            expr.span.line_start, expr.span.col_start,
        )
        return None

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

        # RNG builtins
        if expr.callee in _RNG_BUILTINS:
            return self._check_rng_call(expr, env)

        # sum/mean/max/min — reduction with optional axis
        if expr.callee in ("sum", "mean", "max", "min", "logsumexp", "prod"):
            return self._check_reduction(expr, env)

        # all/any — boolean reduction
        if expr.callee in ("all", "any"):
            return self._check_bool_reduction(expr, env)

        # argmax/argmin — returns i32 index
        if expr.callee in ("argmax", "argmin"):
            return self._check_argmax(expr, env)

        # reshape(array, dim1, dim2, ...) — variadic shape builtin
        if expr.callee == "reshape":
            return self._check_reshape(expr, env)

        # concat(arr1, arr2, ...) or concat(arr1, arr2, ..., axis)
        if expr.callee == "concat":
            return self._check_concat(expr, env)

        # expand_dims(x, axis) — insert size-1 dim
        if expr.callee == "expand_dims":
            return self._check_expand_dims(expr, env)

        # squeeze(x, axis) — remove size-1 dim
        if expr.callee == "squeeze":
            return self._check_squeeze(expr, env)

        # broadcast_to(x, d1, d2, ...) — broadcast to target shape
        if expr.callee == "broadcast_to":
            return self._check_broadcast_to(expr, env)

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

        # one_hot(index, n) — convert i32 indices to one-hot f32 vectors
        if expr.callee == "one_hot":
            return self._check_one_hot(expr, env)

        # zeros(d1, d2, ...) / ones(d1, d2, ...) — filled f32 arrays
        if expr.callee in ("zeros", "ones"):
            if len(expr.args) < 1:
                self._error(f"{expr.callee} expects at least 1 dimension argument", expr.span.line_start, expr.span.col_start)
                return None
            dims: list[int] = []
            for a in expr.args:
                if not isinstance(a, IntLiteral):
                    self._error(f"{expr.callee} dimension arguments must be integer literals", expr.span.line_start, expr.span.col_start)
                    return None
                if a.value <= 0:
                    self._error(f"{expr.callee} dimensions must be positive", expr.span.line_start, expr.span.col_start)
                    return None
                self._infer(a, env)
                dims.append(a.value)
            return ArrayType("f32", tuple(dims))

        # full(value, d1, d2, ...) — f32 array filled with value
        if expr.callee == "full":
            if len(expr.args) < 2:
                self._error("full expects a value and at least 1 dimension argument", expr.span.line_start, expr.span.col_start)
                return None
            val_type = self._infer(expr.args[0], env)
            if val_type is None:
                return None
            if not isinstance(val_type, ScalarType) or val_type.base != "f32":
                self._error("full value must be f32 scalar", expr.span.line_start, expr.span.col_start)
                return None
            dims_f: list[int] = []
            for a in expr.args[1:]:
                if not isinstance(a, IntLiteral):
                    self._error("full dimension arguments must be integer literals", expr.span.line_start, expr.span.col_start)
                    return None
                if a.value <= 0:
                    self._error("full dimensions must be positive", expr.span.line_start, expr.span.col_start)
                    return None
                self._infer(a, env)
                dims_f.append(a.value)
            return ArrayType("f32", tuple(dims_f))

        # arange(start, stop, step) — integer range
        if expr.callee == "arange":
            if len(expr.args) != 3:
                self._error("arange expects exactly 3 arguments (start, stop, step)", expr.span.line_start, expr.span.col_start)
                return None
            for i, name in enumerate(["start", "stop", "step"]):
                if not isinstance(expr.args[i], IntLiteral):
                    self._error(f"arange {name} must be an integer literal", expr.span.line_start, expr.span.col_start)
                    return None
                self._infer(expr.args[i], env)
            start_val = expr.args[0].value
            stop_val = expr.args[1].value
            step_val = expr.args[2].value
            if step_val == 0:
                self._error("arange step must not be zero", expr.span.line_start, expr.span.col_start)
                return None
            n = math.ceil((stop_val - start_val) / step_val)
            if n <= 0:
                self._error("arange produces empty range", expr.span.line_start, expr.span.col_start)
                return None
            return ArrayType("i32", (n,))

        # linspace(start, stop, n) — linearly spaced floats
        if expr.callee == "linspace":
            if len(expr.args) != 3:
                self._error("linspace expects exactly 3 arguments (start, stop, n)", expr.span.line_start, expr.span.col_start)
                return None
            start_f = _try_float_literal(expr.args[0])
            if start_f is None:
                self._error("linspace start must be a float literal", expr.span.line_start, expr.span.col_start)
                return None
            stop_f = _try_float_literal(expr.args[1])
            if stop_f is None:
                self._error("linspace stop must be a float literal", expr.span.line_start, expr.span.col_start)
                return None
            if not isinstance(expr.args[2], IntLiteral):
                self._error("linspace n must be an integer literal", expr.span.line_start, expr.span.col_start)
                return None
            if expr.args[2].value <= 0:
                self._error("linspace n must be positive", expr.span.line_start, expr.span.col_start)
                return None
            for a in expr.args:
                self._infer(a, env)
            return ArrayType("f32", (expr.args[2].value,))

        # eye(n) / eye(n, m) — identity matrix
        if expr.callee == "eye":
            if len(expr.args) < 1 or len(expr.args) > 2:
                self._error("eye expects 1 or 2 arguments", expr.span.line_start, expr.span.col_start)
                return None
            for i, a in enumerate(expr.args):
                if not isinstance(a, IntLiteral):
                    self._error("eye arguments must be integer literals", expr.span.line_start, expr.span.col_start)
                    return None
                if a.value <= 0:
                    self._error("eye dimensions must be positive", expr.span.line_start, expr.span.col_start)
                    return None
                self._infer(a, env)
            n_val = expr.args[0].value
            m_val = expr.args[1].value if len(expr.args) == 2 else n_val
            return ArrayType("f32", (n_val, m_val))

        # Linear algebra builtins
        if expr.callee in ("cholesky", "triangular_solve"):
            return self._check_linalg(expr, env)

        # conv2d(input, kernel, ...) — 2D convolution
        if expr.callee == "conv2d":
            return self._check_conv2d(expr, env)

        # max_pool / avg_pool(input, wh, ww, sh, sw)
        if expr.callee in ("max_pool", "avg_pool"):
            return self._check_pool(expr, env)

        # stop_gradient(expr) — identity, prevents gradient flow
        if expr.callee == "stop_gradient":
            if len(expr.args) != 1:
                self._error("stop_gradient expects exactly 1 argument", expr.span.line_start, expr.span.col_start)
                return None
            return self._infer(expr.args[0], env)

        # where(cond, x, y) — element-wise conditional
        if expr.callee == "where":
            return self._check_where(expr, env)

        # clip(x, lo, hi) — clamp values to range
        if expr.callee == "clip":
            return self._check_clip(expr, env)

        # transpose(matrix) — swap dims of a 2D array
        if expr.callee == "transpose":
            return self._check_transpose(expr, env)

        # isfinite(x) — returns bool with same shape
        if expr.callee == "isfinite":
            return self._check_isfinite(expr, env)

        # zeros_like(x) / ones_like(x) — same type filled with 0 or 1
        if expr.callee in ("zeros_like", "ones_like"):
            return self._check_like(expr, env)

        # maximum(x, y), minimum(x, y), pow(x, y), atan2(y, x), logaddexp, hypot, remainder, copysign — two-arg elementwise
        if expr.callee in ("maximum", "minimum", "pow", "atan2", "logaddexp", "hypot", "remainder", "copysign"):
            return self._check_two_arg_elementwise(expr, env)
        # stack(a, b, ..., axis) — stack arrays along new axis
        if expr.callee == "stack":
            return self._check_stack(expr, env)

        # pad(x, val, pad_lo, pad_hi) — pad array with constant value
        if expr.callee == "pad":
            return self._check_pad(expr, env)
        # einsum("spec", a, b) — Einstein summation
        if expr.callee == "einsum":
            return self._check_einsum(expr, env)

        # cumsum(x, axis=N), cumprod(x, axis=N) — cumulative reductions
        if expr.callee in ("cumsum", "cumprod"):
            return self._check_cumulative(expr, env)

        # sort(x), sort(x, axis=N), argsort(x), argsort(x, axis=N)
        if expr.callee in ("sort", "argsort"):
            return self._check_sort(expr, env)

        # flip(x, axis) — reverse along axis
        if expr.callee == "flip":
            return self._check_flip(expr, env)

        # tril(x), triu(x) — triangular masks
        if expr.callee in ("tril", "triu"):
            return self._check_tril_triu(expr, env)

        sig = self.fn_table.get(expr.callee)
        if sig is None:
            # Check for generic (wildcard) function
            if expr.callee in self._generic_fns:
                return self._check_generic_call(expr, env)
            self._error(
                f"undefined function: '{expr.callee}'",
                expr.span.line_start,
                expr.span.col_start,
            )
            return None

        # Resolve named arguments to positional
        self._resolve_named_args(expr, sig.param_names)

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

            if isinstance(arg_type, StringType):
                self._error(
                    "string literals can only be used as callback arguments",
                    arg.span.line_start,
                    arg.span.col_start,
                )
                continue

            # Elementwise builtins on struct types — skip normal unification
            if expr.callee in _ELEMENTWISE_BUILTINS and isinstance(arg_type, StructType):
                continue

            # Elementwise builtins accept any float type — skip normal unification for non-f32 floats
            if expr.callee in _ELEMENTWISE_BUILTINS:
                arg_base = arg_type.base if isinstance(arg_type, (ScalarType, ArrayType)) else None
                if arg_base in FLOAT_BASES and arg_base != "f32":
                    continue

            if not self._unify_arg(arg_type, param_type, substitution, expr, i):
                continue

        # Apply substitution to return type
        ret = self._apply_substitution(sig.return_type, substitution)

        # For elementwise builtins (scalar->scalar), lift return to match arg type
        if expr.callee in _ELEMENTWISE_BUILTINS and isinstance(ret, ScalarType):
            for at in arg_types:
                if isinstance(at, ArrayType) and at.base in FLOAT_BASES:
                    ret = ArrayType(at.base, at.dims)
                    break
                if isinstance(at, ScalarType) and at.base in FLOAT_BASES and at.base != "f32":
                    ret = ScalarType(at.base)
                    break
                if isinstance(at, StructType):
                    # Struct-level builtin: apply field-by-field
                    if not _struct_has_float_leaves(at):
                        self._error(
                            f"builtin '{expr.callee}' on struct {at.name}: all leaf fields must be float",
                            expr.span.line_start, expr.span.col_start,
                        )
                        return None
                    ret = at
                    break

        return ret

    def _check_reduction(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        """Check sum(x) / sum(x, axis) / sum(x, axis=1, keepdims=true) etc."""
        self._resolve_named_args(expr, ["x", "axis", "keepdims"])

        if len(expr.args) < 1 or len(expr.args) > 3:
            self._error(
                f"{expr.callee} expects 1 to 3 arguments",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        arg_type = self._infer(expr.args[0], env)
        if arg_type is None:
            return None

        # Lift scalar arg: sum/mean of scalar is itself
        if isinstance(arg_type, ScalarType):
            if len(expr.args) >= 2:
                self._error(
                    f"{expr.callee} with axis requires an array argument",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            return arg_type

        if not isinstance(arg_type, ArrayType):
            self._error(
                f"{expr.callee} requires a numeric argument",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # Check keepdims (3rd arg)
        keepdims = False
        if len(expr.args) == 3:
            kd_arg = expr.args[2]
            self._infer(kd_arg, env)
            if not isinstance(kd_arg, BoolLiteral):
                self._error(
                    f"{expr.callee} keepdims must be true or false",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            keepdims = kd_arg.value

        if len(expr.args) == 1:
            # Reduce all dims → scalar
            return ScalarType(arg_type.base)

        # 2+ args: axis-specific reduction
        axis_arg = expr.args[1]
        self._infer(axis_arg, env)
        if not isinstance(axis_arg, IntLiteral):
            self._error(
                f"{expr.callee} axis must be an integer literal",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        axis = axis_arg.value
        ndim = len(arg_type.dims)
        if axis < 0 or axis >= ndim:
            self._error(
                f"{expr.callee} axis {axis} out of range for {ndim}D array",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        if keepdims:
            # Keep reduced dim as size 1
            new_dims = tuple(1 if i == axis else d for i, d in enumerate(arg_type.dims))
        else:
            # Remove the reduced dimension
            new_dims = tuple(d for i, d in enumerate(arg_type.dims) if i != axis)
        if len(new_dims) == 0:
            return ScalarType(arg_type.base)
        return ArrayType(arg_type.base, new_dims)

    def _check_bool_reduction(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        """Check all(x) / all(x, axis) / any(x) / any(x, axis) — boolean reductions."""
        self._resolve_named_args(expr, ["x", "axis"])

        if len(expr.args) < 1 or len(expr.args) > 2:
            self._error(
                f"{expr.callee} expects 1 or 2 arguments",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        arg_type = self._infer(expr.args[0], env)
        if arg_type is None:
            return None

        # Input must be bool
        if isinstance(arg_type, ScalarType):
            if arg_type.base != "bool":
                self._error(
                    f"{expr.callee} requires a bool argument, got {arg_type.base}",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            if len(expr.args) >= 2:
                self._error(
                    f"{expr.callee} with axis requires an array argument",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            return arg_type

        if not isinstance(arg_type, ArrayType) or arg_type.base != "bool":
            self._error(
                f"{expr.callee} requires a bool array argument, got {arg_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        if len(expr.args) == 1:
            # Reduce all dims -> bool scalar
            return ScalarType("bool")

        # Axis-specific reduction
        axis_arg = expr.args[1]
        self._infer(axis_arg, env)
        if not isinstance(axis_arg, IntLiteral):
            self._error(
                f"{expr.callee} axis must be an integer literal",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        axis = axis_arg.value
        ndim = len(arg_type.dims)
        if axis < 0 or axis >= ndim:
            self._error(
                f"{expr.callee} axis {axis} out of range for {ndim}D array",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        new_dims = tuple(d for i, d in enumerate(arg_type.dims) if i != axis)
        if len(new_dims) == 0:
            return ScalarType("bool")
        return ArrayType("bool", new_dims)

    def _check_argmax(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        """Check argmax(x) or argmax(x, axis) — returns i32 indices."""
        if len(expr.args) < 1 or len(expr.args) > 2:
            self._error(
                f"{expr.callee} expects 1 or 2 arguments",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        arg_type = self._infer(expr.args[0], env)
        if arg_type is None:
            return None

        if not isinstance(arg_type, ArrayType):
            self._error(
                f"{expr.callee} requires an array argument",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        if len(expr.args) == 1:
            # All-dims argmax → scalar i32
            return ScalarType("i32")

        # 2 args: axis-specific
        axis_arg = expr.args[1]
        self._infer(axis_arg, env)
        if not isinstance(axis_arg, IntLiteral):
            self._error(
                f"{expr.callee} axis must be an integer literal",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        axis = axis_arg.value
        ndim = len(arg_type.dims)
        if axis < 0 or axis >= ndim:
            self._error(
                f"{expr.callee} axis {axis} out of range for {ndim}D array",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # Remove the reduced dimension, result is i32
        new_dims = tuple(d for i, d in enumerate(arg_type.dims) if i != axis)
        if len(new_dims) == 0:
            return ScalarType("i32")
        return ArrayType("i32", new_dims)

    def _check_where(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        if len(expr.args) != 3:
            self._error(
                "where expects exactly 3 arguments: where(cond, x, y)",
                expr.span.line_start, expr.span.col_start,
            )
            return None
        cond_type = self._infer(expr.args[0], env)
        x_type = self._infer(expr.args[1], env)
        y_type = self._infer(expr.args[2], env)
        if cond_type is None or x_type is None or y_type is None:
            return None

        # Condition must be bool
        cond_base = _base_of(cond_type) if cond_type else None
        if cond_base != "bool":
            self._error(
                f"where condition must be bool, got {cond_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # x and y must be broadcastable
        result_type = self._broadcast(x_type, y_type)
        if result_type is None:
            self._error(
                f"where branches must be compatible types, got {x_type} and {y_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        return result_type

    def _check_isfinite(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        if len(expr.args) != 1:
            self._error("isfinite expects exactly 1 argument", expr.span.line_start, expr.span.col_start)
            return None
        arg_type = self._infer(expr.args[0], env)
        if arg_type is None:
            return None
        if not isinstance(arg_type, (ScalarType, ArrayType)) or arg_type.base not in ("f32", "f64"):
            self._error(f"isfinite requires a float scalar or array, got {arg_type}", expr.span.line_start, expr.span.col_start)
            return None
        if isinstance(arg_type, ScalarType):
            return BOOL
        return ArrayType("bool", arg_type.dims)

    def _check_like(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        if len(expr.args) != 1:
            self._error(f"{expr.callee} expects exactly 1 argument", expr.span.line_start, expr.span.col_start)
            return None
        arg_type = self._infer(expr.args[0], env)
        if arg_type is None:
            return None
        if isinstance(arg_type, (ScalarType, ArrayType)):
            if arg_type.base not in ("f32", "f64"):
                self._error(f"{expr.callee} requires a float type, got {arg_type}", expr.span.line_start, expr.span.col_start)
                return None
            return arg_type
        self._error(f"{expr.callee} requires a float scalar or array, got {arg_type}", expr.span.line_start, expr.span.col_start)
        return None

    def _check_two_arg_elementwise(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        """Check maximum(x, y), minimum(x, y), pow(x, y) — two-arg elementwise builtins."""
        if len(expr.args) != 2:
            self._error(
                f"{expr.callee} expects exactly 2 arguments",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        x_type = self._infer(expr.args[0], env)
        y_type = self._infer(expr.args[1], env)
        if x_type is None or y_type is None:
            return None

        # Both must be float types
        x_base = _base_of(x_type)
        y_base = _base_of(y_type)
        if x_base not in ("f32", "f64"):
            self._error(
                f"{expr.callee} arguments must be float, got {x_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None
        if y_base not in ("f32", "f64"):
            self._error(
                f"{expr.callee} arguments must be float, got {y_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # Broadcast types
        result_type = self._broadcast(x_type, y_type)
        if result_type is None:
            self._error(
                f"{expr.callee} arguments must have compatible shapes, got {x_type} and {y_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        return result_type

    def _check_clip(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        if len(expr.args) != 3:
            self._error(
                "clip expects exactly 3 arguments: clip(x, lo, hi)",
                expr.span.line_start, expr.span.col_start,
            )
            return None
        x_type = self._infer(expr.args[0], env)
        lo_type = self._infer(expr.args[1], env)
        hi_type = self._infer(expr.args[2], env)
        if x_type is None or lo_type is None or hi_type is None:
            return None

        # All three must be float
        for arg_type, name in [(x_type, "x"), (lo_type, "lo"), (hi_type, "hi")]:
            base = _base_of(arg_type) if arg_type else None
            if base not in ("f32", "f64"):
                self._error(
                    f"clip argument '{name}' must be float type, got {arg_type}",
                    expr.span.line_start, expr.span.col_start,
                )
                return None

        # Broadcast all three together
        result_type = self._broadcast(x_type, lo_type)
        if result_type is None:
            self._error(
                f"clip arguments must be broadcastable, got {x_type} and {lo_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None
        result_type = self._broadcast(result_type, hi_type)
        if result_type is None:
            self._error(
                f"clip arguments must be broadcastable, got incompatible shapes",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        return result_type

    def _check_einsum(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        """Check einsum("spec", a, b) or einsum("spec", a)."""
        if len(expr.args) < 2:
            self._error(
                "einsum expects at least 2 arguments: einsum(spec, array, ...)",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # First arg must be a string literal
        spec_arg = expr.args[0]
        if not isinstance(spec_arg, StringLiteral):
            self._error(
                "einsum first argument must be a string literal spec, e.g. \"ij,jk->ik\"",
                expr.span.line_start, expr.span.col_start,
            )
            return None
        self._infer(spec_arg, env)

        spec = spec_arg.value

        # Parse the spec: "inputs->output"
        if "->" not in spec:
            self._error(
                f"einsum spec must contain '->' separator, got \"{spec}\"",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        lhs, rhs = spec.split("->", 1)
        input_subscripts = lhs.split(",")
        output_subscript = rhs

        # Number of input subscripts must match number of array args
        num_arrays = len(expr.args) - 1
        if len(input_subscripts) != num_arrays:
            self._error(
                f"einsum spec has {len(input_subscripts)} input(s) but got {num_arrays} array argument(s)",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # Infer types for all array args
        arg_types: list[MaomiType | None] = []
        for arg in expr.args[1:]:
            arg_types.append(self._infer(arg, env))

        # Validate all args are arrays with matching base types
        base_type: str | None = None
        for i, (at, sub) in enumerate(zip(arg_types, input_subscripts)):
            if at is None:
                return None
            if not isinstance(at, ArrayType):
                self._error(
                    f"einsum: argument {i+1} must be an array, got {at}",
                    expr.args[i+1].span.line_start, expr.args[i+1].span.col_start,
                )
                return None
            if base_type is None:
                base_type = at.base
            elif at.base != base_type:
                self._error(
                    f"einsum: all arrays must have the same base type, got {base_type} and {at.base}",
                    expr.args[i+1].span.line_start, expr.args[i+1].span.col_start,
                )
                return None
            # Check rank matches subscript length
            if len(at.dims) != len(sub):
                self._error(
                    f"einsum: subscript \"{sub}\" has {len(sub)} indices but argument {i+1} has rank {len(at.dims)}",
                    expr.args[i+1].span.line_start, expr.args[i+1].span.col_start,
                )
                return None

        # Build dim map: each subscript char -> dimension size
        dim_map: dict[str, int | str] = {}
        for at, sub in zip(arg_types, input_subscripts):
            assert isinstance(at, ArrayType)
            for ch, dim in zip(sub, at.dims):
                if ch in dim_map:
                    existing = dim_map[ch]
                    if isinstance(existing, int) and isinstance(dim, int) and existing != dim:
                        self._error(
                            f"einsum: dimension mismatch for index '{ch}': {existing} vs {dim}",
                            expr.span.line_start, expr.span.col_start,
                        )
                        return None
                else:
                    dim_map[ch] = dim

        # Validate output subscript chars all appear in inputs
        for ch in output_subscript:
            if ch not in dim_map:
                self._error(
                    f"einsum: output index '{ch}' does not appear in any input subscript",
                    expr.span.line_start, expr.span.col_start,
                )
                return None

        # Build output shape
        assert base_type is not None
        output_dims = tuple(dim_map[ch] for ch in output_subscript)
        if len(output_dims) == 0:
            return ScalarType(base_type)
        return ArrayType(base_type, output_dims)

    def _check_cumulative(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        """Check cumsum(x, axis=N), cumprod(x, axis=N)."""
        name = expr.callee
        self._resolve_named_args(expr, ["x", "axis"])
        args = expr.args
        if len(args) != 2:
            self._error(
                f"{name} expects 2 arguments: {name}(x, axis=N)",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        x_type = self._infer(args[0], env)
        if x_type is None:
            return None
        if not isinstance(x_type, ArrayType):
            self._error(
                f"{name} requires an array argument, got {x_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None
        if x_type.base not in ("f32", "f64"):
            self._error(
                f"{name} requires float array, got {x_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        axis_node = args[1]
        self._infer(axis_node, env)
        if isinstance(axis_node, IntLiteral):
            axis_val = axis_node.value
        else:
            neg = _try_negative_literal(axis_node)
            if neg is not None:
                axis_val = neg
            else:
                self._error(
                    f"{name}: axis must be an integer literal",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
        ndim = len(x_type.dims)
        if axis_val < 0:
            axis_val += ndim
        if axis_val < 0 or axis_val >= ndim:
            self._error(
                f"{name}: axis {axis_val} out of range for {ndim}-D array",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        return x_type  # Same shape as input

    def _check_sort(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        """Check sort(x), sort(x, axis=N), argsort(x), argsort(x, axis=N)."""
        name = expr.callee
        self._resolve_named_args(expr, ["x", "axis"])
        args = expr.args
        if len(args) < 1 or len(args) > 2:
            self._error(
                f"{name} expects 1-2 arguments: {name}(x) or {name}(x, axis=N)",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        x_type = self._infer(args[0], env)
        if x_type is None:
            return None
        if not isinstance(x_type, ArrayType):
            self._error(
                f"{name} requires an array argument, got {x_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # Determine axis (default: last axis)
        ndim = len(x_type.dims)
        if len(args) == 2:
            axis_node = args[1]
            self._infer(axis_node, env)
            if isinstance(axis_node, IntLiteral):
                axis_val = axis_node.value
            else:
                neg = _try_negative_literal(axis_node)
                if neg is not None:
                    axis_val = neg
                else:
                    self._error(
                        f"{name}: axis must be an integer literal",
                        expr.span.line_start, expr.span.col_start,
                    )
                    return None
        else:
            axis_val = -1  # default: last axis

        if axis_val < 0:
            axis_val += ndim
        if axis_val < 0 or axis_val >= ndim:
            self._error(
                f"{name}: axis {axis_val} out of range for {ndim}-D array",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        if name == "argsort":
            return ArrayType("i32", x_type.dims)
        return x_type  # Same shape and type as input

    def _check_flip(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        """Check flip(x, axis) — reverse array along an axis."""
        self._resolve_named_args(expr, ["x", "axis"])
        args = expr.args
        if len(args) != 2:
            self._error(
                "flip expects exactly 2 arguments: flip(x, axis)",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        x_type = self._infer(args[0], env)
        if x_type is None:
            return None
        if not isinstance(x_type, ArrayType):
            self._error(
                f"flip requires an array argument, got {x_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        axis_node = args[1]
        self._infer(axis_node, env)
        if isinstance(axis_node, IntLiteral):
            axis_val = axis_node.value
        else:
            neg = _try_negative_literal(axis_node)
            if neg is not None:
                axis_val = neg
            else:
                self._error(
                    "flip: axis must be an integer literal",
                    expr.span.line_start, expr.span.col_start,
                )
                return None

        ndim = len(x_type.dims)
        if axis_val < 0:
            axis_val += ndim
        if axis_val < 0 or axis_val >= ndim:
            self._error(
                f"flip: axis {axis_val} out of range for {ndim}-D array",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        return x_type  # Same shape and type as input

    def _check_tril_triu(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        """Check tril(x) / triu(x) — triangular matrix masks. Input must be 2D float."""
        name = expr.callee
        args = expr.args
        if len(args) != 1:
            self._error(
                f"{name} expects exactly 1 argument: {name}(x)",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        x_type = self._infer(args[0], env)
        if x_type is None:
            return None
        if not isinstance(x_type, ArrayType):
            self._error(
                f"{name} requires a 2D array argument, got {x_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None
        if len(x_type.dims) != 2:
            self._error(
                f"{name} requires a 2D array, got {len(x_type.dims)}-D array",
                expr.span.line_start, expr.span.col_start,
            )
            return None
        if x_type.base not in FLOAT_BASES:
            self._error(
                f"{name} requires a float array, got {x_type.base}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        return x_type  # Same shape and type as input

    def _check_transpose(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        if len(expr.args) < 1:
            self._error(
                "transpose expects at least 1 argument: transpose(array) or transpose(array, axes...)",
                expr.span.line_start, expr.span.col_start,
            )
            return None
        arg_type = self._infer(expr.args[0], env)
        if arg_type is None:
            return None
        if not isinstance(arg_type, ArrayType):
            self._error(
                f"transpose requires an array, got {arg_type}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        rank = len(arg_type.dims)
        if len(expr.args) == 1:
            # Shorthand: transpose(x) — requires 2D, swaps axes
            if rank != 2:
                self._error(
                    f"transpose(x) shorthand requires a 2D array, got {rank}D. Use transpose(x, axes...) for higher ranks.",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            perm = (1, 0)
        else:
            # General: transpose(x, ax0, ax1, ...)
            axes = expr.args[1:]
            if len(axes) != rank:
                self._error(
                    f"transpose permutation must have {rank} axes for a {rank}D array, got {len(axes)}",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            perm_list = []
            for ax in axes:
                if not isinstance(ax, IntLiteral):
                    self._error(
                        "transpose axes must be integer literals",
                        ax.span.line_start, ax.span.col_start,
                    )
                    return None
                if ax.value < 0 or ax.value >= rank:
                    self._error(
                        f"transpose axis {ax.value} out of range for {rank}D array",
                        ax.span.line_start, ax.span.col_start,
                    )
                    return None
                perm_list.append(ax.value)
            if sorted(perm_list) != list(range(rank)):
                self._error(
                    f"transpose axes must be a permutation of [0..{rank-1}], got {perm_list}",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            perm = tuple(perm_list)

        new_dims = tuple(arg_type.dims[p] for p in perm)
        return ArrayType(arg_type.base, new_dims)

    def _check_one_hot(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        if len(expr.args) != 2:
            self._error("one_hot expects exactly 2 arguments (index, n)", expr.span.line_start, expr.span.col_start)
            return None

        index_type = self._infer(expr.args[0], env)
        if index_type is None:
            return None

        # index must be i32 (scalar or array)
        is_i32 = (isinstance(index_type, (ScalarType, ArrayType))
                  and index_type.base == "i32")
        if not is_i32:
            self._error("one_hot index must be i32", expr.span.line_start, expr.span.col_start)
            return None

        # n must be positive integer literal
        n_arg = expr.args[1]
        if not isinstance(n_arg, IntLiteral):
            self._error("one_hot n must be an integer literal", expr.span.line_start, expr.span.col_start)
            return None
        if n_arg.value <= 0:
            self._error("one_hot n must be positive", expr.span.line_start, expr.span.col_start)
            return None
        self._infer(n_arg, env)

        n = n_arg.value
        prefix_dims = index_type.dims if isinstance(index_type, ArrayType) else ()
        return ArrayType("f32", prefix_dims + (n,))

    def _check_reshape(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        if len(expr.args) < 2:
            self._error(
                "reshape requires at least 2 arguments: array and target dimensions",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        arg_type = self._infer(expr.args[0], env)
        if arg_type is None:
            return None
        if not isinstance(arg_type, ArrayType):
            self._error(
                "reshape: first argument must be an array",
                expr.args[0].span.line_start, expr.args[0].span.col_start,
            )
            return None

        # Remaining args must be positive integer literals
        target_dims: list[int] = []
        for i, dim_arg in enumerate(expr.args[1:], 1):
            self._infer(dim_arg, env)
            if not isinstance(dim_arg, IntLiteral):
                self._error(
                    "reshape: dimension arguments must be integer literals",
                    dim_arg.span.line_start, dim_arg.span.col_start,
                )
                return None
            if dim_arg.value <= 0:
                self._error(
                    "reshape: dimensions must be positive",
                    dim_arg.span.line_start, dim_arg.span.col_start,
                )
                return None
            target_dims.append(dim_arg.value)

        # Validate element count
        input_numel = 1
        for d in arg_type.dims:
            if not isinstance(d, int):
                self._error(
                    f"reshape: cannot reshape with symbolic dimension '{d}'",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            input_numel *= d
        output_numel = 1
        for d in target_dims:
            output_numel *= d
        if input_numel != output_numel:
            self._error(
                f"reshape: input has {input_numel} elements but target shape has {output_numel}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        return ArrayType(arg_type.base, tuple(target_dims))

    def _check_expand_dims(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        if len(expr.args) != 2:
            self._error(
                "expand_dims requires exactly 2 arguments: array and axis",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        arg_type = self._infer(expr.args[0], env)
        if arg_type is None:
            return None

        # Determine current dims (scalars have no dims)
        if isinstance(arg_type, ScalarType):
            current_dims: tuple[int, ...] = ()
            base = arg_type.base
        elif isinstance(arg_type, ArrayType):
            current_dims = arg_type.dims
            base = arg_type.base
        else:
            self._error(
                "expand_dims: first argument must be a scalar or array",
                expr.args[0].span.line_start, expr.args[0].span.col_start,
            )
            return None

        axis_arg = expr.args[1]
        self._infer(axis_arg, env)
        if not isinstance(axis_arg, IntLiteral):
            self._error(
                "expand_dims: axis must be an integer literal",
                axis_arg.span.line_start, axis_arg.span.col_start,
            )
            return None

        ndim = len(current_dims)
        axis = axis_arg.value
        if axis < 0 or axis > ndim:
            self._error(
                f"expand_dims: axis {axis} out of range for {ndim}D array (must be in [0, {ndim}])",
                axis_arg.span.line_start, axis_arg.span.col_start,
            )
            return None

        new_dims = current_dims[:axis] + (1,) + current_dims[axis:]
        return ArrayType(base, new_dims)

    def _check_squeeze(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        if len(expr.args) != 2:
            self._error(
                "squeeze requires exactly 2 arguments: array and axis",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        arg_type = self._infer(expr.args[0], env)
        if arg_type is None:
            return None
        if not isinstance(arg_type, ArrayType):
            self._error(
                "squeeze: first argument must be an array",
                expr.args[0].span.line_start, expr.args[0].span.col_start,
            )
            return None

        axis_arg = expr.args[1]
        self._infer(axis_arg, env)
        if not isinstance(axis_arg, IntLiteral):
            self._error(
                "squeeze: axis must be an integer literal",
                axis_arg.span.line_start, axis_arg.span.col_start,
            )
            return None

        ndim = len(arg_type.dims)
        axis = axis_arg.value
        if axis < 0 or axis >= ndim:
            self._error(
                f"squeeze: axis {axis} out of range for {ndim}D array (must be in [0, {ndim - 1}])",
                axis_arg.span.line_start, axis_arg.span.col_start,
            )
            return None

        if arg_type.dims[axis] != 1:
            self._error(
                f"squeeze: dimension at axis {axis} has size {arg_type.dims[axis]}, must be size 1",
                axis_arg.span.line_start, axis_arg.span.col_start,
            )
            return None

        new_dims = arg_type.dims[:axis] + arg_type.dims[axis + 1:]
        if len(new_dims) == 0:
            return ScalarType(arg_type.base)
        return ArrayType(arg_type.base, new_dims)

    def _check_broadcast_to(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        if len(expr.args) < 2:
            self._error(
                "broadcast_to requires at least 2 arguments: array and target dimensions",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        arg_type = self._infer(expr.args[0], env)
        if arg_type is None:
            return None

        if isinstance(arg_type, ScalarType):
            src_dims: tuple[int, ...] = ()
            base = arg_type.base
        elif isinstance(arg_type, ArrayType):
            src_dims = arg_type.dims
            base = arg_type.base
        else:
            self._error(
                "broadcast_to: first argument must be a scalar or array",
                expr.args[0].span.line_start, expr.args[0].span.col_start,
            )
            return None

        # Remaining args must be positive integer literals
        target_dims: list[int] = []
        for dim_arg in expr.args[1:]:
            self._infer(dim_arg, env)
            if not isinstance(dim_arg, IntLiteral):
                self._error(
                    "broadcast_to: dimension arguments must be integer literals",
                    dim_arg.span.line_start, dim_arg.span.col_start,
                )
                return None
            if dim_arg.value <= 0:
                self._error(
                    "broadcast_to: dimensions must be positive",
                    dim_arg.span.line_start, dim_arg.span.col_start,
                )
                return None
            target_dims.append(dim_arg.value)

        # Validate broadcast compatibility
        target_rank = len(target_dims)
        src_rank = len(src_dims)
        if src_rank > target_rank:
            self._error(
                f"broadcast_to: source has rank {src_rank} but target has rank {target_rank}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # Right-align src_dims with target_dims and check compatibility
        offset = target_rank - src_rank
        for i, sd in enumerate(src_dims):
            td = target_dims[offset + i]
            if sd != 1 and sd != td:
                self._error(
                    f"broadcast_to: source dimension {sd} at axis {i} is not compatible with target dimension {td}",
                    expr.span.line_start, expr.span.col_start,
                )
                return None

        return ArrayType(base, tuple(target_dims))

    def _check_concat(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        if len(expr.args) < 2:
            self._error(
                "concat requires at least 2 arguments",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # Infer all arg types
        arg_types: list[MaomiType | None] = []
        for arg in expr.args:
            arg_types.append(self._infer(arg, env))

        if any(t is None for t in arg_types):
            return None

        # Detect axis: if last arg is IntLiteral, it's the axis
        if isinstance(expr.args[-1], IntLiteral) and isinstance(arg_types[-1], ScalarType):
            axis = expr.args[-1].value
            array_args = expr.args[:-1]
            array_types = arg_types[:-1]
        else:
            axis = 0
            array_args = expr.args
            array_types = arg_types

        if len(array_args) < 2:
            self._error(
                "concat requires at least 2 array arguments",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # All must be arrays
        for i, (arg, t) in enumerate(zip(array_args, array_types)):
            if not isinstance(t, ArrayType):
                self._error(
                    f"concat: argument {i} must be an array, got {t}",
                    arg.span.line_start, arg.span.col_start,
                )
                return None

        first = array_types[0]
        assert isinstance(first, ArrayType)
        rank = len(first.dims)

        if axis < 0 or axis >= rank:
            self._error(
                f"concat: axis {axis} out of range for rank-{rank} arrays",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        for i, t in enumerate(array_types[1:], 1):
            assert isinstance(t, ArrayType)
            if t.base != first.base:
                self._error(
                    f"concat: base type mismatch at argument {i}: {first.base} vs {t.base}",
                    array_args[i].span.line_start, array_args[i].span.col_start,
                )
                return None
            if len(t.dims) != rank:
                self._error(
                    f"concat: rank mismatch at argument {i}: expected {rank}, got {len(t.dims)}",
                    array_args[i].span.line_start, array_args[i].span.col_start,
                )
                return None
            for j in range(rank):
                if j != axis and t.dims[j] != first.dims[j]:
                    self._error(
                        f"concat: dimension mismatch at axis {j} for argument {i}: {first.dims[j]} vs {t.dims[j]}",
                        array_args[i].span.line_start, array_args[i].span.col_start,
                    )
                    return None

        # Sum concat axis dims (must be concrete)
        axis_sum = 0
        for i, t in enumerate(array_types):
            assert isinstance(t, ArrayType)
            d = t.dims[axis]
            if not isinstance(d, int):
                self._error(
                    f"concat: cannot concat with symbolic dimension '{d}' on axis {axis}",
                    array_args[i].span.line_start, array_args[i].span.col_start,
                )
                return None
            axis_sum += d

        result_dims = list(first.dims)
        result_dims[axis] = axis_sum
        return ArrayType(first.base, tuple(result_dims))

    def _check_stack(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        # stack(a, b, ..., axis) — at least 3 args (2 arrays + axis)
        if len(expr.args) < 3:
            self._error(
                "stack requires at least 3 arguments: two or more arrays and an axis",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # Infer all arg types
        arg_types: list[MaomiType | None] = []
        for arg in expr.args:
            arg_types.append(self._infer(arg, env))

        if any(t is None for t in arg_types):
            return None

        # Last arg must be int literal (axis)
        if not isinstance(expr.args[-1], IntLiteral):
            self._error(
                "stack: last argument (axis) must be an integer literal",
                expr.args[-1].span.line_start, expr.args[-1].span.col_start,
            )
            return None

        axis = expr.args[-1].value
        array_args = expr.args[:-1]
        array_types = arg_types[:-1]

        # All must be arrays
        for i, (arg, t) in enumerate(zip(array_args, array_types)):
            if not isinstance(t, ArrayType):
                self._error(
                    f"stack: argument {i} must be an array, got {t}",
                    arg.span.line_start, arg.span.col_start,
                )
                return None

        first = array_types[0]
        assert isinstance(first, ArrayType)
        rank = len(first.dims)

        # axis must be in [0, rank] (can insert at end)
        if axis < 0 or axis > rank:
            self._error(
                f"stack: axis {axis} out of range for rank-{rank} arrays (valid: 0..{rank})",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # All arrays must have the same shape and base type
        for i, t in enumerate(array_types[1:], 1):
            assert isinstance(t, ArrayType)
            if t.base != first.base:
                self._error(
                    f"stack: base type mismatch at argument {i}: {first.base} vs {t.base}",
                    array_args[i].span.line_start, array_args[i].span.col_start,
                )
                return None
            if t.dims != first.dims:
                self._error(
                    f"stack: shape mismatch at argument {i}: {first.dims} vs {t.dims}",
                    array_args[i].span.line_start, array_args[i].span.col_start,
                )
                return None

        # Result: insert dim of size n_arrays at axis position
        n_arrays = len(array_args)
        result_dims = list(first.dims)
        result_dims.insert(axis, n_arrays)
        return ArrayType(first.base, tuple(result_dims))

    def _check_pad(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        # pad(x, val, pad_lo, pad_hi) — exactly 4 args
        if len(expr.args) != 4:
            self._error(
                "pad expects exactly 4 arguments: pad(x, val, pad_lo, pad_hi)",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        x_type = self._infer(expr.args[0], env)
        val_type = self._infer(expr.args[1], env)
        self._infer(expr.args[2], env)
        self._infer(expr.args[3], env)

        if x_type is None or val_type is None:
            return None

        if not isinstance(x_type, ArrayType):
            self._error(
                f"pad: first argument must be an array, got {x_type}",
                expr.args[0].span.line_start, expr.args[0].span.col_start,
            )
            return None

        if not isinstance(val_type, ScalarType) or val_type.base not in ("f32", "f64"):
            self._error(
                "pad: second argument (padding value) must be a float scalar",
                expr.args[1].span.line_start, expr.args[1].span.col_start,
            )
            return None

        if not isinstance(expr.args[2], IntLiteral):
            self._error(
                "pad: pad_lo must be an integer literal",
                expr.args[2].span.line_start, expr.args[2].span.col_start,
            )
            return None

        if not isinstance(expr.args[3], IntLiteral):
            self._error(
                "pad: pad_hi must be an integer literal",
                expr.args[3].span.line_start, expr.args[3].span.col_start,
            )
            return None

        pad_lo = expr.args[2].value
        pad_hi = expr.args[3].value

        if pad_lo < 0:
            self._error(
                "pad: pad_lo must be non-negative",
                expr.args[2].span.line_start, expr.args[2].span.col_start,
            )
            return None

        if pad_hi < 0:
            self._error(
                "pad: pad_hi must be non-negative",
                expr.args[3].span.line_start, expr.args[3].span.col_start,
            )
            return None

        # Result shape: each dim + pad_lo + pad_hi
        result_dims = tuple(d + pad_lo + pad_hi for d in x_type.dims)
        return ArrayType(x_type.base, result_dims)

    def _check_linalg(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        """Type-check cholesky(x) and triangular_solve(a, b, lower, left_side)."""
        if expr.callee == "cholesky":
            if len(expr.args) != 1:
                self._error("cholesky expects exactly 1 argument", expr.span.line_start, expr.span.col_start)
                return None
            x_type = self._infer(expr.args[0], env)
            if x_type is None:
                return None
            if not isinstance(x_type, ArrayType):
                self._error(f"cholesky: argument must be a 2D array, got {x_type}", expr.span.line_start, expr.span.col_start)
                return None
            if len(x_type.dims) != 2:
                self._error(f"cholesky: argument must be a 2D array, got {len(x_type.dims)}D", expr.span.line_start, expr.span.col_start)
                return None
            if x_type.base not in FLOAT_BASES:
                self._error(f"cholesky: argument must be a float array, got {x_type.base}", expr.span.line_start, expr.span.col_start)
                return None
            d0, d1 = x_type.dims
            if isinstance(d0, int) and isinstance(d1, int) and d0 != d1:
                self._error(f"cholesky: argument must be a square matrix, got {d0}x{d1}", expr.span.line_start, expr.span.col_start)
                return None
            return x_type

        if expr.callee == "triangular_solve":
            if len(expr.args) != 4:
                self._error("triangular_solve expects exactly 4 arguments: (a, b, lower, left_side)", expr.span.line_start, expr.span.col_start)
                return None

            a_type = self._infer(expr.args[0], env)
            b_type = self._infer(expr.args[1], env)

            # lower and left_side must be bool literals
            if not isinstance(expr.args[2], BoolLiteral):
                self._error("triangular_solve: 'lower' (3rd argument) must be a bool literal (true or false)", expr.span.line_start, expr.span.col_start)
                return None
            if not isinstance(expr.args[3], BoolLiteral):
                self._error("triangular_solve: 'left_side' (4th argument) must be a bool literal (true or false)", expr.span.line_start, expr.span.col_start)
                return None
            self._infer(expr.args[2], env)
            self._infer(expr.args[3], env)

            if a_type is None or b_type is None:
                return None

            if not isinstance(a_type, ArrayType):
                self._error(f"triangular_solve: 'a' must be a 2D array, got {a_type}", expr.span.line_start, expr.span.col_start)
                return None
            if len(a_type.dims) != 2:
                self._error(f"triangular_solve: 'a' must be a 2D array, got {len(a_type.dims)}D", expr.span.line_start, expr.span.col_start)
                return None
            if a_type.base not in FLOAT_BASES:
                self._error(f"triangular_solve: 'a' must be a float array, got {a_type.base}", expr.span.line_start, expr.span.col_start)
                return None
            a_d0, a_d1 = a_type.dims
            if isinstance(a_d0, int) and isinstance(a_d1, int) and a_d0 != a_d1:
                self._error(f"triangular_solve: 'a' must be a square matrix, got {a_d0}x{a_d1}", expr.span.line_start, expr.span.col_start)
                return None

            if not isinstance(b_type, ArrayType):
                self._error(f"triangular_solve: 'b' must be a 2D array, got {b_type}", expr.span.line_start, expr.span.col_start)
                return None
            if len(b_type.dims) != 2:
                self._error(f"triangular_solve: 'b' must be a 2D array, got {len(b_type.dims)}D", expr.span.line_start, expr.span.col_start)
                return None
            if b_type.base not in FLOAT_BASES:
                self._error(f"triangular_solve: 'b' must be a float array, got {b_type.base}", expr.span.line_start, expr.span.col_start)
                return None
            if a_type.base != b_type.base:
                self._error(f"triangular_solve: 'a' and 'b' must have the same base type, got {a_type.base} and {b_type.base}", expr.span.line_start, expr.span.col_start)
                return None

            left_side = expr.args[3].value
            b_d0, b_d1 = b_type.dims
            n = a_d0  # a is N x N

            if left_side:
                # Solves A @ X = B, so B must be N x M
                if isinstance(n, int) and isinstance(b_d0, int) and n != b_d0:
                    self._error(f"triangular_solve: with left_side=true, b's first dimension ({b_d0}) must match a's dimension ({n})", expr.span.line_start, expr.span.col_start)
                    return None
            else:
                # Solves X @ A = B, so B must be M x N
                if isinstance(n, int) and isinstance(b_d1, int) and n != b_d1:
                    self._error(f"triangular_solve: with left_side=false, b's second dimension ({b_d1}) must match a's dimension ({n})", expr.span.line_start, expr.span.col_start)
                    return None

            return b_type

        return None  # unreachable

    def _check_conv2d(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        # conv2d(input, kernel) — stride=1, pad=0
        # conv2d(input, kernel, stride, pad) — same for both spatial dims
        # conv2d(input, kernel, stride_h, stride_w, pad_h, pad_w)
        nargs = len(expr.args)
        if nargs not in (2, 4, 6):
            self._error(
                "conv2d expects 2, 4, or 6 arguments: (input, kernel[, stride, pad | stride_h, stride_w, pad_h, pad_w])",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        input_type = self._infer(expr.args[0], env)
        kernel_type = self._infer(expr.args[1], env)
        # Infer remaining args (stride/pad literals)
        for a in expr.args[2:]:
            self._infer(a, env)

        if input_type is None or kernel_type is None:
            return None
        if not isinstance(input_type, ArrayType) or len(input_type.dims) != 4:
            self._error(
                f"conv2d: input must be a 4D array [N, Ci, H, W], got {input_type}",
                expr.args[0].span.line_start, expr.args[0].span.col_start,
            )
            return None
        if not isinstance(kernel_type, ArrayType) or len(kernel_type.dims) != 4:
            self._error(
                f"conv2d: kernel must be a 4D array [Co, Ci, Kh, Kw], got {kernel_type}",
                expr.args[1].span.line_start, expr.args[1].span.col_start,
            )
            return None
        if input_type.base != kernel_type.base:
            self._error(
                f"conv2d: input and kernel must have same base type, got {input_type.base} and {kernel_type.base}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        N, Ci, H, W = input_type.dims
        Co, Ki, Kh, Kw = kernel_type.dims
        if isinstance(Ci, int) and isinstance(Ki, int) and Ci != Ki:
            self._error(
                f"conv2d: input channels ({Ci}) must match kernel input channels ({Ki})",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # Extract stride/pad
        if nargs == 2:
            sh, sw, ph, pw = 1, 1, 0, 0
        elif nargs == 4:
            for i in (2, 3):
                if not isinstance(expr.args[i], IntLiteral):
                    self._error("conv2d: stride and pad must be integer literals",
                                expr.args[i].span.line_start, expr.args[i].span.col_start)
                    return None
            sh = sw = expr.args[2].value
            ph = pw = expr.args[3].value
        else:  # nargs == 6
            for i in range(2, 6):
                if not isinstance(expr.args[i], IntLiteral):
                    self._error("conv2d: stride and pad must be integer literals",
                                expr.args[i].span.line_start, expr.args[i].span.col_start)
                    return None
            sh, sw = expr.args[2].value, expr.args[3].value
            ph, pw = expr.args[4].value, expr.args[5].value

        # Compute output spatial dims
        if not isinstance(H, int) or not isinstance(W, int):
            self._error("conv2d: input spatial dimensions must be concrete",
                        expr.span.line_start, expr.span.col_start)
            return None
        if not isinstance(Kh, int) or not isinstance(Kw, int):
            self._error("conv2d: kernel spatial dimensions must be concrete",
                        expr.span.line_start, expr.span.col_start)
            return None

        OH = (H + 2 * ph - Kh) // sh + 1
        OW = (W + 2 * pw - Kw) // sw + 1
        if OH <= 0 or OW <= 0:
            self._error(
                f"conv2d: output spatial dimensions must be positive, got ({OH}, {OW})",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        return ArrayType(input_type.base, (N, Co, OH, OW))

    def _check_pool(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        # max_pool(input, wh, ww, sh, sw) or avg_pool(input, wh, ww, sh, sw)
        name = expr.callee
        if len(expr.args) != 5:
            self._error(
                f"{name} expects 5 arguments: (input, window_h, window_w, stride_h, stride_w)",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        input_type = self._infer(expr.args[0], env)
        for a in expr.args[1:]:
            self._infer(a, env)

        if input_type is None:
            return None
        if not isinstance(input_type, ArrayType) or len(input_type.dims) != 4:
            self._error(
                f"{name}: input must be a 4D array [N, C, H, W], got {input_type}",
                expr.args[0].span.line_start, expr.args[0].span.col_start,
            )
            return None

        for i in range(1, 5):
            if not isinstance(expr.args[i], IntLiteral):
                self._error(f"{name}: window and stride must be integer literals",
                            expr.args[i].span.line_start, expr.args[i].span.col_start)
                return None

        wh, ww = expr.args[1].value, expr.args[2].value
        sh, sw = expr.args[3].value, expr.args[4].value

        N, C, H, W = input_type.dims
        if not isinstance(H, int) or not isinstance(W, int):
            self._error(f"{name}: spatial dimensions must be concrete",
                        expr.span.line_start, expr.span.col_start)
            return None

        OH = (H - wh) // sh + 1
        OW = (W - ww) // sw + 1
        if OH <= 0 or OW <= 0:
            self._error(
                f"{name}: output spatial dimensions must be positive, got ({OH}, {OW})",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        return ArrayType(input_type.base, (N, C, OH, OW))

    def _check_generic_call(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        """Handle calls to generic (wildcard / symbolic-dim / comptime) functions via monomorphization."""
        import copy
        from .ast_nodes import Dim, Span as ASTSpan, IntLiteral

        generic_fn = self._generic_fns[expr.callee]
        generic_sig = self._resolve_signature(generic_fn)
        if generic_sig is None:
            return None

        # Resolve named arguments to positional
        self._resolve_named_args(expr, generic_sig.param_names)

        # Check arg count
        if len(expr.args) != len(generic_sig.param_types):
            self._error(
                f"function '{expr.callee}' expects {len(generic_sig.param_types)} arguments, got {len(expr.args)}",
                expr.span.line_start, expr.span.col_start,
            )
            return None

        # Validate comptime args are integer literals
        comptime_values: dict[str, IntLiteral] = {}
        if generic_sig.comptime:
            for i, is_ct in enumerate(generic_sig.comptime):
                if is_ct:
                    if not isinstance(expr.args[i], IntLiteral):
                        self._error(
                            f"comptime parameter '{generic_sig.param_names[i]}' requires an integer literal",
                            expr.args[i].span.line_start, expr.args[i].span.col_start,
                        )
                        return None
                    comptime_values[generic_sig.param_names[i]] = expr.args[i]

        # Infer arg types, unify symbolic dims, resolve wildcard shapes
        substitution: dict[str, int | str] = {}
        typevar_bindings: dict[str, StructType] = {}
        wildcard_shape: tuple[int | str, ...] | None = None
        wildcard_scalar = False
        arg_types: list[MaomiType | None] = []

        for i, (arg, param_type) in enumerate(zip(expr.args, generic_sig.param_types)):
            # Skip comptime params for type unification — they're literals, not arrays
            if generic_sig.comptime and i < len(generic_sig.comptime) and generic_sig.comptime[i]:
                arg_type = self._infer(arg, env)
                arg_types.append(arg_type)
                continue

            arg_type = self._infer(arg, env)
            arg_types.append(arg_type)
            if arg_type is None:
                continue

            if isinstance(param_type, TypeVar):
                # Type variable param — must be a struct type
                if not isinstance(arg_type, StructType):
                    self._error(
                        f"argument {i+1}: type variable '{param_type.name}' expects a struct type, got {arg_type}",
                        arg.span.line_start, arg.span.col_start,
                    )
                    return None
                if param_type.name in typevar_bindings:
                    if typevar_bindings[param_type.name].name != arg_type.name:
                        self._error(
                            f"argument {i+1}: type variable '{param_type.name}' bound to {typevar_bindings[param_type.name]} but got {arg_type}",
                            arg.span.line_start, arg.span.col_start,
                        )
                        return None
                else:
                    typevar_bindings[param_type.name] = arg_type
            elif isinstance(param_type, WildcardArrayType):
                # Wildcard param — match any shape with matching dtype
                if isinstance(arg_type, ArrayType):
                    if arg_type.base != param_type.base:
                        self._error(
                            f"argument {i+1}: expected {param_type.base}[..], got {arg_type}",
                            arg.span.line_start, arg.span.col_start,
                        )
                        return None
                    if wildcard_shape is None:
                        wildcard_shape = arg_type.dims
                    elif wildcard_shape != arg_type.dims:
                        self._error(
                            f"shape mismatch: wildcard resolved to {wildcard_shape} but argument {i+1} has shape {arg_type.dims}",
                            arg.span.line_start, arg.span.col_start,
                        )
                        return None
                elif isinstance(arg_type, ScalarType):
                    if arg_type.base != param_type.base:
                        self._error(
                            f"argument {i+1}: expected {param_type.base}[..], got {arg_type}",
                            arg.span.line_start, arg.span.col_start,
                        )
                        return None
                    if wildcard_shape is None:
                        wildcard_scalar = True
                else:
                    self._error(
                        f"argument {i+1}: expected {param_type.base}[..], got {arg_type}",
                        arg.span.line_start, arg.span.col_start,
                    )
                    return None
            else:
                # Non-wildcard param — normal type unification (handles symbolic dims)
                if not self._unify_arg(arg_type, param_type, substitution, expr, i):
                    return None

        # Resolve return type
        ret = generic_sig.return_type
        if isinstance(ret, TypeVar):
            if ret.name not in typevar_bindings:
                self._error(
                    f"type variable '{ret.name}' in return type not bound by any parameter",
                    expr.span.line_start, expr.span.col_start,
                )
                return None
            concrete_ret: MaomiType = typevar_bindings[ret.name]
        elif isinstance(ret, WildcardArrayType):
            if wildcard_scalar or wildcard_shape is None:
                concrete_ret = ScalarType(ret.base)
            else:
                concrete_ret = ArrayType(ret.base, wildcard_shape)
        else:
            concrete_ret = self._apply_substitution(ret, substitution)

        # Build monomorphized name from all resolved shapes/dims
        shape_parts: list[str] = []
        for i, at in enumerate(arg_types):
            if generic_sig.comptime and i < len(generic_sig.comptime) and generic_sig.comptime[i]:
                shape_parts.append(f"ct{expr.args[i].value}")
            elif at is None:
                shape_parts.append("?")
            elif isinstance(at, ArrayType):
                shape_parts.append("x".join(str(d) for d in at.dims))
            elif isinstance(at, ScalarType):
                shape_parts.append(at.base)
            elif isinstance(at, StructType):
                shape_parts.append(at.name)
            else:
                shape_parts.append(str(at))
        mono_name = f"{expr.callee}${'_'.join(shape_parts)}"

        # Create monomorphized copy if needed
        if mono_name not in self.fn_table:
            mono_fn = copy.deepcopy(generic_fn)
            mono_fn.name = mono_name

            # Replace type annotations with concrete types
            for p in mono_fn.params:
                ta = p.type_annotation
                if ta.base in typevar_bindings:
                    # TypeVar → concrete struct name
                    p.type_annotation = TypeAnnotation(typevar_bindings[ta.base].name, None, ta.span)
                elif ta.wildcard:
                    if wildcard_scalar or wildcard_shape is None:
                        p.type_annotation = TypeAnnotation(ta.base, None, ta.span)
                    else:
                        dims = [Dim(d, ASTSpan(0, 0, 0, 0)) for d in wildcard_shape]
                        p.type_annotation = TypeAnnotation(ta.base, dims, ta.span)
                elif ta.dims is not None:
                    # Resolve symbolic dims in annotation
                    new_dims = []
                    for d in ta.dims:
                        if isinstance(d.value, str) and d.value in substitution:
                            new_dims.append(Dim(substitution[d.value], d.span))
                        else:
                            new_dims.append(d)
                    p.type_annotation = TypeAnnotation(ta.base, new_dims, ta.span)

            # Resolve return type annotation
            rta = mono_fn.return_type
            ret_was_wildcard = rta.wildcard
            if rta.base in typevar_bindings:
                mono_fn.return_type = TypeAnnotation(typevar_bindings[rta.base].name, None, rta.span)
            elif rta.wildcard:
                # Keep wildcard — _check_fn will accept any shape with matching base.
                # Concrete return type will be inferred from the body after type-checking.
                pass
            elif rta.dims is not None:
                new_dims = []
                for d in rta.dims:
                    if isinstance(d.value, str) and d.value in substitution:
                        new_dims.append(Dim(substitution[d.value], d.span))
                    else:
                        new_dims.append(d)
                mono_fn.return_type = TypeAnnotation(rta.base, new_dims, rta.span)

            # Substitute comptime params with literals in the body, then remove from param list
            if comptime_values:
                _substitute_comptime(mono_fn.body, comptime_values)
                mono_fn.params = [p for p in mono_fn.params if not p.comptime]

            # Register and type-check the monomorphized function
            sig = self._resolve_signature(mono_fn)
            if sig is None:
                return None
            self.fn_table[mono_name] = sig
            self._check_fn(mono_fn)

            # If the return type was a wildcard, infer the concrete return from the body
            if ret_was_wildcard:
                env = TypeEnv()
                for p in mono_fn.params:
                    t = self._resolve_type_annotation(p.type_annotation)
                    if t:
                        env.define(p.name, t)
                body_ret = self._check_block(mono_fn.body, env)
                if body_ret is not None:
                    concrete_ret = body_ret
                    # Update the signature and return type annotation to match the body
                    self.fn_table[mono_name] = FnSignature(
                        sig.param_names, sig.param_types, concrete_ret, sig.comptime
                    )
                    if isinstance(concrete_ret, ArrayType):
                        dims = [Dim(d, ASTSpan(0, 0, 0, 0)) for d in concrete_ret.dims]
                        mono_fn.return_type = TypeAnnotation(concrete_ret.base, dims, mono_fn.return_type.span)
                    elif isinstance(concrete_ret, ScalarType):
                        mono_fn.return_type = TypeAnnotation(concrete_ret.base, None, mono_fn.return_type.span)

            # Add to program's function list so codegen can find it
            if self._program is not None:
                self._program.functions.append(mono_fn)

        # If the function was already monomorphized and the return was a wildcard,
        # retrieve the actual return type from the registered signature.
        if mono_name in self.fn_table and isinstance(ret, WildcardArrayType):
            concrete_ret = self.fn_table[mono_name].return_type

        # Rewrite the call to target the monomorphized function
        expr.callee = mono_name
        # Remove comptime args from call site — they're baked into the mono function body
        if comptime_values:
            expr.args = [a for i, a in enumerate(expr.args)
                         if not (generic_sig.comptime and i < len(generic_sig.comptime) and generic_sig.comptime[i])]
        return concrete_ret

    def _check_rng_call(self, expr: CallExpr, env: TypeEnv) -> MaomiType | None:
        callee = expr.callee
        args = expr.args
        span = expr.span

        if callee == "random.key":
            if len(args) != 1:
                self._error(f"random.key expects 1 argument (seed), got {len(args)}", span.line_start, span.col_start)
                return None
            seed_type = self._infer(args[0], env)
            if seed_type is not None and seed_type != I32:
                self._error(f"random.key: seed must be i32, got {seed_type}", args[0].span.line_start, args[0].span.col_start)
                return None
            return KEY_TYPE

        if callee == "random.split":
            if len(args) != 2:
                self._error(f"random.split expects 2 arguments (key, count), got {len(args)}", span.line_start, span.col_start)
                return None
            key_type = self._infer(args[0], env)
            if key_type is not None and key_type != KEY_TYPE:
                self._error(f"random.split: first argument must be Key (i32[4]), got {key_type}", args[0].span.line_start, args[0].span.col_start)
                return None
            if not isinstance(args[1], IntLiteral):
                self._error("random.split: count must be an integer literal", args[1].span.line_start, args[1].span.col_start)
                return None
            n = args[1].value
            if n < 1:
                self._error(f"random.split: count must be >= 1, got {n}", args[1].span.line_start, args[1].span.col_start)
                return None
            # Infer the literal so it gets a type in the type_map
            self._infer(args[1], env)
            return ArrayType("i32", (n, 4))

        # random.bernoulli(key, prob, d1, d2, ...)
        if callee == "random.bernoulli":
            if len(args) < 3:
                self._error(
                    f"{callee} expects at least 3 arguments (key, prob, dim...), got {len(args)}",
                    span.line_start, span.col_start,
                )
                return None
            key_type = self._infer(args[0], env)
            if key_type is not None and key_type != KEY_TYPE:
                self._error(f"{callee}: first argument must be Key (i32[4]), got {key_type}", args[0].span.line_start, args[0].span.col_start)
                return None
            prob_type = self._infer(args[1], env)
            if prob_type is not None and prob_type != F32:
                self._error(f"{callee}: prob must be f32, got {prob_type}", args[1].span.line_start, args[1].span.col_start)
                return None
            dims: list[int] = []
            for i in range(2, len(args)):
                if not isinstance(args[i], IntLiteral):
                    self._error(f"{callee}: dimension arguments must be integer literals", args[i].span.line_start, args[i].span.col_start)
                    return None
                if args[i].value < 1:
                    self._error(f"{callee}: dimensions must be >= 1, got {args[i].value}", args[i].span.line_start, args[i].span.col_start)
                    return None
                dims.append(args[i].value)
                self._infer(args[i], env)
            return ArrayType("f32", tuple(dims))

        # random.categorical(key, logits)
        if callee == "random.categorical":
            if len(args) != 2:
                self._error(
                    f"{callee} expects 2 arguments (key, logits), got {len(args)}",
                    span.line_start, span.col_start,
                )
                return None
            key_type = self._infer(args[0], env)
            if key_type is not None and key_type != KEY_TYPE:
                self._error(f"{callee}: first argument must be Key (i32[4]), got {key_type}", args[0].span.line_start, args[0].span.col_start)
                return None
            logits_type = self._infer(args[1], env)
            if logits_type is None:
                return None
            if not isinstance(logits_type, ArrayType) or logits_type.base != "f32":
                self._error(f"{callee}: logits must be f32 array, got {logits_type}", args[1].span.line_start, args[1].span.col_start)
                return None
            if len(logits_type.dims) < 1:
                self._error(f"{callee}: logits must have at least 1 dimension", args[1].span.line_start, args[1].span.col_start)
                return None
            # Return type: strip last dimension, change base to i32
            if len(logits_type.dims) == 1:
                return ScalarType("i32")
            return ArrayType("i32", tuple(logits_type.dims[:-1]))

        # random.uniform / random.normal / random.truncated_normal
        if callee in ("random.uniform", "random.normal", "random.truncated_normal"):
            param_names = {
                "random.uniform": ("low", "high"),
                "random.normal": ("mean", "stddev"),
                "random.truncated_normal": ("lo", "hi"),
            }
            if len(args) < 4:
                self._error(
                    f"{callee} expects at least 4 arguments (key, param1, param2, dim...), got {len(args)}",
                    span.line_start, span.col_start,
                )
                return None
            key_type = self._infer(args[0], env)
            if key_type is not None and key_type != KEY_TYPE:
                self._error(f"{callee}: first argument must be Key (i32[4]), got {key_type}", args[0].span.line_start, args[0].span.col_start)
                return None
            for i in (1, 2):
                t = self._infer(args[i], env)
                if t is not None and t != F32:
                    self._error(
                        f"{callee}: {param_names[callee][i-1]} must be f32, got {t}",
                        args[i].span.line_start, args[i].span.col_start,
                    )
                    return None
            dims: list[int] = []
            for i in range(3, len(args)):
                if not isinstance(args[i], IntLiteral):
                    self._error(f"{callee}: dimension arguments must be integer literals", args[i].span.line_start, args[i].span.col_start)
                    return None
                if args[i].value < 1:
                    self._error(f"{callee}: dimensions must be >= 1, got {args[i].value}", args[i].span.line_start, args[i].span.col_start)
                    return None
                dims.append(args[i].value)
                # Infer the literal so it gets a type in the type_map
                self._infer(args[i], env)
            return ArrayType("f32", tuple(dims))

        self._error(f"unknown RNG builtin '{callee}'", span.line_start, span.col_start)
        return None

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

        # Struct vs Struct — same name means same type
        if isinstance(arg_type, StructType) and isinstance(param_type, StructType):
            if arg_type.name == param_type.name:
                return True
            self._error(
                f"argument {arg_index} of '{expr.callee}': expected {param_type}, got {arg_type}",
                expr.args[arg_index].span.line_start,
                expr.args[arg_index].span.col_start,
            )
            return False

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

        # Array + Array — numpy-style broadcasting with size-1 dims
        if isinstance(a, ArrayType) and isinstance(b, ArrayType):
            max_rank = max(len(a.dims), len(b.dims))
            # Left-pad shorter dims with 1s
            a_padded = (1,) * (max_rank - len(a.dims)) + tuple(a.dims)
            b_padded = (1,) * (max_rank - len(b.dims)) + tuple(b.dims)
            result_dims: list[int | str] = []
            for ad, bd in zip(a_padded, b_padded):
                if self._dims_match(ad, bd):
                    result_dims.append(ad)
                elif isinstance(ad, int) and ad == 1:
                    result_dims.append(bd)
                elif isinstance(bd, int) and bd == 1:
                    result_dims.append(ad)
                else:
                    return None
            return ArrayType(a.base, tuple(result_dims))

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
