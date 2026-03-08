from __future__ import annotations

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
    CastExpr,
    FoldExpr,
    ArrayLiteral,
    StructLiteral,
    FieldAccess,
    WithExpr,
    IndexExpr,
    _BroadcastExpr,
    _ReduceSum,
    Expr,
)
from ...types import MaomiType, ScalarType, ArrayType, StructType
from ...errors import MaomiError

from ...builtins import ELEMENTWISE as _EW_REGISTRY


# Maps Maomi base types to MLIR element types
_MLIR_ETYPE = {
    "f32": "f32",
    "f64": "f64",
    "bf16": "bf16",
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

# Derived from central registry — maps builtin name to StableHLO op
_BUILTIN_OPS = {n: b.stablehlo_op for n, b in _EW_REGISTRY.items() if b.stablehlo_op}


def _block_references_var(block: Block, var_name: str) -> bool:
    """Check if a block references a variable name (stmts + trailing expr)."""
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt):
            if _expr_references_var(stmt.value, var_name):
                return True
        elif isinstance(stmt, ExprStmt):
            if _expr_references_var(stmt.expr, var_name):
                return True
    if block.expr is not None:
        return _expr_references_var(block.expr, var_name)
    return False


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
        case ArrayLiteral(elements=elems):
            return any(_expr_references_var(e, var_name) for e in elems)
        case StructLiteral(fields=fields):
            return any(_expr_references_var(v, var_name) for _, v in fields)
        case FieldAccess(object=obj):
            return _expr_references_var(obj, var_name)
        case WithExpr(base=base, updates=updates):
            return (_expr_references_var(base, var_name)
                    or any(_expr_references_var(v, var_name) for _, v in updates))
        case IndexExpr(base=base, indices=indices):
            if _expr_references_var(base, var_name):
                return True
            for ic in indices:
                for e in (ic.value, ic.start, ic.end):
                    if e is not None and _expr_references_var(e, var_name):
                        return True
            return False
        case CastExpr(expr=e):
            return _expr_references_var(e, var_name)
        case ScanExpr(init=init, sequences=seqs, body=body):
            return (_expr_references_var(init, var_name)
                    or any(_expr_references_var(s, var_name) for s in seqs)
                    or _block_references_var(body, var_name))
        case FoldExpr(init=init, sequences=seqs, body=body):
            return (_expr_references_var(init, var_name)
                    or any(_expr_references_var(s, var_name) for s in seqs)
                    or _block_references_var(body, var_name))
        case MapExpr(sequence=seq, body=body):
            return (_expr_references_var(seq, var_name)
                    or _block_references_var(body, var_name))
        case _:
            return False


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
        case ArrayLiteral(elements=elems):
            for e in elems:
                _collect_refs_expr(e, refs)
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
