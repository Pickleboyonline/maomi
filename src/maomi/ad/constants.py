from __future__ import annotations

from ..ast_nodes import (
    Identifier,
    UnaryOp,
    BinOp,
    CallExpr,
    IfExpr,
    ScanExpr,
    WhileExpr,
    MapExpr,
    GradExpr,
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
    _CumsumGrad,
    _SortGrad,
    Span,
    Expr,
)

_DUMMY_SPAN = Span(0, 0, 0, 0)

# Builtin sets derived from central registry
from ..builtins import ELEMENTWISE as _EW_REGISTRY, COMPLEX as _CX_REGISTRY, ALL_NAMES as _ALL_BUILTIN_NAMES

_ELEMENTWISE_BUILTINS = set(_EW_REGISTRY)
_REDUCTION_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "reduction"}
_SHAPE_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "shape"}
_NONDIFF_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.ad_behavior == "nondiff"}
_IOTA_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "construction"}
_CONV_POOL_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "conv_pool"}
_RNG_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "rng"}
_STOP_GRAD_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "stop_grad"}
_WHERE_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "where"}
_CLIP_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "clip"}
_ARGMAX_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "argmax"}
_TWO_ARG_EW_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "two_arg_elementwise"}
_EINSUM_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "einsum"}
_CUMULATIVE_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "cumulative"}
_SORTING_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "sorting"}
_BOOL_REDUCTION_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "bool_reduction"}
_ARRAY_MANIP_BUILTINS = {n for n, b in _CX_REGISTRY.items() if b.category == "array_manip"}
_MAX_GRAD_DEPTH = 10


def _collect_free_vars(expr: Expr) -> set[str]:
    """Collect all Identifier names referenced in an expression."""
    result: set[str] = set()
    _collect_free_vars_inner(expr, result)
    return result


def _collect_free_vars_inner(expr: Expr, result: set[str]):
    match expr:
        case Identifier(name=name):
            result.add(name)
        case UnaryOp(operand=operand):
            _collect_free_vars_inner(operand, result)
        case BinOp(left=left, right=right):
            _collect_free_vars_inner(left, result)
            _collect_free_vars_inner(right, result)
        case CallExpr(args=args):
            for a in args:
                _collect_free_vars_inner(a, result)
        case IfExpr(condition=cond, then_block=then_b, else_block=else_b):
            _collect_free_vars_inner(cond, result)
            if then_b.expr:
                _collect_free_vars_inner(then_b.expr, result)
            if else_b.expr:
                _collect_free_vars_inner(else_b.expr, result)
        case MapExpr(elem_var=ev, sequence=seq, body=body):
            _collect_free_vars_inner(seq, result)
            if body.expr:
                inner = set[str]()
                _collect_free_vars_inner(body.expr, inner)
                inner.discard(ev)  # elem_var is bound, not free
                result.update(inner)
        case CastExpr(expr=inner):
            _collect_free_vars_inner(inner, result)
        case ScanExpr(carry_var=cv, elem_vars=evs, init=init, sequences=seqs, body=body):
            _collect_free_vars_inner(init, result)
            for s in seqs:
                _collect_free_vars_inner(s, result)
            if body.expr:
                inner = set[str]()
                _collect_free_vars_inner(body.expr, inner)
                inner.discard(cv)
                for ev in evs:
                    inner.discard(ev)
                result.update(inner)
        case FoldExpr(carry_var=cv, elem_vars=evs, init=init, sequences=seqs, body=body):
            _collect_free_vars_inner(init, result)
            for s in seqs:
                _collect_free_vars_inner(s, result)
            if body.expr:
                inner = set[str]()
                _collect_free_vars_inner(body.expr, inner)
                inner.discard(cv)
                for ev in evs:
                    inner.discard(ev)
                result.update(inner)
        case WhileExpr(state_var=sv, init=init, cond=cond, body=body):
            _collect_free_vars_inner(init, result)
            inner = set[str]()
            if cond.expr:
                _collect_free_vars_inner(cond.expr, inner)
            if body.expr:
                _collect_free_vars_inner(body.expr, inner)
            inner.discard(sv)
            result.update(inner)
        case ArrayLiteral(elements=elems):
            for e in elems:
                _collect_free_vars_inner(e, result)
        case StructLiteral(fields=fields):
            for _, fv in fields:
                _collect_free_vars_inner(fv, result)
        case FieldAccess(object=obj):
            _collect_free_vars_inner(obj, result)
        case WithExpr(base=base, updates=updates):
            _collect_free_vars_inner(base, result)
            for _, ve in updates:
                _collect_free_vars_inner(ve, result)
        case IndexExpr(base=base, indices=indices):
            _collect_free_vars_inner(base, result)
            for ic in indices:
                if ic.value is not None:
                    _collect_free_vars_inner(ic.value, result)
                if ic.start is not None:
                    _collect_free_vars_inner(ic.start, result)
                if ic.end is not None:
                    _collect_free_vars_inner(ic.end, result)
        case _IndexGrad(base_expr=base, adj=adj, indices=indices):
            _collect_free_vars_inner(base, result)
            _collect_free_vars_inner(adj, result)
            for ic in indices:
                if ic.value is not None:
                    _collect_free_vars_inner(ic.value, result)
                if ic.start is not None:
                    _collect_free_vars_inner(ic.start, result)
                if ic.end is not None:
                    _collect_free_vars_inner(ic.end, result)
        case _GatherGrad(base_expr=base, adj=adj, indices=indices):
            _collect_free_vars_inner(base, result)
            _collect_free_vars_inner(adj, result)
            _collect_free_vars_inner(indices, result)
        case _ScanGrad():
            _collect_free_vars_inner(expr.d_body_d_carry, result)
            for de in expr.d_body_d_elems:
                _collect_free_vars_inner(de, result)
            _collect_free_vars_inner(expr.init, result)
            for s in expr.sequences:
                _collect_free_vars_inner(s, result)
            _collect_free_vars_inner(expr.forward_result, result)
            _collect_free_vars_inner(expr.adj, result)
        case _WhileGrad():
            _collect_free_vars_inner(expr.d_body_d_state, result)
            _collect_free_vars_inner(expr.init, result)
            _collect_free_vars_inner(expr.forward_result, result)
            _collect_free_vars_inner(expr.adj, result)
        case GradExpr(expr=inner_expr):
            _collect_free_vars_inner(inner_expr, result)
        case _Conv2dGrad(input_expr=ie, kernel_expr=ke, adj=adj):
            _collect_free_vars_inner(ie, result)
            _collect_free_vars_inner(ke, result)
            _collect_free_vars_inner(adj, result)
        case _MaxPoolGrad(input_expr=ie, adj=adj):
            _collect_free_vars_inner(ie, result)
            _collect_free_vars_inner(adj, result)
        case _AvgPoolGrad(input_expr=ie, adj=adj):
            _collect_free_vars_inner(ie, result)
            _collect_free_vars_inner(adj, result)
        case _FoldGrad(d_body_d_carry=dc, d_body_d_elems=des, init=init, sequences=seqs, adj=adj):
            _collect_free_vars_inner(dc, result)
            for de in des:
                _collect_free_vars_inner(de, result)
            _collect_free_vars_inner(init, result)
            for s in seqs:
                _collect_free_vars_inner(s, result)
            _collect_free_vars_inner(adj, result)
        case _BroadcastExpr(expr=e):
            _collect_free_vars_inner(e, result)
        case _ReduceSum(expr=e):
            _collect_free_vars_inner(e, result)
        case _CumsumGrad(input_expr=ie, adj=adj):
            _collect_free_vars_inner(ie, result)
            _collect_free_vars_inner(adj, result)
        case _SortGrad(input_expr=ie, adj=adj):
            _collect_free_vars_inner(ie, result)
            _collect_free_vars_inner(adj, result)
