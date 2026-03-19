from __future__ import annotations

from lsprotocol import types

from ..ast_nodes import (
    FnDef, Block, LetStmt, ExprStmt, Param,
    BinOp, UnaryOp, IfExpr, CallExpr, ScanExpr, WhileExpr, MapExpr,
    GradExpr, ValueAndGradExpr, CastExpr, FoldExpr, ArrayLiteral,
    Identifier, StructLiteral, FieldAccess, WithExpr, IndexExpr, StructDef,
)


def _span_contains(span, line: int, col: int) -> bool:
    if line < span.line_start or line > span.line_end:
        return False
    if line == span.line_start and col < span.col_start:
        return False
    if line == span.line_end and col > span.col_end:
        return False
    return True


def _children_of(node):
    match node:
        case FnDef(params=params, body=body):
            yield from params
            yield body
        case Block(stmts=stmts, expr=expr):
            yield from stmts
            if expr is not None:
                yield expr
        case LetStmt(value=v):
            yield v
        case ExprStmt(expr=e):
            yield e
        case BinOp(left=l, right=r):
            yield l
            yield r
        case UnaryOp(operand=o):
            yield o
        case IfExpr(condition=c, then_block=t, else_block=e):
            yield c
            yield t
            yield e
        case CallExpr(args=args):
            yield from args
        case ScanExpr(init=init, sequences=seqs, body=body):
            yield init
            yield from seqs
            yield body
        case FoldExpr(init=init, sequences=seqs, body=body):
            yield init
            yield from seqs
            yield body
        case WhileExpr(init=init, cond=cond, body=body):
            yield init
            yield cond
            yield body
        case MapExpr(sequence=seq, body=body):
            yield seq
            yield body
        case CastExpr(expr=e):
            yield e
        case GradExpr(expr=e):
            yield e
        case ValueAndGradExpr(expr=e):
            yield e
        case ArrayLiteral(elements=elems):
            yield from elems
        case StructLiteral(fields=fields):
            for _, expr in fields:
                yield expr
        case FieldAccess(object=obj):
            yield obj
        case WithExpr(base=b, updates=updates):
            yield b
            for _, expr in updates:
                yield expr
        case IndexExpr(base=b, indices=indices):
            yield b
            for ic in indices:
                if ic.value is not None:
                    yield ic.value
                if ic.start is not None:
                    yield ic.start
                if ic.end is not None:
                    yield ic.end
        case _:
            pass


def _find_node_at(node, line: int, col: int):
    if not hasattr(node, "span") or not _span_contains(node.span, line, col):
        return None
    for child in _children_of(node):
        found = _find_node_at(child, line, col)
        if found is not None:
            return found
    return node


def _span_to_range(span) -> types.Range:
    """Convert 1-indexed Maomi Span to 0-indexed LSP Range."""
    return types.Range(
        start=types.Position(line=span.line_start - 1, character=span.col_start - 1),
        end=types.Position(line=span.line_end - 1, character=span.col_end - 1),
    )


def classify_symbol(node, line=None, col=None):
    """Determine the symbol name and kind from the node under cursor.

    Returns (name, kind) where kind is "function", "variable", or "struct",
    or (None, None) if the node is not a classifiable symbol.

    When line/col are provided, checks if cursor is on a Param's type
    annotation (returns "struct"). Without them, Param always returns "variable".
    """
    if isinstance(node, CallExpr):
        return node.callee, "function"
    if isinstance(node, FnDef):
        return node.name, "function"
    if isinstance(node, StructDef):
        return node.name, "struct"
    if isinstance(node, StructLiteral):
        return node.name, "struct"
    if isinstance(node, Identifier):
        return node.name, "variable"
    if isinstance(node, Param):
        if line is not None and col is not None:
            ta = node.type_annotation
            if ta and _span_contains(ta.span, line, col):
                return ta.base, "struct"
        return node.name, "variable"
    if isinstance(node, LetStmt):
        return node.name, "variable"
    return None, None
