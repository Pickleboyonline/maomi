from __future__ import annotations

from pathlib import Path

from lsprotocol import types

from ..ast_nodes import (
    FnDef, Block, LetStmt, ExprStmt, Param,
    CallExpr, ScanExpr, MapExpr, IfExpr,
    Identifier, StructLiteral, FieldAccess,
)
from ..types import StructType
from ._core import server, _cache, _local_functions
from ._ast_utils import _span_contains, _find_node_at, _span_to_range


def _goto_find_binding_in_block(name, block, line, col):
    """Search a Block for a LetStmt that binds `name` before (line, col)."""
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt) and stmt.name == name:
            if stmt.span.line_end < line or (
                stmt.span.line_end == line and stmt.span.col_end < col
            ):
                return stmt.span
        # Check inside expressions (scan/map bodies)
        if isinstance(stmt, ExprStmt):
            result = _goto_find_binding_in_expr(name, stmt.expr, line, col)
            if result is not None:
                return result
    if block.expr is not None:
        result = _goto_find_binding_in_expr(name, block.expr, line, col)
        if result is not None:
            return result
    return None


def _goto_find_binding_in_expr(name, expr, line, col):
    """Check if the cursor is inside a scan/map body and `name` matches a loop var."""
    if not hasattr(expr, "span") or not _span_contains(expr.span, line, col):
        return None

    if isinstance(expr, ScanExpr):
        if expr.carry_var == name:
            return expr.span
        if name in expr.elem_vars:
            return expr.span
        result = _goto_find_binding_in_block(name, expr.body, line, col)
        if result is not None:
            return result

    elif isinstance(expr, MapExpr):
        if expr.elem_var == name:
            return expr.span
        result = _goto_find_binding_in_block(name, expr.body, line, col)
        if result is not None:
            return result

    elif isinstance(expr, IfExpr):
        result = _goto_find_binding_in_block(name, expr.then_block, line, col)
        if result is not None:
            return result
        result = _goto_find_binding_in_block(name, expr.else_block, line, col)
        if result is not None:
            return result

    return None


def _goto_find_binding(name, fn, line, col):
    """Find where a variable name is bound in the enclosing function."""
    for param in fn.params:
        if param.name == name:
            return param.span
    return _goto_find_binding_in_block(name, fn.body, line, col)


def _goto_find_definition(node, fn, result):
    """Find the definition span and source file for the given node.

    Returns (span, source_file) or None. source_file is None for local definitions.
    """
    program = result.program
    if program is None:
        return None

    if isinstance(node, CallExpr):
        for fndef in program.functions:
            if fndef.name == node.callee:
                return fndef.span, fndef.source_file
        return None

    if isinstance(node, Identifier):
        binding = _goto_find_binding(node.name, fn, node.span.line_start, node.span.col_start)
        if binding is not None:
            return binding, None
        return None

    if isinstance(node, StructLiteral):
        for sdef in program.struct_defs:
            if sdef.name == node.name:
                return sdef.span, None
        return None

    if isinstance(node, FieldAccess):
        typ = result.type_map.get(id(node.object))
        if isinstance(typ, StructType):
            for sdef in program.struct_defs:
                if sdef.name == typ.name:
                    return sdef.span, None
        return None

    return None


@server.feature(types.TEXT_DOCUMENT_DEFINITION)
def goto_definition(ls, params: types.DefinitionParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None
    line = params.position.line + 1
    col = params.position.character + 1
    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            found = _goto_find_definition(node, fn, result)
            if found is not None:
                defn_span, source_file = found
                if source_file is not None:
                    target_uri = Path(source_file).as_uri()
                else:
                    target_uri = uri
                return types.Location(uri=target_uri, range=_span_to_range(defn_span))
    return None
