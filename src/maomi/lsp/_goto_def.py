from __future__ import annotations

import logging
from pathlib import Path

from lsprotocol import types

from ..ast_nodes import (
    FnDef, Block, LetStmt, ExprStmt, Param,
    CallExpr, ScanExpr, MapExpr, IfExpr, FoldExpr, WhileExpr,
    Identifier, StructLiteral, FieldAccess, Span,
)
from ..types import StructType
from ._core import server, _cache, _local_functions, _uri_to_path
from ._ast_utils import _span_contains, _find_node_at, _span_to_range
from ._completion import _find_module_file

logger = logging.getLogger("maomi-lsp")


def _goto_find_binding_in_block(name, block, line, col):
    """Search a Block for the NEAREST LetStmt that binds `name` before (line, col)."""
    last_match = None
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt) and stmt.name == name:
            if stmt.span.line_end < line or (
                stmt.span.line_end == line and stmt.span.col_end < col
            ):
                last_match = stmt.span  # keep searching for later shadow
        # Check inside expressions (scan/map bodies)
        if isinstance(stmt, ExprStmt):
            result = _goto_find_binding_in_expr(name, stmt.expr, line, col)
            if result is not None:
                last_match = result
    if block.expr is not None:
        result = _goto_find_binding_in_expr(name, block.expr, line, col)
        if result is not None:
            last_match = result
    return last_match


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

    elif isinstance(expr, FoldExpr):
        if expr.carry_var == name:
            return expr.span
        if name in expr.elem_vars:
            return expr.span
        result = _goto_find_binding_in_block(name, expr.body, line, col)
        if result is not None:
            return result

    elif isinstance(expr, WhileExpr):
        if expr.state_var == name:
            return expr.span
        result = _goto_find_binding_in_block(name, expr.cond, line, col)
        if result is not None:
            return result
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
    """Find where a variable name is bound in the enclosing function.

    Check let stmts first so that shadowed variables resolve to the let,
    not the param.
    """
    block_match = _goto_find_binding_in_block(name, fn.body, line, col)
    if block_match is not None:
        return block_match
    for param in fn.params:
        if param.name == name:
            return param.span
    return None


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

    if isinstance(node, Param):
        ta = node.type_annotation
        if ta:
            for sdef in program.struct_defs:
                if sdef.name == ta.base:
                    return sdef.span, getattr(sdef, 'source_file', None)
        return None

    return None


def _goto_import_definition(result, line: int, col: int, filepath: str):
    """Check if cursor is on an import and return (target_file, target_span) or None."""
    for imp in result.program.imports:
        # Click on module name → jump to module file
        if imp.module_span and _span_contains(imp.module_span, line, col):
            mod_path = _find_module_file(imp.module_path, filepath)
            if mod_path:
                # Jump to line 1, col 1 of the module file
                return mod_path, Span(1, 1, 1, 1)
            return None

        # Click on imported name → jump to its definition in the module
        if imp.names and imp.name_spans:
            for name, span in zip(imp.names, imp.name_spans):
                if _span_contains(span, line, col):
                    # Find the function/struct in the resolved program
                    module_name = imp.alias or imp.module_path
                    qualified = f"{module_name}.{name}"
                    for fn_def in result.program.functions:
                        if fn_def.name == qualified or fn_def.name == name:
                            return fn_def.source_file, fn_def.span
                    for sd in result.program.struct_defs:
                        if sd.name == qualified or sd.name == name:
                            source = getattr(sd, 'source_file', None)
                            return source, sd.span
                    # Fallback: jump to module file
                    mod_path = _find_module_file(imp.module_path, filepath)
                    if mod_path:
                        return mod_path, Span(1, 1, 1, 1)
                    return None

        # Click on alias → jump to module file
        if imp.alias_span and _span_contains(imp.alias_span, line, col):
            mod_path = _find_module_file(imp.module_path, filepath)
            if mod_path:
                return mod_path, Span(1, 1, 1, 1)
            return None

    return None


@server.feature(types.TEXT_DOCUMENT_DEFINITION)
def goto_definition(ls, params: types.DefinitionParams):
    logger.debug("goto_definition: %s at %d:%d", params.text_document.uri,
                 params.position.line, params.position.character)
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None
    line = params.position.line + 1
    col = params.position.character + 1

    # Check imports first
    filepath = _uri_to_path(uri)
    import_def = _goto_import_definition(result, line, col, filepath)
    if import_def is not None:
        target_file, target_span = import_def
        if target_file is not None:
            target_uri = Path(target_file).as_uri()
        else:
            target_uri = uri
        return types.Location(uri=target_uri, range=_span_to_range(target_span))

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
