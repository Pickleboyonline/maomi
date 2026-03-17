from __future__ import annotations

import logging

from lsprotocol import types

from ..ast_nodes import (
    FnDef, Block, LetStmt, ExprStmt, Param,
    CallExpr, GradExpr, ValueAndGradExpr,
    Identifier, StructLiteral, StructDef,
)
from ._core import server, _cache, _local_functions
from ._ast_utils import (
    _span_contains, _find_node_at, _span_to_range, _children_of,
    classify_symbol,
)

logger = logging.getLogger("maomi-lsp")


# Keep _refs_classify_node as alias for backward compat (tests import it)
_refs_classify_node = classify_symbol


def _refs_walk_node(node, name, kind, spans):
    """Recursively walk AST collecting spans of references to the named symbol."""
    if kind == "function":
        if isinstance(node, CallExpr) and node.callee == name:
            spans.append(node.span)
    elif kind == "variable":
        if isinstance(node, Identifier) and node.name == name:
            spans.append(node.span)
        if isinstance(node, GradExpr):
            wrt_root = node.wrt[0] if isinstance(node.wrt, tuple) else node.wrt
            if wrt_root == name:
                spans.append(node.span)
        if isinstance(node, ValueAndGradExpr):
            wrt_root = node.wrt[0] if isinstance(node.wrt, tuple) else node.wrt
            if wrt_root == name:
                spans.append(node.span)
    elif kind == "struct":
        if isinstance(node, StructLiteral) and node.name == name:
            spans.append(node.span)

    if kind == "struct":
        if isinstance(node, Param) and node.type_annotation.base == name:
            spans.append(node.type_annotation.span)
        if isinstance(node, FnDef) and node.return_type and node.return_type.base == name:
            spans.append(node.return_type.span)

    for child in _children_of(node):
        _refs_walk_node(child, name, kind, spans)


def _refs_collect_all(result, name, kind, include_declaration, fn_scope=None):
    """Collect all reference spans for the given symbol."""
    spans = []

    if kind == "function":
        if include_declaration:
            for fn in _local_functions(result.program):
                if fn.name == name:
                    spans.append(fn.span)
                    break
        for fn in _local_functions(result.program):
            _refs_walk_node(fn, name, kind, spans)

    elif kind == "variable":
        if fn_scope is not None:
            if include_declaration:
                for p in fn_scope.params:
                    if p.name == name:
                        spans.append(p.span)
                        break
                else:
                    _refs_find_let_decl(fn_scope.body, name, spans)
            _refs_walk_node(fn_scope.body, name, kind, spans)

    elif kind == "struct":
        if include_declaration:
            for sd in result.program.struct_defs:
                if sd.name == name:
                    spans.append(sd.span)
                    break
        for fn in _local_functions(result.program):
            _refs_walk_node(fn, name, kind, spans)

    return spans


def _refs_find_let_decl(block, name, spans):
    """Find a LetStmt declaration for the given variable name within a Block."""
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt) and stmt.name == name:
            spans.append(stmt.span)
            return
        if isinstance(stmt, ExprStmt):
            for child in _children_of(stmt.expr):
                if isinstance(child, Block):
                    _refs_find_let_decl(child, name, spans)


@server.feature(types.TEXT_DOCUMENT_REFERENCES)
def find_references(ls, params: types.ReferenceParams):
    logger.debug("find_references: %s at %d:%d", params.text_document.uri,
                 params.position.line, params.position.character)
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None

    line = params.position.line + 1
    col = params.position.character + 1

    for sd in result.program.struct_defs:
        if _span_contains(sd.span, line, col):
            name, kind = sd.name, "struct"
            spans = _refs_collect_all(result, name, kind,
                                      params.context.include_declaration)
            return [types.Location(uri=uri, range=_span_to_range(s))
                    for s in spans]

    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            name, kind = _refs_classify_node(node, line, col)
            if name:
                spans = _refs_collect_all(result, name, kind,
                                          params.context.include_declaration,
                                          fn_scope=fn)
                return [types.Location(uri=uri, range=_span_to_range(s))
                        for s in spans]
    return None
