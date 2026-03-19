from __future__ import annotations

import logging

from lsprotocol import types

from ..ast_nodes import (
    FnDef, Block, LetStmt, ExprStmt, Param, Span,
    CallExpr, GradExpr, ValueAndGradExpr,
    Identifier, StructLiteral, StructDef,
    FieldAccess, WithExpr,
    ScanExpr, MapExpr, FoldExpr, WhileExpr,
)
from ._core import server, _cache, _local_functions
from ._ast_utils import (
    _span_contains, _find_node_at, _span_to_range, _children_of,
    classify_symbol,
)

logger = logging.getLogger("maomi-lsp")


# Keep _refs_classify_node as alias for backward compat (tests import it)
_refs_classify_node = classify_symbol


def _wrt_name(node):
    """Return the root wrt variable name from a GradExpr or ValueAndGradExpr."""
    return node.wrt[0] if isinstance(node.wrt, tuple) else node.wrt


def _wrt_span(node, source_lines):
    """Compute a narrow Span covering only the wrt variable name.

    *source_lines* is a list of source lines (0-indexed).  The wrt name
    is the last argument inside the grad/value_and_grad expression so we
    search backward from the closing parenthesis.
    """
    wrt_name = _wrt_name(node)
    # Search the source line that contains the end of the expression
    end_line_idx = node.span.line_end - 1  # 0-indexed
    if source_lines and 0 <= end_line_idx < len(source_lines):
        line_text = source_lines[end_line_idx]
        # Search backward from col_end for the wrt variable name
        search_end = node.span.col_end - 1  # 0-indexed, exclusive end
        idx = line_text.rfind(wrt_name, 0, search_end)
        if idx >= 0:
            # Span is 1-indexed
            return Span(
                node.span.line_end,
                idx + 1,
                node.span.line_end,
                idx + 1 + len(wrt_name),
            )
    # Fallback: use the full expression span
    return node.span


def _find_name_in_span(name, container_span, source_lines):
    """Find a standalone identifier *name* within *container_span*.

    Searches source text line-by-line within the span bounds.  Returns a
    narrow Span covering just the name, or *container_span* as fallback.
    """
    if not source_lines:
        return container_span
    for line_idx in range(container_span.line_start - 1, min(container_span.line_end, len(source_lines))):
        line_text = source_lines[line_idx]
        start = container_span.col_start - 1 if line_idx == container_span.line_start - 1 else 0
        while True:
            idx = line_text.find(name, start)
            if idx < 0:
                break
            end_idx = idx + len(name)
            before_ok = idx == 0 or not (line_text[idx - 1].isalnum() or line_text[idx - 1] == '_')
            after_ok = end_idx >= len(line_text) or not (line_text[end_idx].isalnum() or line_text[end_idx] == '_')
            if before_ok and after_ok:
                return Span(line_idx + 1, idx + 1, line_idx + 1, end_idx + 1)
            start = idx + 1
    return container_span


def _refs_find_loop_var_decl(fn, name, source_lines):
    """Walk a function body looking for scan/map/fold/while binding variables.

    Returns a Span for the declaration if found, None otherwise.
    """
    def _walk(node):
        if isinstance(node, ScanExpr):
            if node.carry_var == name or name in node.elem_vars:
                return _find_name_in_span(name, node.span, source_lines)
        elif isinstance(node, MapExpr):
            if node.elem_var == name:
                return _find_name_in_span(name, node.span, source_lines)
        elif isinstance(node, FoldExpr):
            if node.carry_var == name or name in node.elem_vars:
                return _find_name_in_span(name, node.span, source_lines)
        elif isinstance(node, WhileExpr):
            if node.state_var == name:
                return _find_name_in_span(name, node.span, source_lines)

        for child in _children_of(node):
            result = _walk(child)
            if result is not None:
                return result
        return None

    return _walk(fn.body)


def _refs_walk_node(node, name, kind, spans, source_lines=None):
    """Recursively walk AST collecting spans of references to the named symbol."""
    if kind == "function":
        if isinstance(node, CallExpr) and node.callee == name:
            spans.append(node.span)
    elif kind == "variable":
        if isinstance(node, Identifier) and node.name == name:
            spans.append(node.span)
        if isinstance(node, (GradExpr, ValueAndGradExpr)):
            if _wrt_name(node) == name:
                spans.append(_wrt_span(node, source_lines))
    elif kind == "struct":
        if isinstance(node, StructLiteral) and node.name == name:
            spans.append(node.span)
    elif kind == "field":
        if isinstance(node, FieldAccess) and node.field == name:
            # Narrow span: field name starts after the dot
            col_start = node.span.col_end - len(node.field)
            col_end = node.span.col_end
            spans.append(Span(node.span.line_end, col_start, node.span.line_end, col_end))
        if isinstance(node, StructLiteral):
            for field_name, _ in node.fields:
                if field_name == name:
                    spans.append(_find_name_in_span(field_name, node.span, source_lines))
        if isinstance(node, WithExpr):
            for path, _ in node.updates:
                if name in path:
                    spans.append(_find_name_in_span(name, node.span, source_lines))

    if kind == "struct":
        if isinstance(node, Param) and node.type_annotation and node.type_annotation.base == name:
            spans.append(node.type_annotation.span)
        if isinstance(node, FnDef) and node.return_type and node.return_type.base == name:
            spans.append(node.return_type.span)
        if isinstance(node, LetStmt) and node.type_annotation and node.type_annotation.base == name:
            spans.append(node.type_annotation.span)

    for child in _children_of(node):
        _refs_walk_node(child, name, kind, spans, source_lines=source_lines)


def _refs_collect_all(result, name, kind, include_declaration, fn_scope=None, source_lines=None):
    """Collect all reference spans for the given symbol."""
    spans = []

    if kind == "function":
        if include_declaration:
            for fn in _local_functions(result.program):
                if fn.name == name:
                    spans.append(fn.span)
                    break
        for fn in _local_functions(result.program):
            _refs_walk_node(fn, name, kind, spans, source_lines=source_lines)

    elif kind == "variable":
        if fn_scope is not None:
            if include_declaration:
                for p in fn_scope.params:
                    if p.name == name:
                        spans.append(p.span)
                        break
                else:
                    _refs_find_let_decl(fn_scope.body, name, spans)
                    # G8: scan/map/fold/while variable declarations
                    if not spans:
                        loop_span = _refs_find_loop_var_decl(fn_scope, name, source_lines)
                        if loop_span is not None:
                            spans.append(loop_span)
            _refs_walk_node(fn_scope.body, name, kind, spans, source_lines=source_lines)

    elif kind == "struct":
        if include_declaration:
            for sd in result.program.struct_defs:
                if sd.name == name:
                    spans.append(sd.span)
                    break
        for fn in _local_functions(result.program):
            _refs_walk_node(fn, name, kind, spans, source_lines=source_lines)
        # G11: struct-def field type annotations
        for sd in result.program.struct_defs:
            for _, field_type in sd.fields:
                if field_type.base == name:
                    spans.append(field_type.span)

    elif kind == "field":
        # G9: field references — walk AST + check struct definitions
        if include_declaration:
            for sd in result.program.struct_defs:
                for field_name, _ in sd.fields:
                    if field_name == name:
                        spans.append(sd.span)
                        break
        for fn in _local_functions(result.program):
            _refs_walk_node(fn, name, kind, spans, source_lines=source_lines)

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

    # Get source lines for narrow wrt span computation
    doc = ls.workspace.get_text_document(uri)
    source_lines = doc.source.splitlines() if doc else None

    for sd in result.program.struct_defs:
        if _span_contains(sd.span, line, col):
            name, kind = sd.name, "struct"
            spans = _refs_collect_all(result, name, kind,
                                      params.context.include_declaration,
                                      source_lines=source_lines)
            return [types.Location(uri=uri, range=_span_to_range(s))
                    for s in spans]

    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            name, kind = _refs_classify_node(node, line, col)
            if name:
                spans = _refs_collect_all(result, name, kind,
                                          params.context.include_declaration,
                                          fn_scope=fn,
                                          source_lines=source_lines)
                return [types.Location(uri=uri, range=_span_to_range(s))
                        for s in spans]
    return None
