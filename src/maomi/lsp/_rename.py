from __future__ import annotations

import logging
import re

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ..ast_nodes import (
    Program, FnDef, LetStmt, Param, Block,
    CallExpr,
    Identifier, StructLiteral, StructDef,
)
from ._core import server, _cache, AnalysisResult, _local_functions
from ._ast_utils import _span_contains, _find_node_at, _children_of, classify_symbol
from ._builtin_data import _BUILTIN_SET, _KEYWORDS, _TYPE_NAMES

_VALID_IDENT = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
_KEYWORD_SET = frozenset(_KEYWORDS)
_TYPE_NAME_SET = frozenset(_TYPE_NAMES)

logger = logging.getLogger("maomi-lsp")


def _rename_name_range(name: str, span, source_lines: list[str]) -> types.Range | None:
    """Find the exact Range of *name* within the source region of *span*."""
    line_idx = span.line_start - 1
    if line_idx < 0 or line_idx >= len(source_lines):
        return None
    line_text = source_lines[line_idx]
    col_start = span.col_start - 1
    idx = line_text.find(name, col_start)
    if idx >= 0:
        return types.Range(
            start=types.Position(line=line_idx, character=idx),
            end=types.Position(line=line_idx, character=idx + len(name)),
        )
    return None


def _rename_walk(node, callback):
    """Recursively walk *node* and all its children, calling *callback* on each."""
    callback(node)
    for child in _children_of(node):
        _rename_walk(child, callback)


class _EditCollector:
    """Accumulates unique TextEdit ranges, deduplicating by position."""

    def __init__(self, new_name: str):
        self.new_name = new_name
        self.edits: list[types.TextEdit] = []
        self._seen: set[tuple[int, int, int, int]] = set()

    def add(self, rng: types.Range | None):
        if rng is None:
            return
        key = (rng.start.line, rng.start.character, rng.end.line, rng.end.character)
        if key not in self._seen:
            self._seen.add(key)
            self.edits.append(types.TextEdit(range=rng, new_text=self.new_name))


def _rename_collect_function_edits(
    program: Program, name: str, new_name: str, source_lines: list[str],
) -> list[types.TextEdit]:
    ec = _EditCollector(new_name)
    for fn in program.functions:
        if fn.name == name:
            ec.add(_rename_name_range(name, fn.span, source_lines))

        def _visit(node, _name=name):
            if isinstance(node, CallExpr) and node.callee == _name:
                ec.add(_rename_name_range(_name, node.span, source_lines))

        _rename_walk(fn, _visit)
    return ec.edits


def _rename_collect_variable_edits(
    fn_scope: FnDef, name: str, new_name: str, source_lines: list[str],
    cursor_node=None,
) -> list[types.TextEdit]:
    ec = _EditCollector(new_name)

    # Determine which scope the cursor is in.  Build a list of declaration
    # boundaries: the param (if any) and each LetStmt that re-binds *name*.
    # Then figure out which scope the cursor_node belongs to.

    def _span_key(span):
        return (span.line_start, span.col_start, span.line_end, span.col_end)

    # Find which declaration the cursor is on.  If cursor_node is a Param, it
    # is the param scope.  If it is a LetStmt with matching name, it is that
    # let-scope.  If it is an Identifier (usage), we need to find which scope
    # it lives in — we do this by walking the body and tracking scope state.
    cursor_is_param = isinstance(cursor_node, Param) and cursor_node.name == name

    # Determine if cursor is on a shadowing LetStmt
    cursor_let_key = None
    if isinstance(cursor_node, LetStmt) and cursor_node.name == name:
        cursor_let_key = _span_key(cursor_node.span)

    # Determine if cursor is on a usage (Identifier)
    cursor_ident_key = None
    if isinstance(cursor_node, Identifier) and cursor_node.name == name:
        cursor_ident_key = _span_key(cursor_node.span)

    # Build ordered list of all LetStmt shadow declarations for this name
    shadow_let_keys: list[tuple[int, int, int, int]] = []

    def _find_shadows(node):
        if isinstance(node, Block):
            for stmt in node.stmts:
                if isinstance(stmt, LetStmt) and stmt.name == name:
                    shadow_let_keys.append(_span_key(stmt.span))
                _find_shadows(stmt)
            if node.expr is not None:
                _find_shadows(node.expr)
        else:
            for child in _children_of(node):
                _find_shadows(child)

    _find_shadows(fn_scope.body)

    # If cursor is on a param, the "origin" is the param scope
    # If cursor is on a let, the "origin" is that let scope
    # If cursor is on an identifier, we need to figure out which scope
    # it belongs to by walking and tracking scope state

    # For simplicity, use a scope-tracking walk that collects references
    # per-scope, then determine which scope the cursor belongs to.
    # scope_id: -1 = param, i = index into shadow_let_keys
    shadow_key_to_idx = {k: i for i, k in enumerate(shadow_let_keys)}

    # Collect (scope_id, range) pairs
    scope_refs: dict[int, list[types.Range]] = {}
    # Track which scope each span_key belongs to
    span_to_scope: dict[tuple[int, int, int, int], int] = {}

    # Check if param matches
    param_match = False
    for param in fn_scope.params:
        if param.name == name:
            param_match = True
            rng = _rename_name_range(name, param.span, source_lines)
            if rng is not None:
                scope_refs.setdefault(-1, []).append(rng)
                span_to_scope[_span_key(param.span)] = -1

    def _walk_scope_tracking(node, current_scope: int):
        """Walk collecting refs, tracking which scope they belong to."""
        if isinstance(node, Block):
            scope = current_scope
            for stmt in node.stmts:
                if isinstance(stmt, LetStmt) and stmt.name == name:
                    sk = _span_key(stmt.span)
                    idx = shadow_key_to_idx.get(sk)
                    if idx is not None:
                        # RHS still belongs to the previous scope
                        _walk_scope_tracking(stmt.value, scope)
                        # The binding name starts a new scope
                        scope = idx
                        rng = _rename_name_range(name, stmt.span, source_lines)
                        if rng is not None:
                            scope_refs.setdefault(scope, []).append(rng)
                            span_to_scope[sk] = scope
                    else:
                        _walk_scope_tracking(stmt, scope)
                else:
                    _walk_scope_tracking(stmt, scope)
            if node.expr is not None:
                _walk_scope_tracking(node.expr, scope)
            return

        if isinstance(node, Identifier) and node.name == name:
            rng = _rename_name_range(name, node.span, source_lines)
            if rng is not None:
                scope_refs.setdefault(current_scope, []).append(rng)
                span_to_scope[_span_key(node.span)] = current_scope

        for child in _children_of(node):
            _walk_scope_tracking(child, current_scope)

    _walk_scope_tracking(fn_scope.body, -1 if param_match else (0 if shadow_let_keys else -1))

    # Determine which scope the cursor belongs to
    target_scope = None
    if cursor_is_param:
        target_scope = -1
    elif cursor_let_key is not None:
        target_scope = span_to_scope.get(cursor_let_key)
    elif cursor_ident_key is not None:
        target_scope = span_to_scope.get(cursor_ident_key)

    if target_scope is None:
        # Fallback: return all refs (old behavior)
        for refs in scope_refs.values():
            for rng in refs:
                ec.add(rng)
    else:
        for rng in scope_refs.get(target_scope, []):
            ec.add(rng)

    return ec.edits


def _rename_collect_struct_edits(
    program: Program, name: str, new_name: str, source_lines: list[str],
) -> list[types.TextEdit]:
    ec = _EditCollector(new_name)
    for sd in program.struct_defs:
        if sd.name == name:
            ec.add(_rename_name_range(name, sd.span, source_lines))

    for fn in program.functions:
        for param in fn.params:
            if param.type_annotation and param.type_annotation.base == name:
                ec.add(_rename_name_range(name, param.type_annotation.span, source_lines))
        if fn.return_type and fn.return_type.base == name:
            ec.add(_rename_name_range(name, fn.return_type.span, source_lines))

        def _visit(node, _name=name):
            if isinstance(node, StructLiteral) and node.name == _name:
                ec.add(_rename_name_range(_name, node.span, source_lines))
            elif isinstance(node, LetStmt) and node.type_annotation and node.type_annotation.base == _name:
                ec.add(_rename_name_range(_name, node.type_annotation.span, source_lines))

        _rename_walk(fn.body, _visit)
    return ec.edits


def prepare_rename_at(
    source: str, result: AnalysisResult, line_0: int, col_0: int,
) -> types.Range | None:
    """Core logic for prepare_rename. Returns the Range of the symbol, or None."""
    if not result or not result.program:
        return None

    line = line_0 + 1
    col = col_0 + 1
    source_lines = source.splitlines()

    struct_names = {sd.name for sd in result.program.struct_defs} if result.program else set()

    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            name, kind = classify_symbol(node, line, col, struct_names=struct_names)
            if name is None:
                return None
            if kind == "function" and name in _BUILTIN_SET:
                return None
            return _rename_name_range(name, node.span, source_lines)

    for sd in result.program.struct_defs:
        if _span_contains(sd.span, line, col):
            return _rename_name_range(sd.name, sd.span, source_lines)

    return None


def rename_at(
    source: str, result: AnalysisResult, line_0: int, col_0: int, new_name: str,
) -> list[types.TextEdit] | None:
    """Core logic for rename. Returns a list of TextEdits, or None."""
    if not result or not result.program:
        return None

    # Validate new_name: must be a valid identifier, not a keyword or type name
    if not new_name or not _VALID_IDENT.match(new_name):
        return None
    if new_name in _KEYWORD_SET or new_name in _TYPE_NAME_SET:
        return None

    line = line_0 + 1
    col = col_0 + 1
    source_lines = source.splitlines()

    node = None
    fn_scope = None

    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            fn_scope = fn
            break

    if node is None:
        for sd in result.program.struct_defs:
            if _span_contains(sd.span, line, col):
                node = sd
                break

    if node is None:
        return None

    struct_names = {sd.name for sd in result.program.struct_defs} if result.program else set()
    name, kind = classify_symbol(node, line, col, struct_names=struct_names)
    if name is None:
        return None
    if kind == "function" and name in _BUILTIN_SET:
        return None

    if kind == "function":
        edits = _rename_collect_function_edits(result.program, name, new_name, source_lines)
    elif kind == "variable":
        if fn_scope is None:
            return None
        edits = _rename_collect_variable_edits(fn_scope, name, new_name, source_lines, cursor_node=node)
    elif kind == "struct":
        edits = _rename_collect_struct_edits(result.program, name, new_name, source_lines)
    else:
        return None

    return edits if edits else None


@server.feature(types.TEXT_DOCUMENT_PREPARE_RENAME)
def prepare_rename(ls: LanguageServer, params: types.PrepareRenameParams):
    logger.debug("prepare_rename: %s at %d:%d", params.text_document.uri,
                 params.position.line, params.position.character)
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None
    doc = ls.workspace.get_text_document(uri)
    return prepare_rename_at(
        doc.source, result, params.position.line, params.position.character,
    )


@server.feature(types.TEXT_DOCUMENT_RENAME)
def rename(ls: LanguageServer, params: types.RenameParams):
    logger.debug("rename: %s at %d:%d to %s", params.text_document.uri,
                 params.position.line, params.position.character, params.new_name)
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None
    doc = ls.workspace.get_text_document(uri)
    edits = rename_at(
        doc.source, result,
        params.position.line, params.position.character,
        params.new_name,
    )
    if edits is None:
        return None
    return types.WorkspaceEdit(changes={uri: edits})
