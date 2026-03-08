from __future__ import annotations

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ..ast_nodes import (
    Program, FnDef, LetStmt, Param,
    CallExpr,
    Identifier, StructLiteral, StructDef,
)
from ._core import server, _cache, AnalysisResult, _local_functions
from ._ast_utils import _span_contains, _find_node_at, _children_of, classify_symbol
from ._builtin_data import _BUILTIN_SET


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
) -> list[types.TextEdit]:
    ec = _EditCollector(new_name)
    for param in fn_scope.params:
        if param.name == name:
            ec.add(_rename_name_range(name, param.span, source_lines))

    def _visit(node, _name=name):
        if isinstance(node, Identifier) and node.name == _name:
            ec.add(_rename_name_range(_name, node.span, source_lines))
        elif isinstance(node, LetStmt) and node.name == _name:
            ec.add(_rename_name_range(_name, node.span, source_lines))

    _rename_walk(fn_scope.body, _visit)
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

    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            name, kind = classify_symbol(node)
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

    name, kind = classify_symbol(node)
    if name is None:
        return None
    if kind == "function" and name in _BUILTIN_SET:
        return None

    if kind == "function":
        edits = _rename_collect_function_edits(result.program, name, new_name, source_lines)
    elif kind == "variable":
        if fn_scope is None:
            return None
        edits = _rename_collect_variable_edits(fn_scope, name, new_name, source_lines)
    elif kind == "struct":
        edits = _rename_collect_struct_edits(result.program, name, new_name, source_lines)
    else:
        return None

    return edits if edits else None


@server.feature(types.TEXT_DOCUMENT_PREPARE_RENAME)
def prepare_rename(ls: LanguageServer, params: types.PrepareRenameParams):
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
