from __future__ import annotations

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ..ast_nodes import Block, LetStmt, ExprStmt, ScanExpr, MapExpr, IfExpr, FoldExpr, WhileExpr
from ._core import server, _cache, AnalysisResult, _local_functions


def _inlay_collect_hints(
    block: Block, type_map: dict, start_line: int, end_line: int,
    hints: list, source_lines: list[str],
):
    """Collect inlay hints from a block (recursive into nested blocks)."""
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt) and stmt.type_annotation is None:
            typ = type_map.get(id(stmt.value))
            if typ is not None and start_line <= stmt.span.line_start <= end_line:
                line_idx = stmt.span.line_start - 1
                line_text = source_lines[line_idx]
                search_start = stmt.span.col_start - 1
                name_start = line_text.find("let ", search_start)
                if name_start >= 0:
                    name_end = name_start + 4 + len(stmt.name)
                    hints.append(types.InlayHint(
                        position=types.Position(line=line_idx, character=name_end),
                        label=f": {typ}",
                        kind=types.InlayHintKind.Type,
                        padding_left=False,
                        padding_right=True,
                    ))

        if isinstance(stmt, ExprStmt):
            _inlay_collect_from_expr(
                stmt.expr, type_map, start_line, end_line, hints, source_lines,
            )

    if block.expr is not None:
        _inlay_collect_from_expr(
            block.expr, type_map, start_line, end_line, hints, source_lines,
        )


def _inlay_collect_from_expr(expr, type_map, start_line, end_line, hints, source_lines):
    """Recurse into expressions that contain blocks."""
    if isinstance(expr, ScanExpr):
        _inlay_collect_hints(
            expr.body, type_map, start_line, end_line, hints, source_lines,
        )
    elif isinstance(expr, MapExpr):
        _inlay_collect_hints(
            expr.body, type_map, start_line, end_line, hints, source_lines,
        )
    elif isinstance(expr, FoldExpr):
        _inlay_collect_hints(
            expr.body, type_map, start_line, end_line, hints, source_lines,
        )
    elif isinstance(expr, WhileExpr):
        _inlay_collect_hints(
            expr.cond, type_map, start_line, end_line, hints, source_lines,
        )
        _inlay_collect_hints(
            expr.body, type_map, start_line, end_line, hints, source_lines,
        )
    elif isinstance(expr, IfExpr):
        _inlay_collect_hints(
            expr.then_block, type_map, start_line, end_line, hints, source_lines,
        )
        _inlay_collect_hints(
            expr.else_block, type_map, start_line, end_line, hints, source_lines,
        )


def _build_inlay_hints(
    result: AnalysisResult, start_line: int, end_line: int, source: str,
) -> list[types.InlayHint]:
    """Build inlay hints for the given 1-indexed line range."""
    if not result.program:
        return []
    source_lines = source.splitlines()
    hints: list[types.InlayHint] = []
    for fn in _local_functions(result.program):
        _inlay_collect_hints(
            fn.body, result.type_map, start_line, end_line, hints, source_lines,
        )
    return hints


@server.feature(types.TEXT_DOCUMENT_INLAY_HINT)
def inlay_hints(ls: LanguageServer, params: types.InlayHintParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None

    start_line = params.range.start.line + 1
    end_line = params.range.end.line + 1
    doc = ls.workspace.get_text_document(uri)

    return _build_inlay_hints(result, start_line, end_line, doc.source)
