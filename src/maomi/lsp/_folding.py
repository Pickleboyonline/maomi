from __future__ import annotations

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ..ast_nodes import FnDef, Block, StructDef, ScanExpr, MapExpr
from ._core import server, _cache, AnalysisResult, _local_functions
from ._ast_utils import _children_of


def _fold_collect_ranges(node, ranges: list):
    """Recursively collect folding ranges from AST nodes."""
    if hasattr(node, 'span') and node.span.line_end > node.span.line_start:
        if isinstance(node, (StructDef, ScanExpr, MapExpr, Block)):
            ranges.append(types.FoldingRange(
                start_line=node.span.line_start - 1,
                end_line=node.span.line_end - 1,
                kind=types.FoldingRangeKind.Region,
            ))

    for child in _children_of(node):
        _fold_collect_ranges(child, ranges)


def _build_folding_ranges(result: AnalysisResult) -> list[types.FoldingRange]:
    if not result or not result.program:
        return []

    ranges: list[types.FoldingRange] = []

    for sd in result.program.struct_defs:
        _fold_collect_ranges(sd, ranges)

    for fn in _local_functions(result.program):
        _fold_collect_ranges_fn(fn, ranges)

    return ranges


def _fold_collect_ranges_fn(fn: FnDef, ranges: list):
    if fn.span.line_end > fn.span.line_start:
        ranges.append(types.FoldingRange(
            start_line=fn.span.line_start - 1,
            end_line=fn.span.line_end - 1,
            kind=types.FoldingRangeKind.Region,
        ))

    if fn.body:
        for child in _children_of(fn.body):
            _fold_collect_ranges(child, ranges)


@server.feature(types.TEXT_DOCUMENT_FOLDING_RANGE)
def folding_ranges(ls: LanguageServer, params: types.FoldingRangeParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None

    ranges = _build_folding_ranges(result)
    return ranges if ranges else None
