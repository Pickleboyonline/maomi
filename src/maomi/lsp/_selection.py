from __future__ import annotations

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ._core import server, _cache, _local_functions
from ._ast_utils import _span_contains, _span_to_range, _children_of


def _sel_collect_ancestors(node, line: int, col: int, ancestors: list) -> bool:
    """Walk AST depth-first, collecting ancestor nodes that contain the position."""
    if not hasattr(node, "span") or not _span_contains(node.span, line, col):
        return False

    ancestors.append(node)

    for child in _children_of(node):
        if _sel_collect_ancestors(child, line, col, ancestors):
            return True

    return True


def _sel_build_chain(ancestors: list) -> types.SelectionRange | None:
    """Build a SelectionRange linked list from a list of ancestors (outermost first)."""
    if not ancestors:
        return None

    chain = None
    for node in ancestors:
        r = _span_to_range(node.span)
        chain = types.SelectionRange(range=r, parent=chain)

    return chain


@server.feature(types.TEXT_DOCUMENT_SELECTION_RANGE)
def selection_ranges(ls: LanguageServer, params: types.SelectionRangeParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None

    results: list[types.SelectionRange] = []

    for pos in params.positions:
        line = pos.line + 1
        col = pos.character + 1

        ancestors: list = []
        for fn in _local_functions(result.program):
            if _sel_collect_ancestors(fn, line, col, ancestors):
                break

        if not ancestors:
            for sd in result.program.struct_defs:
                if hasattr(sd, "span") and _span_contains(sd.span, line, col):
                    ancestors.append(sd)
                    break

        chain = _sel_build_chain(ancestors)
        if chain is None:
            chain = types.SelectionRange(
                range=types.Range(start=pos, end=pos),
                parent=None,
            )

        results.append(chain)

    return results
