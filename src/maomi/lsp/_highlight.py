from __future__ import annotations

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ._core import server, _cache, AnalysisResult, _local_functions
from ._ast_utils import _span_contains, _find_node_at, _span_to_range
from ._references import _refs_classify_node, _refs_collect_all


def _spans_to_highlights(result, name, kind, fn_scope=None):
    """Convert symbol spans to DocumentHighlight items with Read/Write classification."""
    all_spans = _refs_collect_all(result, name, kind,
                                  include_declaration=True, fn_scope=fn_scope)
    decl_spans = set()
    if all_spans:
        usage_spans = _refs_collect_all(result, name, kind,
                                        include_declaration=False, fn_scope=fn_scope)
        usage_set = set(usage_spans)
        decl_spans = {s for s in all_spans if s not in usage_set}
    highlights = []
    for s in all_spans:
        hk = (types.DocumentHighlightKind.Write
               if s in decl_spans
               else types.DocumentHighlightKind.Read)
        highlights.append(types.DocumentHighlight(
            range=_span_to_range(s), kind=hk))
    return highlights or None


def _build_document_highlights(result: AnalysisResult, line: int, col: int) -> list[types.DocumentHighlight] | None:
    """Return highlights for all occurrences of the symbol at (line, col)."""
    if not result or not result.program:
        return None

    # Check struct definitions first (same pattern as references handler).
    for sd in result.program.struct_defs:
        if _span_contains(sd.span, line, col):
            return _spans_to_highlights(result, sd.name, "struct")

    # Check inside local functions.
    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            name, kind = _refs_classify_node(node, line, col)
            if name:
                fn_scope = fn if kind == "variable" else None
                return _spans_to_highlights(result, name, kind, fn_scope)


@server.feature(types.TEXT_DOCUMENT_DOCUMENT_HIGHLIGHT)
def document_highlight(ls: LanguageServer, params: types.DocumentHighlightParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None
    line = params.position.line + 1
    col = params.position.character + 1
    return _build_document_highlights(result, line, col)
