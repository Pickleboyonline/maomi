from __future__ import annotations

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ._core import server, _cache, AnalysisResult, _local_functions
from ._ast_utils import _span_to_range


def _build_document_symbols(result: AnalysisResult) -> list[types.DocumentSymbol] | None:
    if not result or not result.program:
        return None

    symbols: list[types.DocumentSymbol] = []

    for sd in result.program.struct_defs:
        r = _span_to_range(sd.span)
        children = []
        for field_name, field_type_ann in sd.fields:
            children.append(types.DocumentSymbol(
                name=field_name,
                kind=types.SymbolKind.Property,
                range=r,
                selection_range=r,
            ))
        symbols.append(types.DocumentSymbol(
            name=sd.name,
            kind=types.SymbolKind.Struct,
            range=r,
            selection_range=r,
            children=children,
        ))

    for fn in _local_functions(result.program):
        r = _span_to_range(fn.span)
        children = []
        for param in fn.params:
            pr = _span_to_range(param.span)
            children.append(types.DocumentSymbol(
                name=param.name,
                kind=types.SymbolKind.Variable,
                range=pr,
                selection_range=pr,
            ))
        symbols.append(types.DocumentSymbol(
            name=fn.name,
            kind=types.SymbolKind.Function,
            range=r,
            selection_range=r,
            children=children,
        ))

    return symbols


def _workspace_symbols(query: str) -> list[types.SymbolInformation]:
    """Search for functions/structs across all cached files."""
    result_symbols: list[types.SymbolInformation] = []
    q = query.lower()

    for uri, analysis in _cache.items():
        if not analysis.program:
            continue

        for sd in analysis.program.struct_defs:
            if q and q not in sd.name.lower():
                continue
            result_symbols.append(types.SymbolInformation(
                name=sd.name,
                kind=types.SymbolKind.Struct,
                location=types.Location(uri=uri, range=_span_to_range(sd.span)),
            ))

        for fn in _local_functions(analysis.program):
            if q and q not in fn.name.lower():
                continue
            result_symbols.append(types.SymbolInformation(
                name=fn.name,
                kind=types.SymbolKind.Function,
                location=types.Location(uri=uri, range=_span_to_range(fn.span)),
            ))

    return result_symbols


@server.feature(types.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbols(ls: LanguageServer, params: types.DocumentSymbolParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    return _build_document_symbols(result)


@server.feature(types.WORKSPACE_SYMBOL)
def workspace_symbols(ls: LanguageServer, params: types.WorkspaceSymbolParams):
    return _workspace_symbols(params.query)
