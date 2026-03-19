from __future__ import annotations

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ..types import ArrayType, WildcardArrayType, StructType, StructArrayType
from ._core import server, _cache, AnalysisResult, _local_functions
from ._ast_utils import _span_to_range
from ._references import _refs_collect_all


def _build_code_lenses(result: AnalysisResult, uri: str) -> list[types.CodeLens]:
    """Build code lenses for each function: 'Run' and 'N references'."""
    if not result.program:
        return []
    lenses: list[types.CodeLens] = []

    for fn in _local_functions(result.program):
        line = fn.span.line_start - 1  # 1-indexed -> 0-indexed
        lens_range = types.Range(
            start=types.Position(line=line, character=0),
            end=types.Position(line=line, character=0),
        )

        # "Run" lens -- only if all params have concrete types (no symbolic dims)
        sig = result.fn_table.get(fn.name)
        if sig is not None:
            all_concrete = all(
                not isinstance(pt, WildcardArrayType)
                and not isinstance(pt, (StructType, StructArrayType))
                and not (isinstance(pt, ArrayType) and any(isinstance(d, str) for d in pt.dims))
                for pt in sig.param_types
            )
            if all_concrete:
                lenses.append(types.CodeLens(
                    range=lens_range,
                    command=types.Command(
                        title="\u25b6 Run",
                        command="maomi.run",
                        arguments=[uri, fn.name],
                    ),
                ))

        # "N references" lens
        refs = _refs_collect_all(result, fn.name, "function", include_declaration=False)
        count = len(refs)
        title = f"{count} reference{'s' if count != 1 else ''}"
        lenses.append(types.CodeLens(
            range=lens_range,
            command=types.Command(
                title=title,
                command="",
            ),
        ))

    return lenses


@server.feature(types.TEXT_DOCUMENT_CODE_LENS)
def code_lens(ls: LanguageServer, params: types.CodeLensParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None
    return _build_code_lenses(result, uri)
