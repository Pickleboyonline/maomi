from __future__ import annotations

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ._core import server, _cache
from ._builtin_data import _BUILTIN_SIGNATURES, _BUILTIN_DOCS


def _sig_parse_call_context(source: str, position: types.Position) -> tuple[str | None, int]:
    """Parse source text backward from cursor to find function name and active param index."""
    lines = source.splitlines()
    if position.line >= len(lines):
        return None, 0

    depth = 0
    comma_count = 0

    line = position.line
    col = position.character
    line_text = lines[line]
    i = min(col - 1, len(line_text) - 1)

    while line >= 0:
        while i >= 0:
            ch = line_text[i]
            if ch == ')':
                depth += 1
            elif ch == '(':
                if depth == 0:
                    j = i - 1
                    while j >= 0 and line_text[j] == ' ':
                        j -= 1
                    end = j + 1
                    while j >= 0 and (line_text[j].isalnum() or line_text[j] in ('_', '.')):
                        j -= 1
                    name = line_text[j + 1:end]
                    if name:
                        return name, comma_count
                    return None, 0
                depth -= 1
            elif ch == ',' and depth == 0:
                comma_count += 1
            i -= 1

        line -= 1
        if line >= 0:
            line_text = lines[line]
            i = len(line_text) - 1

    return None, 0


def _build_signature_help(
    callee: str, pnames: list[str], ptypes: list, ret, active_param: int,
    doc: str | None = None,
) -> types.SignatureHelp:
    params_info = [
        types.ParameterInformation(label=f"{n}: {t}")
        for n, t in zip(pnames, ptypes)
    ]
    label = f"{callee}({', '.join(f'{n}: {t}' for n, t in zip(pnames, ptypes))}) -> {ret}"
    sig_info = types.SignatureInformation(
        label=label, parameters=params_info,
        documentation=types.MarkupContent(
            kind=types.MarkupKind.Markdown, value=doc,
        ) if doc else None,
    )
    return types.SignatureHelp(
        signatures=[sig_info],
        active_parameter=min(active_param, len(params_info) - 1) if params_info else 0,
    )


@server.feature(
    types.TEXT_DOCUMENT_SIGNATURE_HELP,
    types.SignatureHelpOptions(trigger_characters=["(", ","]),
)
def signature_help(ls: LanguageServer, params: types.SignatureHelpParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    doc = ls.workspace.get_text_document(uri)

    callee, active_param = _sig_parse_call_context(doc.source, params.position)
    if callee is None:
        return None

    if result and result.fn_table:
        sig = result.fn_table.get(callee)
        if sig is not None:
            fn_doc = None
            if result.program:
                for f in result.program.functions:
                    if f.name == callee:
                        fn_doc = f.doc
                        break
            return _build_signature_help(
                callee, sig.param_names, sig.param_types, sig.return_type, active_param,
                doc=fn_doc,
            )

    builtin = _BUILTIN_SIGNATURES.get(callee)
    if builtin is not None:
        pnames, ptypes, ret = builtin
        return _build_signature_help(
            callee, pnames, ptypes, ret, active_param,
            doc=_BUILTIN_DOCS.get(callee),
        )

    return None
