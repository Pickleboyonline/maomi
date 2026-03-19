from __future__ import annotations

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ._core import server, _cache
from ._builtin_data import _BUILTIN_SIGNATURES, _BUILTIN_DOCS


def _sig_parse_call_context(
    source: str, position: types.Position,
) -> tuple[str | None, int, str | None]:
    """Parse source text backward from cursor to find function name and active param index.

    Returns ``(func_name, comma_count, named_param)``.  *named_param* is the
    parameter name when the cursor is inside a named argument (e.g. ``axis=1``),
    otherwise ``None``.
    """
    lines = source.splitlines()
    if position.line >= len(lines):
        return None, 0, None

    depth = 0
    comma_count = 0

    line = position.line
    col = position.character
    line_text = lines[line]
    i = min(col - 1, len(line_text) - 1)

    # Track the start of the current argument (for named-arg detection)
    arg_start: int | None = None
    cursor_line = line

    while line >= 0:
        while i >= 0:
            ch = line_text[i]
            # Skip characters inside string literals
            if ch == '"' and (i == 0 or line_text[i - 1] != '\\'):
                # Scan backward to find the matching quote
                i -= 1
                while i >= 0:
                    if line_text[i] == '"' and (i == 0 or line_text[i - 1] != '\\'):
                        break
                    i -= 1
                i -= 1
                continue
            if ch == ')':
                depth += 1
            elif ch == '(':
                if depth == 0:
                    # Found the opening paren — extract function name
                    if arg_start is None and line == cursor_line:
                        arg_start = i + 1
                    j = i - 1
                    while j >= 0 and line_text[j] == ' ':
                        j -= 1
                    end = j + 1
                    while j >= 0 and (line_text[j].isalnum() or line_text[j] in ('_', '.')):
                        j -= 1
                    name = line_text[j + 1:end]
                    if name:
                        # Check for named argument on the cursor line
                        named_param: str | None = None
                        if arg_start is not None and line == cursor_line:
                            current_arg = lines[cursor_line][arg_start:col].strip()
                            if '=' in current_arg:
                                named_param = current_arg.split('=')[0].strip()
                        return name, comma_count, named_param
                    return None, 0, None
                depth -= 1
            elif ch == ',' and depth == 0:
                if comma_count == 0 and line == cursor_line:
                    arg_start = i + 1
                comma_count += 1
            i -= 1

        line -= 1
        if line >= 0:
            line_text = lines[line]
            i = len(line_text) - 1

    return None, 0, None


def _resolve_active_param(
    comma_count: int,
    named_param: str | None,
    param_names: list[str],
) -> int:
    """Resolve the active parameter index, handling named arguments."""
    if named_param is not None:
        try:
            return param_names.index(named_param)
        except ValueError:
            pass
    return comma_count


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

    callee, comma_count, named_param = _sig_parse_call_context(doc.source, params.position)
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
            active_param = _resolve_active_param(comma_count, named_param, sig.param_names)
            return _build_signature_help(
                callee, sig.param_names, sig.param_types, sig.return_type, active_param,
                doc=fn_doc,
            )

    builtin = _BUILTIN_SIGNATURES.get(callee)
    if builtin is not None:
        pnames, ptypes, ret = builtin
        active_param = _resolve_active_param(comma_count, named_param, pnames)
        return _build_signature_help(
            callee, pnames, ptypes, ret, active_param,
            doc=_BUILTIN_DOCS.get(callee),
        )

    return None
