from __future__ import annotations

import logging

from lsprotocol import types

from ..ast_nodes import (
    FnDef, LetStmt, Param, CallExpr, Identifier,
)
from ._core import server, _cache, AnalysisResult, _local_functions
from ._ast_utils import _find_node_at
from ._builtin_data import _BUILTIN_SIGNATURES, _BUILTIN_DOCS

logger = logging.getLogger("maomi-lsp")


def _fmt_annotation(ta) -> str:
    """Format a TypeAnnotation as a readable string like 'f32[B, N]'."""
    if ta is None:
        return "?"
    if not ta.dims:
        return ta.base
    dims = ", ".join(str(d.value) for d in ta.dims)
    return f"{ta.base}[{dims}]"


def _get_hover_text(node, fn: FnDef, result: AnalysisResult) -> str | None:
    # LetStmt: show "let name: type"
    if isinstance(node, LetStmt):
        typ = result.type_map.get(id(node.value))
        if typ is not None:
            return f"```maomi\nlet {node.name}: {typ}\n```"
        return None

    # Param: show "param name: type"
    if isinstance(node, Param):
        return f"```maomi\n{node.name}: {node.type_annotation.base}" + (
            f"[{', '.join(str(d.value) for d in node.type_annotation.dims)}]"
            if node.type_annotation.dims else ""
        ) + "\n```"

    # FnDef: show full signature + doc
    if isinstance(node, FnDef):
        sig = result.fn_table.get(node.name)
        if sig is not None:
            params = ", ".join(f"{n}: {t}" for n, t in zip(sig.param_names, sig.param_types))
            text = f"```maomi\nfn {node.name}({params}) -> {sig.return_type}\n```"
        else:
            # Fallback for generic functions (symbolic dims) — build from AST
            params = ", ".join(f"{p.name}: {_fmt_annotation(p.type_annotation)}" for p in node.params)
            ret = _fmt_annotation(node.return_type) if node.return_type else "?"
            text = f"```maomi\nfn {node.name}({params}) -> {ret}\n```"
        if node.doc:
            text += f"\n\n{node.doc}"
        return text

    # CallExpr: show signature + doc for builtins and user functions
    if isinstance(node, CallExpr):
        callee = node.callee
        # Check builtins first
        builtin = _BUILTIN_SIGNATURES.get(callee)
        if builtin is not None:
            pnames, ptypes, ret = builtin
            params = ", ".join(f"{n}: {t}" for n, t in zip(pnames, ptypes))
            text = f"```maomi\nfn {callee}({params}) -> {ret}\n```"
            doc = _BUILTIN_DOCS.get(callee)
            if doc:
                text += f"\n\n{doc}"
            return text
        # User-defined function
        sig = result.fn_table.get(callee)
        if sig is not None:
            params = ", ".join(f"{n}: {t}" for n, t in zip(sig.param_names, sig.param_types))
            text = f"```maomi\nfn {callee}({params}) -> {sig.return_type}\n```"
            if result.program:
                for f in result.program.functions:
                    if f.name == callee and f.doc:
                        text += f"\n\n{f.doc}"
                        break
            return text
        # Fallback for generic functions (symbolic dims) — build from AST
        if result.program:
            for f in result.program.functions:
                if f.name == callee:
                    params = ", ".join(f"{p.name}: {_fmt_annotation(p.type_annotation)}" for p in f.params)
                    ret = _fmt_annotation(f.return_type) if f.return_type else "?"
                    text = f"```maomi\nfn {callee}({params}) -> {ret}\n```"
                    if f.doc:
                        text += f"\n\n{f.doc}"
                    return text

    # General expression: look up type_map
    typ = result.type_map.get(id(node))
    if typ is not None:
        # For identifiers, show "name: type"
        if isinstance(node, Identifier):
            return f"```maomi\n{node.name}: {typ}\n```"
        return f"```maomi\n{typ}\n```"
    return None


@server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(ls, params: types.HoverParams):
    logger.debug("hover: %s at %d:%d", params.text_document.uri,
                 params.position.line, params.position.character)
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None

    # Convert 0-indexed LSP → 1-indexed Maomi
    line = params.position.line + 1
    col = params.position.character + 1

    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            hover_text = _get_hover_text(node, fn, result)
            if hover_text is not None:
                return types.Hover(
                    contents=types.MarkupContent(
                        kind=types.MarkupKind.Markdown,
                        value=hover_text,
                    )
                )
    return None
