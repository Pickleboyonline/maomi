from __future__ import annotations

import logging

from lsprotocol import types

from ..ast_nodes import (
    FnDef, LetStmt, Param, CallExpr, Identifier, ImportDecl, FieldAccess,
)
from ._core import server, _cache, AnalysisResult, _local_functions
from ._ast_utils import _find_node_at, _span_contains
from ._builtin_data import _BUILTIN_SIGNATURES, _BUILTIN_DOCS
from ._completion import _find_module_file

logger = logging.getLogger("maomi-lsp")


def _fmt_annotation(ta) -> str:
    """Format a TypeAnnotation as a readable string like 'f32[B, N]'."""
    if ta is None:
        return "?"
    if getattr(ta, 'wildcard', False):
        return f"{ta.base}[..]"
    if not ta.dims:
        return ta.base
    dims = ", ".join(str(d.value) for d in ta.dims)
    return f"{ta.base}[{dims}]"


def _get_hover_text(node, fn: FnDef, result: AnalysisResult) -> str | None:
    # LetStmt: show "let name: type" (skip compiler-generated names)
    if isinstance(node, LetStmt):
        if node.name.startswith("__maomi"):
            return None
        typ = result.type_map.get(id(node.value))
        if typ is not None:
            return f"```maomi\nlet {node.name}: {typ}\n```"
        return None

    # Param: show "param name: type"
    if isinstance(node, Param):
        prefix = "comptime " if getattr(node, 'comptime', False) else ""
        type_str = _fmt_annotation(node.type_annotation)
        return f"```maomi\n{prefix}{node.name}: {type_str}\n```"

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
        # Strip monomorphization suffix (e.g., "identity$3" → "identity")
        display_name = callee.split("$")[0] if "$" in callee else callee
        # Check builtins first
        builtin = _BUILTIN_SIGNATURES.get(callee)
        if builtin is not None:
            pnames, ptypes, ret = builtin
            params = ", ".join(f"{n}: {t}" for n, t in zip(pnames, ptypes))
            text = f"```maomi\nfn {display_name}({params}) -> {ret}\n```"
            doc = _BUILTIN_DOCS.get(callee)
            if doc:
                text += f"\n\n{doc}"
            return text
        # User-defined function
        sig = result.fn_table.get(callee)
        if sig is not None:
            params = ", ".join(f"{n}: {t}" for n, t in zip(sig.param_names, sig.param_types))
            text = f"```maomi\nfn {display_name}({params}) -> {sig.return_type}\n```"
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
                    text = f"```maomi\nfn {display_name}({params}) -> {ret}\n```"
                    if f.doc:
                        text += f"\n\n{f.doc}"
                    return text

    # FieldAccess: show "(field) name: type"
    if isinstance(node, FieldAccess):
        typ = result.type_map.get(id(node))
        if typ is not None:
            return f"```maomi\n(field) {node.field}: {typ}\n```"

    # General expression: look up type_map
    typ = result.type_map.get(id(node))
    if typ is not None:
        if isinstance(node, Identifier):
            return f"```maomi\n{node.name}: {typ}\n```"
        return f"```maomi\n{typ}\n```"
    return None


def _hover_import(result: AnalysisResult, line: int, col: int, filepath: str) -> str | None:
    """Check if cursor is on an import and return hover text."""
    for imp in result.program.imports:
        # Hover on module name → show module path and exports
        if imp.module_span and _span_contains(imp.module_span, line, col):
            mod_path = _find_module_file(imp.module_path, filepath)
            if mod_path is None:
                return f"Module `{imp.module_path}` (not found)"
            text = f"**module** `{imp.module_path}`\n\n`{mod_path}`"
            # Show exports
            try:
                from ..lexer import Lexer
                from ..parser import Parser
                source = open(mod_path).read()
                tokens = Lexer(source, filename=mod_path).tokenize()
                parser = Parser(tokens, filename=mod_path)
                program = parser.parse()
                fns = [f.name for f in program.functions]
                structs = [s.name for s in program.struct_defs]
                if fns:
                    text += "\n\n**Functions:** " + ", ".join(f"`{n}`" for n in fns)
                if structs:
                    text += "\n\n**Structs:** " + ", ".join(f"`{n}`" for n in structs)
            except Exception:
                pass
            return text

        # Hover on imported name → show function/struct signature from module
        if imp.names and imp.name_spans:
            for name, span in zip(imp.names, imp.name_spans):
                if _span_contains(span, line, col):
                    # Look up in fn_table (resolved program has it)
                    qualified = f"{imp.alias or imp.module_path}.{name}"
                    sig = result.fn_table.get(qualified) or result.fn_table.get(name)
                    if sig:
                        params = ", ".join(
                            f"{n}: {t}" for n, t in zip(sig.param_names, sig.param_types)
                        )
                        return f"```maomi\nfn {name}({params}) -> {sig.return_type}\n```"
                    # Check if it's a struct
                    if name in result.struct_defs or qualified in result.struct_defs:
                        sd = result.struct_defs.get(name) or result.struct_defs.get(qualified)
                        if sd:
                            fields = ", ".join(f"{n}: {t}" for n, t in sd.fields)
                            return f"```maomi\nstruct {name} {{ {fields} }}\n```"
                    return f"`{name}` (from `{imp.module_path}`)"

        # Hover on alias name
        if imp.alias_span and _span_contains(imp.alias_span, line, col):
            mod_path = _find_module_file(imp.module_path, filepath)
            return f"**module alias** `{imp.alias}` → `{imp.module_path}`\n\n`{mod_path or 'not found'}`"

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

    # Check imports first
    from ._core import _uri_to_path
    filepath = _uri_to_path(uri)
    import_hover = _hover_import(result, line, col, filepath)
    if import_hover is not None:
        return types.Hover(
            contents=types.MarkupContent(
                kind=types.MarkupKind.Markdown,
                value=import_hover,
            )
        )

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

    # Top-level struct definitions
    for sd in result.program.struct_defs:
        if sd.canonical_name is not None:
            continue  # skip imported struct aliases
        if _span_contains(sd.span, line, col):
            fields = ", ".join(f"{n}: {t}" for n, t in sd.fields)
            text = f"```maomi\nstruct {sd.name} {{ {fields} }}\n```"
            if sd.doc:
                text += f"\n\n{sd.doc}"
            return types.Hover(
                contents=types.MarkupContent(kind=types.MarkupKind.Markdown, value=text)
            )

    # Top-level type aliases
    for ta in result.program.type_aliases:
        if _span_contains(ta.span, line, col):
            text = f"```maomi\ntype {ta.name} = {_fmt_annotation(ta.type_annotation)}\n```"
            return types.Hover(
                contents=types.MarkupContent(kind=types.MarkupKind.Markdown, value=text)
            )

    return None
