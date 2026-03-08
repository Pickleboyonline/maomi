from __future__ import annotations

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ..ast_nodes import Block, LetStmt, ExprStmt, ScanExpr, MapExpr, IfExpr
from ..types import MaomiType, StructType
from ._core import server, _cache, AnalysisResult, _local_functions
from ._ast_utils import _span_contains, _find_node_at
from ._builtin_data import (
    _KEYWORDS, _TYPE_NAMES, _BUILTINS, _BUILTIN_SET,
    _BUILTIN_NAMESPACES, _BUILTIN_DOCS,
)


@server.feature(
    types.TEXT_DOCUMENT_COMPLETION,
    types.CompletionOptions(trigger_characters=["."]),
)
def completions(ls: LanguageServer, params: types.CompletionParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    doc = ls.workspace.get_text_document(uri)

    lines = doc.source.splitlines()
    if params.position.line >= len(lines):
        return None
    line_text = lines[params.position.line]
    col = params.position.character

    # Dot context: struct field or module function completion
    if col > 0 and line_text[col - 1] == ".":
        # Extract the word before the dot
        j = col - 2
        while j >= 0 and (line_text[j].isalnum() or line_text[j] == "_"):
            j -= 1
        prefix = line_text[j + 1:col - 1]

        # Check if prefix is a module name (i.e., fn_table has "prefix.something")
        module_result = _complete_module(result, prefix)
        if module_result is not None:
            return module_result

        # Check if prefix is a builtin namespace (e.g., "random.")
        if prefix in _BUILTIN_NAMESPACES:
            items = []
            for short_name in _BUILTIN_NAMESPACES[prefix]:
                full_name = f"{prefix}.{short_name}"
                doc = _BUILTIN_DOCS.get(full_name)
                items.append(types.CompletionItem(
                    label=short_name,
                    kind=types.CompletionItemKind.Function,
                    detail="builtin",
                    documentation=types.MarkupContent(
                        kind=types.MarkupKind.Markdown, value=doc,
                    ) if doc else None,
                ))
            return types.CompletionList(is_incomplete=False, items=items)

        return _complete_dot(result, params.position)

    return _complete_general(result, params.position)


def _complete_dot(result: AnalysisResult | None, position: types.Position):
    if not result or not result.program:
        return None

    line = position.line + 1
    # The dot is at position.character (0-indexed), so the expr before it
    # ends at position.character - 1 (0-indexed) = position.character (1-indexed)
    col = position.character

    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            typ = result.type_map.get(id(node))
            if isinstance(typ, StructType):
                return types.CompletionList(
                    is_incomplete=False,
                    items=[
                        types.CompletionItem(
                            label=fname,
                            kind=types.CompletionItemKind.Field,
                            detail=str(ftype),
                        )
                        for fname, ftype in typ.fields
                    ],
                )
    return None


def _complete_module(result: AnalysisResult | None, module_name: str):
    """Complete functions from an imported module (e.g., 'cnn.' -> cnn.relu, cnn.forward)."""
    if not result or not result.fn_table or not module_name:
        return None
    prefix = module_name + "."
    items: list[types.CompletionItem] = []
    fn_docs = {f.name: f.doc for f in result.program.functions if f.doc} if result.program else {}
    for name, sig in result.fn_table.items():
        if not name.startswith(prefix):
            continue
        short_name = name[len(prefix):]
        params = ", ".join(
            f"{n}: {t}" for n, t in zip(sig.param_names, sig.param_types)
        )
        doc = fn_docs.get(name)
        items.append(types.CompletionItem(
            label=short_name,
            kind=types.CompletionItemKind.Function,
            detail=f"({params}) -> {sig.return_type}",
            documentation=types.MarkupContent(
                kind=types.MarkupKind.Markdown, value=doc,
            ) if doc else None,
        ))
    if not items:
        return None
    return types.CompletionList(is_incomplete=False, items=items)


def _complete_general(result: AnalysisResult | None, position: types.Position):
    items: list[types.CompletionItem] = []

    for kw in _KEYWORDS:
        items.append(types.CompletionItem(
            label=kw, kind=types.CompletionItemKind.Keyword,
        ))

    for t in _TYPE_NAMES:
        items.append(types.CompletionItem(
            label=t, kind=types.CompletionItemKind.TypeParameter,
        ))

    for b in _BUILTINS:
        if "." in b:
            continue  # namespaced builtins offered via dot-completion
        doc = _BUILTIN_DOCS.get(b)
        items.append(types.CompletionItem(
            label=b, kind=types.CompletionItemKind.Function, detail="builtin",
            documentation=types.MarkupContent(
                kind=types.MarkupKind.Markdown, value=doc,
            ) if doc else None,
        ))

    for ns in _BUILTIN_NAMESPACES:
        items.append(types.CompletionItem(
            label=ns, kind=types.CompletionItemKind.Module, detail="builtin namespace",
        ))

    if result and result.program:
        # Imported module names (e.g., "nn" from "nn.relu")
        modules_seen: set[str] = set()
        for name in result.fn_table:
            if "." in name and name not in _BUILTIN_SET:
                mod = name.split(".", 1)[0]
                if mod not in _BUILTIN_NAMESPACES and mod not in modules_seen:
                    modules_seen.add(mod)
                    items.append(types.CompletionItem(
                        label=mod, kind=types.CompletionItemKind.Module,
                        detail="imported module",
                    ))

        # Build fn doc lookup
        fn_docs = {f.name: f.doc for f in result.program.functions if f.doc}
        # User-defined functions
        for name, sig in result.fn_table.items():
            if name in _BUILTIN_SET or "." in name:
                continue
            params = ", ".join(
                f"{n}: {t}" for n, t in zip(sig.param_names, sig.param_types)
            )
            doc = fn_docs.get(name)
            items.append(types.CompletionItem(
                label=name,
                kind=types.CompletionItemKind.Function,
                detail=f"({params}) -> {sig.return_type}",
                documentation=types.MarkupContent(
                    kind=types.MarkupKind.Markdown, value=doc,
                ) if doc else None,
            ))

        # Struct names
        struct_docs = {sd.name: sd.doc for sd in result.program.struct_defs if sd.doc}
        for name in result.struct_defs:
            doc = struct_docs.get(name)
            items.append(types.CompletionItem(
                label=name, kind=types.CompletionItemKind.Struct,
                documentation=types.MarkupContent(
                    kind=types.MarkupKind.Markdown, value=doc,
                ) if doc else None,
            ))

        # Variables in scope
        for var_name, var_type in _vars_in_scope(result, position):
            items.append(types.CompletionItem(
                label=var_name,
                kind=types.CompletionItemKind.Variable,
                detail=str(var_type) if var_type else None,
            ))

    return types.CompletionList(is_incomplete=False, items=items)


def _vars_in_scope(
    result: AnalysisResult, position: types.Position
) -> list[tuple[str, MaomiType | None]]:
    if not result.program:
        return []

    line = position.line + 1
    col = position.character + 1
    variables: list[tuple[str, MaomiType | None]] = []

    for fn in _local_functions(result.program):
        if not _span_contains(fn.span, line, col):
            continue

        # Function params
        sig = result.fn_table.get(fn.name)
        if sig:
            for pname, ptype in zip(sig.param_names, sig.param_types):
                variables.append((pname, ptype))

        # Let bindings and loop vars in scope
        _collect_scope_vars(fn.body, line, col, result.type_map, variables)
        break

    return variables


def _collect_scope_vars(
    block: Block, line: int, col: int,
    type_map: dict[int, MaomiType], variables: list,
):
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt):
            # Only include let bindings that appear before the cursor
            if stmt.span.line_start < line or (
                stmt.span.line_start == line and stmt.span.col_end < col
            ):
                typ = type_map.get(id(stmt.value))
                variables.append((stmt.name, typ))

        if isinstance(stmt, ExprStmt):
            _collect_from_expr(stmt.expr, line, col, type_map, variables)

    if block.expr is not None:
        _collect_from_expr(block.expr, line, col, type_map, variables)


def _collect_from_expr(
    expr, line: int, col: int,
    type_map: dict[int, MaomiType], variables: list,
):
    if not hasattr(expr, "span") or not _span_contains(expr.span, line, col):
        return

    if isinstance(expr, ScanExpr):
        # carry_var and elem_vars are in scope inside the body
        variables.append((expr.carry_var, None))
        for ev in expr.elem_vars:
            variables.append((ev, None))
        _collect_scope_vars(expr.body, line, col, type_map, variables)

    elif isinstance(expr, MapExpr):
        variables.append((expr.elem_var, None))
        _collect_scope_vars(expr.body, line, col, type_map, variables)

    elif isinstance(expr, IfExpr):
        _collect_scope_vars(expr.then_block, line, col, type_map, variables)
        _collect_scope_vars(expr.else_block, line, col, type_map, variables)
