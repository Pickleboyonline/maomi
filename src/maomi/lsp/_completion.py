from __future__ import annotations

import logging

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ..ast_nodes import Block, LetStmt, ExprStmt, ScanExpr, MapExpr, IfExpr
from ..types import MaomiType, ScalarType, ArrayType, StructType, WildcardArrayType, TypeVar, FLOAT_BASES
from ._core import server, _cache, AnalysisResult, _local_functions
from ._ast_utils import _span_contains, _find_node_at
from ._builtin_data import (
    _KEYWORDS, _TYPE_NAMES, _BUILTINS, _BUILTIN_SET,
    _BUILTIN_NAMESPACES, _BUILTIN_DOCS, _BUILTIN_CATEGORIES, _EW_NAMES,
    _BUILTIN_SIGNATURES,
)
from ..builtins import COMPLEX as _CX_REGISTRY

logger = logging.getLogger("maomi-lsp")


@server.feature(
    types.TEXT_DOCUMENT_COMPLETION,
    types.CompletionOptions(trigger_characters=["."]),
)
def completions(ls: LanguageServer, params: types.CompletionParams):
    logger.debug("completions: %s at %d:%d", params.text_document.uri,
                 params.position.line, params.position.character)
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

        return _complete_dot(result, params.position, prefix)

    return _complete_general(result, params.position)


def _complete_dot(result: AnalysisResult | None, position: types.Position, prefix: str = ""):
    if not result or not result.program:
        return None

    line = position.line + 1
    # The dot is at position.character (0-indexed), so the expr before it
    # ends at position.character - 1 (0-indexed) = position.character (1-indexed)
    col = position.character

    typ: MaomiType | None = None

    # Try AST-based lookup first (works when the dot is part of a parsed expression)
    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            typ = result.type_map.get(id(node))
            break

    # Fallback: look up the identifier in scope by name
    # (needed when the line doesn't parse, e.g. user just typed "a.")
    if typ is None and prefix:
        # Try precise scope lookup first
        for var_name, var_type in _vars_in_scope(result, position):
            if var_name == prefix and var_type is not None:
                typ = var_type
                break
        # If scope lookup fails (cursor outside cached function span due to
        # added lines), scan all function parameters from fn_table
        if typ is None:
            for fn in result.program.functions:
                sig = result.fn_table.get(fn.name)
                if sig:
                    for pname, ptype in zip(sig.param_names, sig.param_types):
                        if pname == prefix:
                            typ = ptype
                            break
                if typ is not None:
                    break

    if typ is None:
        return None

    items: list[types.CompletionItem] = []

    # Struct fields (sorted first)
    if isinstance(typ, StructType):
        for fname, ftype in typ.fields:
            items.append(types.CompletionItem(
                label=fname,
                kind=types.CompletionItemKind.Field,
                detail=str(ftype),
                sort_text=f"0_{fname}",
            ))

    # Pipe-compatible functions
    items.extend(_pipe_completions(result, typ, position))

    if items:
        return types.CompletionList(is_incomplete=False, items=items)
    return None


def _is_pipe_compatible(
    expr_type: MaomiType, fn_name: str, sig: "FnSignature",
) -> bool:
    """Check if a function's first parameter is compatible with expr_type for pipe completion."""
    if not sig.param_types:
        return False
    first = sig.param_types[0]

    # Generic functions (TypeVar first param) match anything
    if isinstance(first, TypeVar):
        return True

    # Exact type match
    if first == expr_type:
        return True

    # Elementwise builtins accept any float scalar/array/struct
    if fn_name in _EW_NAMES:
        if isinstance(expr_type, (ScalarType, ArrayType)):
            return expr_type.base in FLOAT_BASES
        if isinstance(expr_type, WildcardArrayType):
            return expr_type.base in FLOAT_BASES
        if isinstance(expr_type, StructType):
            return True
        return False

    # Extract base types for family matching
    expr_base = None
    if isinstance(expr_type, (ScalarType, ArrayType)):
        expr_base = expr_type.base
    elif isinstance(expr_type, WildcardArrayType):
        expr_base = expr_type.base

    first_base = None
    if isinstance(first, (ScalarType, ArrayType)):
        first_base = first.base
    elif isinstance(first, WildcardArrayType):
        first_base = first.base

    # Same base family match (f32 param matches f32[3,3] expr)
    if first_base and expr_base and first_base == expr_base:
        return True

    # Struct name match
    if isinstance(expr_type, StructType) and isinstance(first, StructType):
        return first.name == expr_type.name

    return False


def _is_complex_builtin_pipe_compatible(
    expr_type: MaomiType, category: str,
) -> bool:
    """Check if a complex builtin's category is compatible with expr_type for pipe completion."""
    # Categories that operate on arrays/scalars of any float type
    _FLOAT_ARRAY_CATEGORIES = {
        "reduction", "shape", "array_manip", "sorting", "cumulative",
        "two_arg_elementwise", "clip", "stop_grad", "where",
    }
    # Categories that require specific shapes
    _ARRAY_ONLY_CATEGORIES = {"conv_pool", "linalg"}

    if category in _FLOAT_ARRAY_CATEGORIES:
        if isinstance(expr_type, (ScalarType, ArrayType)):
            return expr_type.base in FLOAT_BASES
        if isinstance(expr_type, WildcardArrayType):
            return expr_type.base in FLOAT_BASES
        if isinstance(expr_type, StructType):
            # reductions like sum work on structs, shape ops generally don't
            return category in {"reduction", "stop_grad"}
        return False

    if category in _ARRAY_ONLY_CATEGORIES:
        if isinstance(expr_type, ArrayType):
            return expr_type.base in FLOAT_BASES
        return False

    if category == "argmax":
        # argmax/argmin take arrays
        if isinstance(expr_type, ArrayType):
            return expr_type.base in FLOAT_BASES
        return False

    if category == "bool_reduction":
        # all/any take bool arrays
        if isinstance(expr_type, (ScalarType, ArrayType)):
            return expr_type.base == "bool"
        return False

    return False


def _pipe_completions(
    result: AnalysisResult, expr_type: MaomiType, position: types.Position,
) -> list[types.CompletionItem]:
    """Build completion items for functions compatible with piping from expr_type."""
    items: list[types.CompletionItem] = []
    dot_col = position.character - 1  # 0-indexed position of the '.'
    dot_range = types.Range(
        start=types.Position(line=position.line, character=dot_col),
        end=types.Position(line=position.line, character=dot_col + 1),
    )

    seen: set[str] = set()

    fn_docs: dict[str, str] = {}
    if result.program:
        fn_docs = {f.name: f.doc for f in result.program.functions if f.doc}

    def _make_item(name: str, detail: str, doc_text: str | None) -> types.CompletionItem:
        return types.CompletionItem(
            label=name,
            kind=types.CompletionItemKind.Function,
            detail=detail,
            documentation=types.MarkupContent(
                kind=types.MarkupKind.Markdown, value=doc_text,
            ) if doc_text else None,
            text_edit=types.TextEdit(range=dot_range, new_text=f" |> {name}($0)"),
            insert_text_format=types.InsertTextFormat.Snippet,
            sort_text=f"1_{name}",
        )

    # 1) Functions in fn_table (elementwise builtins + user functions)
    if result.fn_table:
        for name, sig in result.fn_table.items():
            if "." in name or "$" in name:
                continue
            if not _is_pipe_compatible(expr_type, name, sig):
                continue

            seen.add(name)

            if name in _BUILTIN_SET:
                detail = _BUILTIN_CATEGORIES.get(name, "builtin")
            else:
                params = ", ".join(
                    f"{n}: {t}" for n, t in zip(sig.param_names, sig.param_types)
                )
                detail = f"({params}) -> {sig.return_type}"

            doc_text = _BUILTIN_DOCS.get(name) or fn_docs.get(name)
            items.append(_make_item(name, detail, doc_text))

    # 2) Complex builtins not in fn_table — use category-based matching
    for name, b in _CX_REGISTRY.items():
        if name in seen or "." in name:
            continue
        if not _is_complex_builtin_pipe_compatible(expr_type, b.category):
            continue

        detail = _BUILTIN_CATEGORIES.get(name, "builtin")
        doc_text = _BUILTIN_DOCS.get(name)
        items.append(_make_item(name, detail, doc_text))

    # 3) Generic user functions not in fn_table (have symbolic dims / type vars)
    if result.program:
        for fn in result.program.functions:
            if fn.name in seen or "." in fn.name or fn.source_file is not None:
                continue
            if not fn.params:
                continue
            first_ann = fn.params[0].type_annotation
            if _annotation_matches_type(first_ann, expr_type, result.struct_defs):
                seen.add(fn.name)
                ann_strs = [f"{p.name}: {_annotation_str(p.type_annotation)}" for p in fn.params]
                detail = f"({', '.join(ann_strs)}) -> {_annotation_str(fn.return_type)}"
                items.append(_make_item(fn.name, detail, fn.doc))

    return items


def _annotation_str(ann) -> str:
    """Format a TypeAnnotation as a readable string."""
    if ann.dims is None:
        return ann.base
    dims = ", ".join(str(d.value) for d in ann.dims)
    return f"{ann.base}[{dims}]"


def _annotation_matches_type(ann, expr_type: MaomiType, struct_defs: dict) -> bool:
    """Check if a type annotation's base matches the expression type (for pipe completion)."""
    base = ann.base

    # Struct name match
    if isinstance(expr_type, StructType):
        return base == expr_type.name

    # Scalar/array base match
    expr_base = None
    if isinstance(expr_type, (ScalarType, ArrayType)):
        expr_base = expr_type.base
    elif isinstance(expr_type, WildcardArrayType):
        expr_base = expr_type.base

    if expr_base and base == expr_base:
        return True

    # Single uppercase letter = type variable, matches anything
    if len(base) == 1 and base.isupper():
        return True

    return False


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
            label=b, kind=types.CompletionItemKind.Function, detail=_BUILTIN_CATEGORIES.get(b, "builtin"),
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

        # Type aliases
        for ta in result.program.type_aliases:
            items.append(types.CompletionItem(
                label=ta.name,
                kind=types.CompletionItemKind.TypeParameter,
                detail=f"type {ta.name} = {ta.type_annotation.base}" + (
                    f"[{', '.join(str(d.value) for d in ta.type_annotation.dims)}]"
                    if ta.type_annotation.dims else ""
                ),
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
