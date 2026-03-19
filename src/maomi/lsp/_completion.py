from __future__ import annotations

import logging
import os
import re

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ..ast_nodes import Block, LetStmt, ExprStmt, ScanExpr, FoldExpr, WhileExpr, MapExpr, IfExpr
from ..types import MaomiType, ScalarType, ArrayType, StructType, StructArrayType, WildcardArrayType, TypeVar, FLOAT_BASES
from ._core import server, _cache, AnalysisResult, _local_functions, completion_validate, _uri_to_path, _FAKE_ID
from ._ast_utils import _span_contains, _find_node_at
from ._builtin_data import (
    _KEYWORDS, _TYPE_NAMES, _BUILTINS, _BUILTIN_SET,
    _BUILTIN_NAMESPACES, _BUILTIN_DOCS, _BUILTIN_CATEGORIES, _EW_NAMES,
)
from ..builtins import COMPLEX as _CX_REGISTRY

logger = logging.getLogger("maomi-lsp")

_STDLIB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stdlib")


def _complete_import(line_text: str, col: int, filepath: str) -> types.CompletionList | None:
    """Detect import context from line text and return appropriate completions."""
    text_before = line_text[:col]

    # Pattern 1: "from X import { names..." — inside braces
    m = re.match(r'^\s*from\s+("?[^"\s]+"?)\s+import\s*\{([^}]*)$', text_before)
    if m:
        module_name = m.group(1).strip('"')
        already_imported = _parse_already_imported(m.group(2))
        return _complete_import_names(module_name, filepath, already_imported)

    # Also handle: "from X as Y import { names..."
    m = re.match(r'^\s*from\s+("?[^"\s]+"?)\s+as\s+\w+\s+import\s*\{([^}]*)$', text_before)
    if m:
        module_name = m.group(1).strip('"')
        already_imported = _parse_already_imported(m.group(2))
        return _complete_import_names(module_name, filepath, already_imported)

    # Pattern 2: "from ___" or "import ___" — module name position
    m = re.match(r'^\s*(?:from|import)\s+(\w*)$', text_before)
    if m:
        return _complete_import_modules(filepath)

    return None


def _parse_already_imported(braces_text: str) -> set[str]:
    """Parse the text inside import braces to find already-imported names.

    E.g., for ' relu, linear' returns {'relu', 'linear'}.
    """
    return {name for part in braces_text.split(",") if (name := part.strip())}


def _complete_import_modules(filepath: str) -> types.CompletionList:
    """List available modules -- sibling .mao files + stdlib modules."""
    items: list[types.CompletionItem] = []
    seen: set[str] = set()

    # Sibling .mao files in the same directory
    if filepath:
        dir_path = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        if os.path.isdir(dir_path):
            for f in sorted(os.listdir(dir_path)):
                if f.endswith(".mao") and f != basename:
                    name = f[:-4]
                    if name not in seen:
                        seen.add(name)
                        items.append(types.CompletionItem(
                            label=name,
                            kind=types.CompletionItemKind.Module,
                            detail="local module",
                        ))

    # Stdlib modules
    if os.path.isdir(_STDLIB_DIR):
        for f in sorted(os.listdir(_STDLIB_DIR)):
            if f.endswith(".mao"):
                name = f[:-4]
                if name not in seen:
                    seen.add(name)
                    items.append(types.CompletionItem(
                        label=name,
                        kind=types.CompletionItemKind.Module,
                        detail="stdlib",
                    ))

    return types.CompletionList(is_incomplete=False, items=items)


def _complete_import_names(
    module_name: str, filepath: str, already_imported: set[str] | None = None,
) -> types.CompletionList | None:
    """List functions and structs exported by a module, filtering already-imported names."""
    mod_path = _find_module_file(module_name, filepath)
    if mod_path is None:
        return None
    try:
        from ..lexer import Lexer
        from ..parser import Parser
        # B18: Use context manager to avoid file handle leak
        with open(mod_path) as f:
            source = f.read()
        tokens = Lexer(source, filename=mod_path).tokenize()
        parser = Parser(tokens, filename=mod_path)
        program = parser.parse()
    except Exception:
        return None

    if already_imported is None:
        already_imported = set()

    items: list[types.CompletionItem] = []
    for fn in program.functions:
        # G14: Filter already-imported names
        if fn.name in already_imported:
            continue
        doc = fn.doc
        items.append(types.CompletionItem(
            label=fn.name,
            kind=types.CompletionItemKind.Function,
            detail="function",
            documentation=types.MarkupContent(
                kind=types.MarkupKind.Markdown, value=doc,
            ) if doc else None,
        ))
    for sd in program.struct_defs:
        # G14: Filter already-imported names
        if sd.name in already_imported:
            continue
        items.append(types.CompletionItem(
            label=sd.name,
            kind=types.CompletionItemKind.Struct,
            detail="struct",
        ))
    return types.CompletionList(is_incomplete=False, items=items)


def _find_module_file(module_name: str, importing_file: str) -> str | None:
    """Find the .mao file for a module name. Returns path or None."""
    if importing_file:
        dir_path = os.path.dirname(importing_file)
        # Check local sibling
        candidate = os.path.join(dir_path, module_name + ".mao")
        if os.path.isfile(candidate):
            return candidate
    # Check stdlib
    candidate = os.path.join(_STDLIB_DIR, module_name + ".mao")
    if os.path.isfile(candidate):
        return candidate
    return None


def _complete_struct_literal(
    line_text: str, col: int, result: AnalysisResult | None,
    source: str, position: types.Position,
) -> types.CompletionList | None:
    """Suggest remaining fields inside a struct literal like 'Point { x: 1.0, |}'."""
    if not result or not result.struct_defs:
        return None
    # Look backward through source for an unclosed struct literal pattern
    # Find the struct name by scanning for 'Name {' with unmatched brace
    text_before = source.splitlines(keepends=True)
    # Collect all text up to cursor
    lines_before = text_before[:position.line]
    partial = "".join(lines_before) + line_text[:col]
    # Find last unmatched '{' preceded by an identifier
    depth = 0
    i = len(partial) - 1
    while i >= 0:
        if partial[i] == '}':
            depth += 1
        elif partial[i] == '{':
            if depth > 0:
                depth -= 1
            else:
                # Found unmatched '{' -- check if preceded by identifier
                j = i - 1
                # E3: Use .isspace() instead of only checking spaces
                while j >= 0 and partial[j].isspace():
                    j -= 1
                if j >= 0 and (partial[j].isalnum() or partial[j] == '_'):
                    end = j + 1
                    while j >= 0 and (partial[j].isalnum() or partial[j] == '_'):
                        j -= 1
                    struct_name = partial[j + 1:end]

                    # B2: Check if the identifier is preceded by '->' (return type annotation).
                    # If so, the '{' is a function body brace, not a struct literal.
                    pre_ident = partial[:j + 1].rstrip()
                    if pre_ident.endswith("->"):
                        break

                    # Check if it's a known struct
                    sd = result.struct_defs.get(struct_name)
                    if sd is not None:
                        # Find already-written fields in the literal
                        inside = partial[i + 1:]
                        written = set(re.findall(r'(\w+)\s*:', inside))
                        # Offer remaining fields
                        items = []
                        for fname, ftype in sd.fields:
                            if fname not in written:
                                items.append(types.CompletionItem(
                                    label=fname,
                                    kind=types.CompletionItemKind.Field,
                                    detail=str(ftype),
                                    insert_text=f"{fname}: $0",
                                    insert_text_format=types.InsertTextFormat.Snippet,
                                ))
                        if items:
                            return types.CompletionList(is_incomplete=False, items=items)
                break
        i -= 1
    return None


@server.feature(
    types.TEXT_DOCUMENT_COMPLETION,
    types.CompletionOptions(trigger_characters=["."]),
)
def completions(ls: LanguageServer, params: types.CompletionParams):
    logger.debug("completions: %s at %d:%d", params.text_document.uri,
                 params.position.line, params.position.character)
    uri = params.text_document.uri
    doc = ls.workspace.get_text_document(uri)

    # Try fresh parse with fake identifier for accurate completions
    filepath = _uri_to_path(uri)
    result = completion_validate(
        doc.source, filepath,
        params.position.line, params.position.character,
    )
    # Fall back to cache if fresh parse produced nothing
    if result.program is None:
        result = _cache.get(uri)

    lines = doc.source.splitlines()
    # C6: EOF cursor -- use empty string instead of returning None
    if params.position.line >= len(lines):
        line_text = ""
    else:
        line_text = lines[params.position.line]
    col = params.position.character

    # C5: Clamp col to line length to prevent IndexError
    col = min(col, len(line_text))

    # Import context: module names or imported item names
    import_result = _complete_import(line_text, col, filepath)
    if import_result is not None:
        return import_result

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

    # Struct literal context: suggest remaining fields
    struct_lit_result = _complete_struct_literal(
        line_text, col, result, doc.source, params.position,
    )
    if struct_lit_result is not None:
        return struct_lit_result

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
    # G24-partial: Also handle StructArrayType -- yield its struct fields
    fields: tuple[tuple[str, MaomiType], ...] | None = None
    if isinstance(typ, StructType):
        fields = typ.fields
    elif isinstance(typ, StructArrayType):
        fields = typ.struct_type.fields
    if fields is not None:
        for fname, ftype in fields:
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
        if isinstance(expr_type, (StructType, StructArrayType)):
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

    # StructArrayType — extract struct_type for compatibility check
    if isinstance(expr_type, StructArrayType):
        if isinstance(first, StructArrayType):
            return first.struct_type.name == expr_type.struct_type.name
        if isinstance(first, StructType):
            return first.name == expr_type.struct_type.name

    return False


def _is_complex_builtin_pipe_compatible(
    expr_type: MaomiType, category: str,
) -> bool:
    """Check if a complex builtin's category is compatible with expr_type for pipe completion."""
    # Categories that operate on arrays/scalars of any float type
    _FLOAT_ARRAY_CATEGORIES = {
        "reduction", "shape", "array_manip", "sorting", "cumulative",
        "two_arg_elementwise", "clip", "stop_grad", "where", "einsum",
    }
    # Categories that require specific shapes
    _ARRAY_ONLY_CATEGORIES = {"conv_pool", "linalg"}

    if category in _FLOAT_ARRAY_CATEGORIES:
        if isinstance(expr_type, (ScalarType, ArrayType)):
            return expr_type.base in FLOAT_BASES
        if isinstance(expr_type, WildcardArrayType):
            return expr_type.base in FLOAT_BASES
        if isinstance(expr_type, (StructType, StructArrayType)):
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
    # Range covering the dot — used by additional_text_edits to replace '.' with ' |> '
    dot_range = types.Range(
        start=types.Position(line=position.line, character=dot_col),
        end=types.Position(line=position.line, character=dot_col + 1),
    )
    # Zero-width range at cursor position — used by text_edit for filtering/insertion
    cursor_range = types.Range(
        start=types.Position(line=position.line, character=position.character),
        end=types.Position(line=position.line, character=position.character),
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
            # text_edit at cursor position (VS Code uses this range for filtering)
            text_edit=types.TextEdit(range=cursor_range, new_text=f"{name}($0)"),
            # additional_text_edits replaces the dot with ' |> '
            additional_text_edits=[
                types.TextEdit(range=dot_range, new_text=" |> "),
            ],
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

    # 2) Complex builtins not in fn_table -- use category-based matching
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
                items.append(_make_item(fn.name, _fn_detail_from_ast(fn), fn.doc))

    return items


def _fn_detail_from_ast(fn) -> str:
    """Build a detail string like '(x: f32, y: f32) -> f32' from an AST FnDef."""
    ann_strs = [f"{p.name}: {_annotation_str(p.type_annotation)}" for p in fn.params]
    return f"({', '.join(ann_strs)}) -> {_annotation_str(fn.return_type)}"


def _annotation_str(ann) -> str:
    """Format a TypeAnnotation as a readable string."""
    # B12: Handle wildcard f32[..]
    if getattr(ann, 'wildcard', False):
        return f"{ann.base}[..]"
    if ann.dims is None:
        return ann.base
    dims = ", ".join(str(d.value) for d in ann.dims)
    return f"{ann.base}[{dims}]"


def _annotation_matches_type(ann, expr_type: MaomiType, struct_defs: dict) -> bool:
    """Check if a type annotation's base matches the expression type (for pipe completion)."""
    base = ann.base

    # Single uppercase letter = type variable, matches anything (check first!)
    if len(base) == 1 and base.isupper():
        return True

    # Struct name match
    if isinstance(expr_type, StructType):
        return base == expr_type.name

    # StructArrayType — match struct name
    if isinstance(expr_type, StructArrayType) and base == expr_type.struct_type.name:
        return True

    # Scalar/array base match
    expr_base = None
    if isinstance(expr_type, (ScalarType, ArrayType)):
        expr_base = expr_type.base
    elif isinstance(expr_type, WildcardArrayType):
        expr_base = expr_type.base

    if expr_base and base == expr_base:
        return True

    return False


def _complete_module(result: AnalysisResult | None, module_name: str):
    """Complete functions and structs from an imported module (e.g., 'cnn.' -> cnn.relu, cnn.Point)."""
    if not result or not module_name:
        return None
    if not result.fn_table and not result.program:
        return None
    prefix = module_name + "."
    items: list[types.CompletionItem] = []
    fn_docs = {f.name: f.doc for f in result.program.functions if f.doc} if result.program else {}
    for name, sig in result.fn_table.items():
        if not name.startswith(prefix):
            continue
        short_name = name[len(prefix):]
        # B1: Filter monomorphized $-copies from module completions
        if "$" in short_name:
            continue
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

    # B1: Generic/wildcard/comptime functions not in fn_table for this module
    mod_fn_seen = {item.label for item in items}
    if result.program:
        for fn in result.program.functions:
            if not fn.name.startswith(prefix):
                continue
            short_name = fn.name[len(prefix):]
            if "$" in short_name or short_name in mod_fn_seen:
                continue
            detail = _fn_detail_from_ast(fn)
            doc = fn_docs.get(fn.name) or fn.doc
            items.append(types.CompletionItem(
                label=short_name,
                kind=types.CompletionItemKind.Function,
                detail=detail,
                documentation=types.MarkupContent(
                    kind=types.MarkupKind.Markdown, value=doc,
                ) if doc else None,
            ))

    # G12: Also include structs from the module
    if result.struct_defs:
        struct_docs = {}
        if result.program:
            struct_docs = {sd.name: sd.doc for sd in result.program.struct_defs if sd.doc}
        for sname, stype in result.struct_defs.items():
            if not sname.startswith(prefix):
                continue
            short_name = sname[len(prefix):]
            doc = struct_docs.get(sname)
            items.append(types.CompletionItem(
                label=short_name,
                kind=types.CompletionItemKind.Struct,
                detail=str(stype),
                documentation=types.MarkupContent(
                    kind=types.MarkupKind.Markdown, value=doc,
                ) if doc else None,
            ))

    if not items:
        return None
    return types.CompletionList(is_incomplete=False, items=items)


def _complete_general(result: AnalysisResult | None, position: types.Position):
    items: list[types.CompletionItem] = []

    # G15: Add sort_text prefix by category
    for kw in _KEYWORDS:
        items.append(types.CompletionItem(
            label=kw, kind=types.CompletionItemKind.Keyword,
            sort_text=f"4_{kw}",
        ))

    for t in _TYPE_NAMES:
        items.append(types.CompletionItem(
            label=t, kind=types.CompletionItemKind.TypeParameter,
            sort_text=f"5_{t}",
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
            sort_text=f"3_{b}",
        ))

    for ns in _BUILTIN_NAMESPACES:
        items.append(types.CompletionItem(
            label=ns, kind=types.CompletionItemKind.Module, detail="builtin namespace",
            sort_text=f"3_{ns}",
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
                        sort_text=f"1_{mod}",
                    ))

        # Build fn doc lookup
        fn_docs = {f.name: f.doc for f in result.program.functions if f.doc}
        # User-defined functions
        user_fn_seen: set[str] = set()
        for name, sig in result.fn_table.items():
            # B1: Filter monomorphized $-copies and module-qualified names
            if name in _BUILTIN_SET or "." in name or "$" in name:
                continue
            user_fn_seen.add(name)
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
                sort_text=f"1_{name}",
            ))

        # B1: Generic/wildcard/comptime functions not in fn_table
        for fn in result.program.functions:
            if fn.name in user_fn_seen or fn.name in _BUILTIN_SET:
                continue
            if "." in fn.name or "$" in fn.name or fn.source_file is not None:
                continue
            items.append(types.CompletionItem(
                label=fn.name,
                kind=types.CompletionItemKind.Function,
                detail=_fn_detail_from_ast(fn),
                documentation=types.MarkupContent(
                    kind=types.MarkupKind.Markdown, value=fn.doc,
                ) if fn.doc else None,
                sort_text=f"1_{fn.name}",
            ))

        # Struct names
        struct_docs = {sd.name: sd.doc for sd in result.program.struct_defs if sd.doc}
        for name in result.struct_defs:
            # B1: Filter module-prefixed structs and $-copies
            if "." in name or "$" in name:
                continue
            doc = struct_docs.get(name)
            items.append(types.CompletionItem(
                label=name, kind=types.CompletionItemKind.Struct,
                documentation=types.MarkupContent(
                    kind=types.MarkupKind.Markdown, value=doc,
                ) if doc else None,
                sort_text=f"2_{name}",
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
                sort_text=f"5_{ta.name}",
            ))

        # Variables in scope
        for var_name, var_type in _vars_in_scope(result, position):
            if var_name == _FAKE_ID or var_name.startswith("__maomi"):
                continue
            items.append(types.CompletionItem(
                label=var_name,
                kind=types.CompletionItemKind.Variable,
                detail=str(var_type) if var_type else None,
                sort_text=f"0_{var_name}",
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
            # B17: Use line_end (not line_start) for col_end comparison
            if stmt.span.line_start < line or (
                stmt.span.line_end == line and stmt.span.col_end < col
            ):
                typ = type_map.get(id(stmt.value))
                # Replace earlier binding with same name (shadowing)
                for i, (vname, _) in enumerate(variables):
                    if vname == stmt.name:
                        variables[i] = (stmt.name, typ)
                        break
                else:
                    variables.append((stmt.name, typ))

            # B3: Descend into compound expressions inside let values
            if isinstance(stmt.value, (ScanExpr, FoldExpr, MapExpr, WhileExpr, IfExpr)):
                _collect_from_expr(stmt.value, line, col, type_map, variables)

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
        variables.append((expr.carry_var, None))
        for ev in expr.elem_vars:
            variables.append((ev, None))
        _collect_scope_vars(expr.body, line, col, type_map, variables)

    elif isinstance(expr, FoldExpr):
        variables.append((expr.carry_var, None))
        for ev in expr.elem_vars:
            variables.append((ev, None))
        _collect_scope_vars(expr.body, line, col, type_map, variables)

    elif isinstance(expr, WhileExpr):
        variables.append((expr.state_var, None))
        _collect_scope_vars(expr.cond, line, col, type_map, variables)
        _collect_scope_vars(expr.body, line, col, type_map, variables)

    elif isinstance(expr, MapExpr):
        variables.append((expr.elem_var, None))
        _collect_scope_vars(expr.body, line, col, type_map, variables)

    elif isinstance(expr, IfExpr):
        # Only collect from the branch containing the cursor (prevent scope leakage)
        if _span_contains(expr.then_block.span, line, col):
            _collect_scope_vars(expr.then_block, line, col, type_map, variables)
        elif _span_contains(expr.else_block.span, line, col):
            _collect_scope_vars(expr.else_block, line, col, type_map, variables)
