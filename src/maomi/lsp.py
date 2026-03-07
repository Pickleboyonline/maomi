from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from .lexer import Lexer
from .parser import Parser
from .resolver import resolve
from .type_checker import TypeChecker
from .errors import MaomiError, LexerError, ParseError
from .ast_nodes import (
    Program, FnDef, Block, LetStmt, ExprStmt, Param,
    BinOp, UnaryOp, IfExpr, CallExpr, ScanExpr, WhileExpr, MapExpr,
    GradExpr, CastExpr, FoldExpr,
    Identifier, IntLiteral, FloatLiteral, BoolLiteral, StringLiteral,
    StructLiteral, FieldAccess, WithExpr, IndexExpr, StructDef,
    TypeAnnotation,
    _ScanGrad, _IndexGrad, _GatherGrad, _Conv2dGrad,
    _MaxPoolGrad, _AvgPoolGrad, _BroadcastExpr,
)
from .types import MaomiType, StructType, ArrayType, WildcardArrayType


@dataclass
class AnalysisResult:
    program: Program | None
    type_map: dict[int, MaomiType]
    fn_table: dict
    struct_defs: dict


_EMPTY_RESULT = AnalysisResult(None, {}, {}, {})

server = LanguageServer("maomi-lsp", "0.1.0")
_cache: dict[str, AnalysisResult] = {}


def _local_functions(program: Program) -> list[FnDef]:
    """Return only functions defined in the current file (not imported)."""
    return [fn for fn in program.functions if "." not in fn.name]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(source: str, filename: str) -> tuple[list[types.Diagnostic], AnalysisResult]:
    diagnostics: list[types.Diagnostic] = []

    try:
        tokens = Lexer(source, filename=filename).tokenize()
    except LexerError as e:
        diagnostics.append(_error_to_diagnostic(e))
        return diagnostics, _EMPTY_RESULT

    try:
        program = Parser(tokens, filename=filename).parse()
    except ParseError as e:
        diagnostics.append(_error_to_diagnostic(e))
        return diagnostics, _EMPTY_RESULT

    try:
        program = resolve(program, filename)
    except MaomiError as e:
        diagnostics.append(_error_to_diagnostic(e))
        return diagnostics, _EMPTY_RESULT

    checker = TypeChecker(filename=filename)
    type_errors = checker.check(program)
    for e in type_errors:
        diagnostics.append(_error_to_diagnostic(e))

    return diagnostics, AnalysisResult(
        program, checker.type_map, dict(checker.fn_table), dict(checker.struct_defs),
    )


def _error_to_diagnostic(e: MaomiError) -> types.Diagnostic:
    line = max(0, e.line - 1)
    col = max(0, e.col - 1)
    return types.Diagnostic(
        range=types.Range(
            start=types.Position(line=line, character=col),
            end=types.Position(line=line, character=col + 1),
        ),
        message=e.message,
        severity=types.DiagnosticSeverity.Error,
        source="maomi",
    )


# ---------------------------------------------------------------------------
# Hover — AST walker
# ---------------------------------------------------------------------------

def _span_contains(span, line: int, col: int) -> bool:
    if line < span.line_start or line > span.line_end:
        return False
    if line == span.line_start and col < span.col_start:
        return False
    if line == span.line_end and col > span.col_end:
        return False
    return True


def _children_of(node):
    match node:
        case FnDef(params=params, body=body):
            yield from params
            yield body
        case Block(stmts=stmts, expr=expr):
            yield from stmts
            if expr is not None:
                yield expr
        case LetStmt(value=v):
            yield v
        case ExprStmt(expr=e):
            yield e
        case BinOp(left=l, right=r):
            yield l
            yield r
        case UnaryOp(operand=o):
            yield o
        case IfExpr(condition=c, then_block=t, else_block=e):
            yield c
            yield t
            yield e
        case CallExpr(args=args):
            yield from args
        case ScanExpr(init=init, sequences=seqs, body=body):
            yield init
            yield from seqs
            yield body
        case MapExpr(sequence=seq, body=body):
            yield seq
            yield body
        case GradExpr(expr=e):
            yield e
        case StructLiteral(fields=fields):
            for _, expr in fields:
                yield expr
        case FieldAccess(object=obj):
            yield obj
        case WithExpr(base=b, updates=updates):
            yield b
            for _, expr in updates:
                yield expr
        case IndexExpr(base=b, indices=indices):
            yield b
            for ic in indices:
                if ic.value is not None:
                    yield ic.value
                if ic.start is not None:
                    yield ic.start
                if ic.end is not None:
                    yield ic.end
        case _:
            pass


def _find_node_at(node, line: int, col: int):
    if not hasattr(node, "span") or not _span_contains(node.span, line, col):
        return None
    for child in _children_of(node):
        found = _find_node_at(child, line, col)
        if found is not None:
            return found
    return node


# ---------------------------------------------------------------------------
# LSP handlers
# ---------------------------------------------------------------------------

def _uri_to_path(uri: str) -> str:
    if uri.startswith("file://"):
        return unquote(urlparse(uri).path)
    return uri


def _do_validate(ls: LanguageServer, uri: str):
    doc = ls.workspace.get_text_document(uri)
    filepath = _uri_to_path(uri)
    diags, result = validate(doc.source, filepath)
    # Only update cache when we have a valid program — keeps stale-but-useful
    # results for completion/hover while the user is mid-typing
    if result.program is not None:
        _cache[uri] = result
    ls.text_document_publish_diagnostics(types.PublishDiagnosticsParams(
        uri=uri, diagnostics=diags,
    ))


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams):
    _do_validate(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams):
    _do_validate(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams):
    _do_validate(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(ls: LanguageServer, params: types.HoverParams):
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
            if node.doc:
                text += f"\n\n{node.doc}"
            return text
        return None

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
            # Find the FnDef for doc
            if result.program:
                for f in result.program.functions:
                    if f.name == callee and f.doc:
                        text += f"\n\n{f.doc}"
                        break
            return text

    # General expression: look up type_map
    typ = result.type_map.get(id(node))
    if typ is not None:
        # For identifiers, show "name: type"
        if isinstance(node, Identifier):
            return f"```maomi\n{node.name}: {typ}\n```"
        return f"```maomi\n{typ}\n```"
    return None


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

_KEYWORDS = [
    "fn", "let", "if", "else", "scan", "map", "grad", "cast", "fold",
    "struct", "with", "import", "from", "in", "true", "false",
    "while", "do", "limit",
]

_TYPE_NAMES = ["f32", "f64", "i32", "i64", "bool"]

_BUILTINS = [
    "mean", "sum", "max", "min", "argmax", "argmin",
    "exp", "log", "tanh", "sqrt", "abs",
    "reshape", "concat", "iota", "transpose", "callback",
    "random.key", "random.split", "random.uniform", "random.normal",
    "conv2d", "max_pool", "avg_pool",
    "stop_gradient", "where",
]

_BUILTIN_SET = set(_BUILTINS)

_BUILTIN_NAMESPACES: dict[str, list[str]] = {
    "random": ["key", "split", "uniform", "normal"],
}


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


# ---------------------------------------------------------------------------
# Go to Definition
# ---------------------------------------------------------------------------

def _span_to_range(span) -> types.Range:
    """Convert 1-indexed Maomi Span to 0-indexed LSP Range."""
    return types.Range(
        start=types.Position(line=span.line_start - 1, character=span.col_start - 1),
        end=types.Position(line=span.line_end - 1, character=span.col_end - 1),
    )


def _goto_find_binding_in_block(name, block, line, col):
    """Search a Block for a LetStmt that binds `name` before (line, col)."""
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt) and stmt.name == name:
            if stmt.span.line_end < line or (
                stmt.span.line_end == line and stmt.span.col_end < col
            ):
                return stmt.span
        # Check inside expressions (scan/map bodies)
        if isinstance(stmt, ExprStmt):
            result = _goto_find_binding_in_expr(name, stmt.expr, line, col)
            if result is not None:
                return result
    if block.expr is not None:
        result = _goto_find_binding_in_expr(name, block.expr, line, col)
        if result is not None:
            return result
    return None


def _goto_find_binding_in_expr(name, expr, line, col):
    """Check if the cursor is inside a scan/map body and `name` matches a loop var."""
    if not hasattr(expr, "span") or not _span_contains(expr.span, line, col):
        return None

    if isinstance(expr, ScanExpr):
        if expr.carry_var == name:
            return expr.span
        if name in expr.elem_vars:
            return expr.span
        result = _goto_find_binding_in_block(name, expr.body, line, col)
        if result is not None:
            return result

    elif isinstance(expr, MapExpr):
        if expr.elem_var == name:
            return expr.span
        result = _goto_find_binding_in_block(name, expr.body, line, col)
        if result is not None:
            return result

    elif isinstance(expr, IfExpr):
        result = _goto_find_binding_in_block(name, expr.then_block, line, col)
        if result is not None:
            return result
        result = _goto_find_binding_in_block(name, expr.else_block, line, col)
        if result is not None:
            return result

    return None


def _goto_find_binding(name, fn, line, col):
    """Find where a variable name is bound in the enclosing function."""
    for param in fn.params:
        if param.name == name:
            return param.span
    return _goto_find_binding_in_block(name, fn.body, line, col)


def _goto_find_definition(node, fn, result):
    """Find the definition span for the given node."""
    program = result.program
    if program is None:
        return None

    if isinstance(node, CallExpr):
        for fndef in program.functions:
            if fndef.name == node.callee:
                return fndef.span
        return None

    if isinstance(node, Identifier):
        return _goto_find_binding(node.name, fn, node.span.line_start, node.span.col_start)

    if isinstance(node, StructLiteral):
        for sdef in program.struct_defs:
            if sdef.name == node.name:
                return sdef.span
        return None

    if isinstance(node, FieldAccess):
        typ = result.type_map.get(id(node.object))
        if isinstance(typ, StructType):
            for sdef in program.struct_defs:
                if sdef.name == typ.name:
                    return sdef.span
        return None

    return None


@server.feature(types.TEXT_DOCUMENT_DEFINITION)
def goto_definition(ls, params: types.DefinitionParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None
    line = params.position.line + 1
    col = params.position.character + 1
    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            defn_span = _goto_find_definition(node, fn, result)
            if defn_span is not None:
                return types.Location(uri=uri, range=_span_to_range(defn_span))
    return None


# ---------------------------------------------------------------------------
# Find References
# ---------------------------------------------------------------------------


def _refs_classify_node(node, line=None, col=None):
    """Determine the symbol name and kind from the node under cursor.

    Returns (name, kind) where kind is "function", "variable", or "struct",
    or (None, None) if the node is not a classifiable symbol.
    """
    if isinstance(node, CallExpr):
        return node.callee, "function"
    if isinstance(node, FnDef):
        return node.name, "function"
    if isinstance(node, StructDef):
        return node.name, "struct"
    if isinstance(node, StructLiteral):
        return node.name, "struct"
    if isinstance(node, Identifier):
        return node.name, "variable"
    if isinstance(node, Param):
        if line is not None and col is not None:
            ta = node.type_annotation
            if ta and _span_contains(ta.span, line, col):
                return ta.base, "struct"
        return node.name, "variable"
    if isinstance(node, LetStmt):
        return node.name, "variable"
    return None, None


def _refs_walk_node(node, name, kind, spans):
    """Recursively walk AST collecting spans of references to the named symbol."""
    if kind == "function":
        if isinstance(node, CallExpr) and node.callee == name:
            spans.append(node.span)
    elif kind == "variable":
        if isinstance(node, Identifier) and node.name == name:
            spans.append(node.span)
        if isinstance(node, GradExpr) and node.wrt == name:
            spans.append(node.span)
    elif kind == "struct":
        if isinstance(node, StructLiteral) and node.name == name:
            spans.append(node.span)

    if kind == "struct":
        if isinstance(node, Param) and node.type_annotation.base == name:
            spans.append(node.type_annotation.span)
        if isinstance(node, FnDef) and node.return_type and node.return_type.base == name:
            spans.append(node.return_type.span)

    for child in _children_of(node):
        _refs_walk_node(child, name, kind, spans)


def _refs_collect_all(result, name, kind, include_declaration, fn_scope=None):
    """Collect all reference spans for the given symbol."""
    spans = []

    if kind == "function":
        if include_declaration:
            for fn in _local_functions(result.program):
                if fn.name == name:
                    spans.append(fn.span)
                    break
        for fn in _local_functions(result.program):
            _refs_walk_node(fn, name, kind, spans)

    elif kind == "variable":
        if fn_scope is not None:
            if include_declaration:
                for p in fn_scope.params:
                    if p.name == name:
                        spans.append(p.span)
                        break
                else:
                    _refs_find_let_decl(fn_scope.body, name, spans)
            _refs_walk_node(fn_scope.body, name, kind, spans)

    elif kind == "struct":
        if include_declaration:
            for sd in result.program.struct_defs:
                if sd.name == name:
                    spans.append(sd.span)
                    break
        for fn in _local_functions(result.program):
            _refs_walk_node(fn, name, kind, spans)

    return spans


def _refs_find_let_decl(block, name, spans):
    """Find a LetStmt declaration for the given variable name within a Block."""
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt) and stmt.name == name:
            spans.append(stmt.span)
            return
        if isinstance(stmt, ExprStmt):
            for child in _children_of(stmt.expr):
                if isinstance(child, Block):
                    _refs_find_let_decl(child, name, spans)


@server.feature(types.TEXT_DOCUMENT_REFERENCES)
def find_references(ls, params: types.ReferenceParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None

    line = params.position.line + 1
    col = params.position.character + 1

    for sd in result.program.struct_defs:
        if _span_contains(sd.span, line, col):
            name, kind = sd.name, "struct"
            spans = _refs_collect_all(result, name, kind,
                                      params.context.include_declaration)
            return [types.Location(uri=uri, range=_span_to_range(s))
                    for s in spans]

    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            name, kind = _refs_classify_node(node, line, col)
            if name:
                spans = _refs_collect_all(result, name, kind,
                                          params.context.include_declaration,
                                          fn_scope=fn)
                return [types.Location(uri=uri, range=_span_to_range(s))
                        for s in spans]
    return None


# ---------------------------------------------------------------------------
# Document Symbol
# ---------------------------------------------------------------------------

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


@server.feature(types.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbols(ls: LanguageServer, params: types.DocumentSymbolParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    return _build_document_symbols(result)


# ---------------------------------------------------------------------------
# Rename Symbol
# ---------------------------------------------------------------------------


def _rename_name_range(name: str, span, source_lines: list[str]) -> types.Range | None:
    """Find the exact Range of *name* within the source region of *span*."""
    line_idx = span.line_start - 1
    if line_idx < 0 or line_idx >= len(source_lines):
        return None
    line_text = source_lines[line_idx]
    col_start = span.col_start - 1
    idx = line_text.find(name, col_start)
    if idx >= 0:
        return types.Range(
            start=types.Position(line=line_idx, character=idx),
            end=types.Position(line=line_idx, character=idx + len(name)),
        )
    return None


def _rename_classify(node):
    """Determine what kind of symbol is under cursor."""
    if isinstance(node, FnDef):
        return node.name, "function"
    if isinstance(node, CallExpr):
        return node.callee, "function"
    if isinstance(node, Identifier):
        return node.name, "variable"
    if isinstance(node, Param):
        return node.name, "variable"
    if isinstance(node, LetStmt):
        return node.name, "variable"
    if isinstance(node, StructLiteral):
        return node.name, "struct"
    if isinstance(node, StructDef):
        return node.name, "struct"
    return None, None


def _rename_walk(node, callback):
    """Recursively walk *node* and all its children, calling *callback* on each."""
    callback(node)
    for child in _children_of(node):
        _rename_walk(child, callback)


class _EditCollector:
    """Accumulates unique TextEdit ranges, deduplicating by position."""

    def __init__(self, new_name: str):
        self.new_name = new_name
        self.edits: list[types.TextEdit] = []
        self._seen: set[tuple[int, int, int, int]] = set()

    def add(self, rng: types.Range | None):
        if rng is None:
            return
        key = (rng.start.line, rng.start.character, rng.end.line, rng.end.character)
        if key not in self._seen:
            self._seen.add(key)
            self.edits.append(types.TextEdit(range=rng, new_text=self.new_name))


def _rename_collect_function_edits(
    program: Program, name: str, new_name: str, source_lines: list[str],
) -> list[types.TextEdit]:
    ec = _EditCollector(new_name)
    for fn in program.functions:
        if fn.name == name:
            ec.add(_rename_name_range(name, fn.span, source_lines))

        def _visit(node, _name=name):
            if isinstance(node, CallExpr) and node.callee == _name:
                ec.add(_rename_name_range(_name, node.span, source_lines))

        _rename_walk(fn, _visit)
    return ec.edits


def _rename_collect_variable_edits(
    fn_scope: FnDef, name: str, new_name: str, source_lines: list[str],
) -> list[types.TextEdit]:
    ec = _EditCollector(new_name)
    for param in fn_scope.params:
        if param.name == name:
            ec.add(_rename_name_range(name, param.span, source_lines))

    def _visit(node, _name=name):
        if isinstance(node, Identifier) and node.name == _name:
            ec.add(_rename_name_range(_name, node.span, source_lines))
        elif isinstance(node, LetStmt) and node.name == _name:
            ec.add(_rename_name_range(_name, node.span, source_lines))

    _rename_walk(fn_scope.body, _visit)
    return ec.edits


def _rename_collect_struct_edits(
    program: Program, name: str, new_name: str, source_lines: list[str],
) -> list[types.TextEdit]:
    ec = _EditCollector(new_name)
    for sd in program.struct_defs:
        if sd.name == name:
            ec.add(_rename_name_range(name, sd.span, source_lines))

    for fn in program.functions:
        for param in fn.params:
            if param.type_annotation and param.type_annotation.base == name:
                ec.add(_rename_name_range(name, param.type_annotation.span, source_lines))
        if fn.return_type and fn.return_type.base == name:
            ec.add(_rename_name_range(name, fn.return_type.span, source_lines))

        def _visit(node, _name=name):
            if isinstance(node, StructLiteral) and node.name == _name:
                ec.add(_rename_name_range(_name, node.span, source_lines))
            elif isinstance(node, LetStmt) and node.type_annotation and node.type_annotation.base == _name:
                ec.add(_rename_name_range(_name, node.type_annotation.span, source_lines))

        _rename_walk(fn.body, _visit)
    return ec.edits


def prepare_rename_at(
    source: str, result: AnalysisResult, line_0: int, col_0: int,
) -> types.Range | None:
    """Core logic for prepare_rename. Returns the Range of the symbol, or None."""
    if not result or not result.program:
        return None

    line = line_0 + 1
    col = col_0 + 1
    source_lines = source.splitlines()

    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            name, kind = _rename_classify(node)
            if name is None:
                return None
            if kind == "function" and name in _BUILTIN_SET:
                return None
            return _rename_name_range(name, node.span, source_lines)

    for sd in result.program.struct_defs:
        if _span_contains(sd.span, line, col):
            return _rename_name_range(sd.name, sd.span, source_lines)

    return None


def rename_at(
    source: str, result: AnalysisResult, line_0: int, col_0: int, new_name: str,
) -> list[types.TextEdit] | None:
    """Core logic for rename. Returns a list of TextEdits, or None."""
    if not result or not result.program:
        return None

    line = line_0 + 1
    col = col_0 + 1
    source_lines = source.splitlines()

    node = None
    fn_scope = None

    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            fn_scope = fn
            break

    if node is None:
        for sd in result.program.struct_defs:
            if _span_contains(sd.span, line, col):
                node = sd
                break

    if node is None:
        return None

    name, kind = _rename_classify(node)
    if name is None:
        return None
    if kind == "function" and name in _BUILTIN_SET:
        return None

    if kind == "function":
        edits = _rename_collect_function_edits(result.program, name, new_name, source_lines)
    elif kind == "variable":
        if fn_scope is None:
            return None
        edits = _rename_collect_variable_edits(fn_scope, name, new_name, source_lines)
    elif kind == "struct":
        edits = _rename_collect_struct_edits(result.program, name, new_name, source_lines)
    else:
        return None

    return edits if edits else None


@server.feature(types.TEXT_DOCUMENT_PREPARE_RENAME)
def prepare_rename(ls: LanguageServer, params: types.PrepareRenameParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None
    doc = ls.workspace.get_text_document(uri)
    return prepare_rename_at(
        doc.source, result, params.position.line, params.position.character,
    )


@server.feature(types.TEXT_DOCUMENT_RENAME)
def rename(ls: LanguageServer, params: types.RenameParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None
    doc = ls.workspace.get_text_document(uri)
    edits = rename_at(
        doc.source, result,
        params.position.line, params.position.character,
        params.new_name,
    )
    if edits is None:
        return None
    return types.WorkspaceEdit(changes={uri: edits})


# ---------------------------------------------------------------------------
# Signature Help
# ---------------------------------------------------------------------------

_BUILTIN_SIGNATURES: dict[str, tuple[list[str], list[str], str]] = {
    "exp": (["x"], ["f32"], "f32"),
    "log": (["x"], ["f32"], "f32"),
    "tanh": (["x"], ["f32"], "f32"),
    "sqrt": (["x"], ["f32"], "f32"),
    "abs": (["x"], ["f32"], "f32"),
    "mean": (["x"], ["f32[...]"], "f32"),
    "sum": (["x"], ["f32[...]"], "f32"),
    "max": (["x"], ["f32[...]"], "f32"),
    "min": (["x"], ["f32[...]"], "f32"),
    "argmax": (["x"], ["f32[...]"], "i32"),
    "argmin": (["x"], ["f32[...]"], "i32"),
    "reshape": (["x", "dims..."], ["f32[...]", "int..."], "f32[...]"),
    "concat": (["arrays...", "axis"], ["f32[...]...", "int"], "f32[...]"),
    "iota": (["n"], ["int"], "i32[n]"),
    "random.key": (["seed"], ["i32"], "Key"),
    "random.split": (["key", "n"], ["Key", "int"], "Key[n]"),
    "random.uniform": (["key", "low", "high", "dims..."], ["Key", "f32", "f32", "int..."], "f32[...]"),
    "random.normal": (["key", "mean", "std", "dims..."], ["Key", "f32", "f32", "int..."], "f32[...]"),
    "conv2d": (["input", "kernel", "strides", "padding"], ["f32[N,C,H,W]", "f32[O,C,kH,kW]", "(sH,sW)", "str"], "f32[...]"),
    "max_pool": (["input", "window", "strides", "padding"], ["f32[N,C,H,W]", "(wH,wW)", "(sH,sW)", "str"], "f32[...]"),
    "avg_pool": (["input", "window", "strides", "padding"], ["f32[N,C,H,W]", "(wH,wW)", "(sH,sW)", "str"], "f32[...]"),
}

_BUILTIN_DOCS: dict[str, str] = {
    "exp": "Compute element-wise exponential (e^x).",
    "log": "Compute element-wise natural logarithm (ln x).",
    "tanh": "Compute element-wise hyperbolic tangent.",
    "sqrt": "Compute element-wise square root.",
    "abs": "Compute element-wise absolute value.",
    "mean": "Compute the mean of all elements in an array.",
    "sum": "Compute the sum of all elements in an array. `sum(x)` over all elements, `sum(x, axis)` along an axis.",
    "max": "Reduce-max. `max(x)` over all elements, `max(x, axis)` along an axis.",
    "min": "Reduce-min. `min(x)` over all elements, `min(x, axis)` along an axis.",
    "argmax": "Index of maximum element. `argmax(x)` over all elements, `argmax(x, axis)` along an axis. Returns i32.",
    "argmin": "Index of minimum element. `argmin(x)` over all elements, `argmin(x, axis)` along an axis. Returns i32.",
    "stop_gradient": "Identity function that blocks gradient flow backward.",
    "where": "Element-wise conditional: `where(cond, x, y)`. Fully differentiable.",
    "reshape": "Reshape an array to the given dimensions.\n\nTotal element count must be preserved.",
    "concat": "Concatenate arrays along the given axis.",
    "iota": "Generate an integer sequence `[0, 1, ..., n-1]` as `i32[n]`.",
    "transpose": "Transpose a 2D matrix (swap rows and columns).",
    "callback": "Host callback (no-op in compiled code). Useful for debugging.",
    "random.key": "Create a PRNG key from an integer seed.",
    "random.split": "Split a PRNG key into `n` independent subkeys.",
    "random.uniform": "Sample uniform random values in `[low, high)`.",
    "random.normal": "Sample normal random values with given mean and std (Box-Muller).",
    "conv2d": "2D convolution.\n\nInput: `[N, C, H, W]`, Kernel: `[O, C, kH, kW]`.\nPadding: `\"valid\"` or `\"same\"`.",
    "max_pool": "2D max pooling.\n\nInput: `[N, C, H, W]`. Reduces spatial dims by window size.",
    "avg_pool": "2D average pooling.\n\nInput: `[N, C, H, W]`. Reduces spatial dims by window size.",
}


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


# ---------------------------------------------------------------------------
# Inlay Hints
# ---------------------------------------------------------------------------

def _inlay_collect_hints(
    block: Block, type_map: dict, start_line: int, end_line: int,
    hints: list, source_lines: list[str],
):
    """Collect inlay hints from a block (recursive into nested blocks)."""
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt) and stmt.type_annotation is None:
            typ = type_map.get(id(stmt.value))
            if typ is not None and start_line <= stmt.span.line_start <= end_line:
                line_idx = stmt.span.line_start - 1
                line_text = source_lines[line_idx]
                search_start = stmt.span.col_start - 1
                name_start = line_text.find("let ", search_start)
                if name_start >= 0:
                    name_end = name_start + 4 + len(stmt.name)
                    hints.append(types.InlayHint(
                        position=types.Position(line=line_idx, character=name_end),
                        label=f": {typ}",
                        kind=types.InlayHintKind.Type,
                        padding_left=False,
                        padding_right=True,
                    ))

        if isinstance(stmt, ExprStmt):
            _inlay_collect_from_expr(
                stmt.expr, type_map, start_line, end_line, hints, source_lines,
            )

    if block.expr is not None:
        _inlay_collect_from_expr(
            block.expr, type_map, start_line, end_line, hints, source_lines,
        )


def _inlay_collect_from_expr(expr, type_map, start_line, end_line, hints, source_lines):
    """Recurse into expressions that contain blocks."""
    if isinstance(expr, ScanExpr):
        _inlay_collect_hints(
            expr.body, type_map, start_line, end_line, hints, source_lines,
        )
    elif isinstance(expr, MapExpr):
        _inlay_collect_hints(
            expr.body, type_map, start_line, end_line, hints, source_lines,
        )
    elif isinstance(expr, IfExpr):
        _inlay_collect_hints(
            expr.then_block, type_map, start_line, end_line, hints, source_lines,
        )
        _inlay_collect_hints(
            expr.else_block, type_map, start_line, end_line, hints, source_lines,
        )


def _build_inlay_hints(
    result: AnalysisResult, start_line: int, end_line: int, source: str,
) -> list[types.InlayHint]:
    """Build inlay hints for the given 1-indexed line range."""
    if not result.program:
        return []
    source_lines = source.splitlines()
    hints: list[types.InlayHint] = []
    for fn in _local_functions(result.program):
        _inlay_collect_hints(
            fn.body, result.type_map, start_line, end_line, hints, source_lines,
        )
    return hints


@server.feature(types.TEXT_DOCUMENT_INLAY_HINT)
def inlay_hints(ls: LanguageServer, params: types.InlayHintParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None

    start_line = params.range.start.line + 1
    end_line = params.range.end.line + 1
    doc = ls.workspace.get_text_document(uri)

    return _build_inlay_hints(result, start_line, end_line, doc.source)


# ---------------------------------------------------------------------------
# Semantic Tokens
# ---------------------------------------------------------------------------

_SEM_TOKEN_TYPES = [
    "function", "parameter", "variable", "struct",
    "property", "type", "number", "keyword", "string",
]

_SEM_TOKEN_MODIFIERS = ["declaration", "definition"]

_SEM_LEGEND = types.SemanticTokensLegend(
    token_types=_SEM_TOKEN_TYPES,
    token_modifiers=_SEM_TOKEN_MODIFIERS,
)

_ST_FUNCTION = 0
_ST_PARAMETER = 1
_ST_VARIABLE = 2
_ST_STRUCT = 3
_ST_PROPERTY = 4
_ST_TYPE = 5
_ST_NUMBER = 6
_ST_KEYWORD = 7
_ST_STRING = 8

_MOD_DECLARATION = 1
_MOD_DEFINITION = 2


def _sem_collect_tokens(node, tokens: list, param_names: set[str]):
    """Recursively collect semantic tokens from AST nodes."""
    if isinstance(node, FnDef):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1 + 3,
            len(node.name),
            _ST_FUNCTION,
            _MOD_DECLARATION | _MOD_DEFINITION,
        ))
        fn_param_names = {p.name for p in node.params}
        for p in node.params:
            _sem_collect_tokens(p, tokens, fn_param_names)
        _sem_collect_tokens(node.body, tokens, fn_param_names)
        return

    if isinstance(node, Param):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            len(node.name),
            _ST_PARAMETER,
            _MOD_DECLARATION,
        ))
        _sem_add_type_annotation(node.type_annotation, tokens)
        return

    if isinstance(node, StructDef):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1 + 7,
            len(node.name),
            _ST_STRUCT,
            _MOD_DECLARATION | _MOD_DEFINITION,
        ))
        for _, type_ann in node.fields:
            _sem_add_type_annotation(type_ann, tokens)
        return

    if isinstance(node, Identifier):
        token_type = _ST_PARAMETER if node.name in param_names else _ST_VARIABLE
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            len(node.name),
            token_type,
            0,
        ))
        return

    if isinstance(node, CallExpr):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            len(node.callee),
            _ST_FUNCTION,
            0,
        ))
        for arg in node.args:
            _sem_collect_tokens(arg, tokens, param_names)
        return

    if isinstance(node, StructLiteral):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            len(node.name),
            _ST_STRUCT,
            0,
        ))
        for _, expr in node.fields:
            _sem_collect_tokens(expr, tokens, param_names)
        return

    if isinstance(node, FieldAccess):
        _sem_collect_tokens(node.object, tokens, param_names)
        tokens.append((
            node.span.line_end - 1,
            node.span.col_end - len(node.field) - 1,
            len(node.field),
            _ST_PROPERTY,
            0,
        ))
        return

    if isinstance(node, (IntLiteral, FloatLiteral)):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            node.span.col_end - node.span.col_start,
            _ST_NUMBER,
            0,
        ))
        return

    if isinstance(node, StringLiteral):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            node.span.col_end - node.span.col_start,
            _ST_STRING,
            0,
        ))
        return

    if isinstance(node, LetStmt):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            3,
            _ST_KEYWORD,
            0,
        ))
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1 + 4,
            len(node.name),
            _ST_VARIABLE,
            _MOD_DECLARATION,
        ))
        _sem_collect_tokens(node.value, tokens, param_names)
        return

    for child in _children_of(node):
        _sem_collect_tokens(child, tokens, param_names)


def _sem_add_type_annotation(ta: TypeAnnotation, tokens: list):
    """Add semantic token for a type annotation."""
    tokens.append((
        ta.span.line_start - 1,
        ta.span.col_start - 1,
        len(ta.base),
        _ST_TYPE,
        0,
    ))


def _sem_delta_encode(tokens: list[tuple]) -> list[int]:
    """Delta-encode sorted tokens into LSP integer array."""
    tokens.sort(key=lambda t: (t[0], t[1]))
    data = []
    prev_line = 0
    prev_start = 0
    for line, col, length, token_type, modifiers in tokens:
        delta_line = line - prev_line
        delta_start = col - prev_start if delta_line == 0 else col
        data.extend([delta_line, delta_start, length, token_type, modifiers])
        prev_line = line
        prev_start = col
    return data


@server.feature(
    types.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
    types.SemanticTokensRegistrationOptions(legend=_SEM_LEGEND, full=True),
)
def semantic_tokens_full(ls: LanguageServer, params: types.SemanticTokensParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return types.SemanticTokens(data=[])

    tokens: list[tuple] = []

    for sd in result.program.struct_defs:
        _sem_collect_tokens(sd, tokens, set())

    for fn in _local_functions(result.program):
        _sem_collect_tokens(fn, tokens, set())

    data = _sem_delta_encode(tokens)
    return types.SemanticTokens(data=data)


# ---------------------------------------------------------------------------
# Code Actions
# ---------------------------------------------------------------------------

def _ca_edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _ca_edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[len(b)]


def _ca_find_similar(name: str, candidates, max_distance: int = 2) -> list[str]:
    """Find candidate names within edit distance of the given name."""
    results = []
    for c in candidates:
        if c == name:
            continue
        dist = _ca_edit_distance(name, c)
        if dist <= max_distance:
            results.append((dist, c))
    results.sort()
    return [c for _, c in results[:5]]


_CA_UNKNOWN_PATTERN = re.compile(r"['\u2018](\w+)['\u2019]")


def _ca_collect_var_names(program: Program | None) -> set[str]:
    """Collect all variable names from the program."""
    names: set[str] = set()
    if not program:
        return names
    for fn in program.functions:
        for p in fn.params:
            names.add(p.name)
        _ca_collect_block_vars(fn.body, names)
    return names


def _ca_collect_block_vars(block: Block, names: set[str]):
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt):
            names.add(stmt.name)
        elif isinstance(stmt, ExprStmt):
            _ca_collect_expr_vars(stmt.expr, names)
    if block.expr is not None:
        _ca_collect_expr_vars(block.expr, names)


def _ca_collect_expr_vars(expr, names: set[str]):
    if isinstance(expr, ScanExpr):
        names.add(expr.carry_var)
        for ev in expr.elem_vars:
            names.add(ev)
        _ca_collect_block_vars(expr.body, names)
    elif isinstance(expr, MapExpr):
        names.add(expr.elem_var)
        _ca_collect_block_vars(expr.body, names)
    elif isinstance(expr, IfExpr):
        _ca_collect_block_vars(expr.then_block, names)
        _ca_collect_block_vars(expr.else_block, names)


@server.feature(types.TEXT_DOCUMENT_CODE_ACTION)
def code_actions(ls: LanguageServer, params: types.CodeActionParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result:
        return None

    actions: list[types.CodeAction] = []

    def _all_candidates() -> set[str]:
        c: set[str] = set()
        c.update(_ca_collect_var_names(result.program))
        if result.fn_table:
            c.update(k for k in result.fn_table if "." not in k)
        c.update(_BUILTINS)
        if result.struct_defs:
            c.update(result.struct_defs.keys())
        return c

    for diag in params.context.diagnostics:
        msg = diag.message.lower()

        candidates: set[str] = set()

        if "function" in msg:
            if result.fn_table:
                candidates.update(k for k in result.fn_table if "." not in k)
            candidates.update(_BUILTINS)
        elif "struct" in msg:
            if result.struct_defs:
                candidates.update(result.struct_defs.keys())
        elif "variable" in msg or "undefined" in msg or "unknown" in msg:
            candidates = _all_candidates()

        if not candidates:
            candidates = _all_candidates()

        match = _CA_UNKNOWN_PATTERN.search(diag.message)
        if not match:
            continue
        unknown_name = match.group(1)

        suggestions = _ca_find_similar(unknown_name, candidates)

        for suggestion in suggestions:
            actions.append(types.CodeAction(
                title=f"Did you mean '{suggestion}'?",
                kind=types.CodeActionKind.QuickFix,
                diagnostics=[diag],
                edit=types.WorkspaceEdit(
                    changes={uri: [
                        types.TextEdit(
                            range=diag.range,
                            new_text=suggestion,
                        )
                    ]}
                ),
            ))

    return actions if actions else None


# ---------------------------------------------------------------------------
# Folding Ranges
# ---------------------------------------------------------------------------

def _fold_collect_ranges(node, ranges: list):
    """Recursively collect folding ranges from AST nodes."""
    if hasattr(node, 'span') and node.span.line_end > node.span.line_start:
        if isinstance(node, (StructDef, ScanExpr, MapExpr, Block)):
            ranges.append(types.FoldingRange(
                start_line=node.span.line_start - 1,
                end_line=node.span.line_end - 1,
                kind=types.FoldingRangeKind.Region,
            ))

    for child in _children_of(node):
        _fold_collect_ranges(child, ranges)


def _build_folding_ranges(result: AnalysisResult) -> list[types.FoldingRange]:
    if not result or not result.program:
        return []

    ranges: list[types.FoldingRange] = []

    for sd in result.program.struct_defs:
        _fold_collect_ranges(sd, ranges)

    for fn in _local_functions(result.program):
        _fold_collect_ranges_fn(fn, ranges)

    return ranges


def _fold_collect_ranges_fn(fn: FnDef, ranges: list):
    if fn.span.line_end > fn.span.line_start:
        ranges.append(types.FoldingRange(
            start_line=fn.span.line_start - 1,
            end_line=fn.span.line_end - 1,
            kind=types.FoldingRangeKind.Region,
        ))

    if fn.body:
        for child in _children_of(fn.body):
            _fold_collect_ranges(child, ranges)


@server.feature(types.TEXT_DOCUMENT_FOLDING_RANGE)
def folding_ranges(ls: LanguageServer, params: types.FoldingRangeParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None

    ranges = _build_folding_ranges(result)
    return ranges if ranges else None


# ---------------------------------------------------------------------------
# Selection Range
# ---------------------------------------------------------------------------


def _sel_collect_ancestors(node, line: int, col: int, ancestors: list) -> bool:
    """Walk AST depth-first, collecting ancestor nodes that contain the position."""
    if not hasattr(node, "span") or not _span_contains(node.span, line, col):
        return False

    ancestors.append(node)

    for child in _children_of(node):
        if _sel_collect_ancestors(child, line, col, ancestors):
            return True

    return True


def _sel_build_chain(ancestors: list) -> types.SelectionRange | None:
    """Build a SelectionRange linked list from a list of ancestors (outermost first)."""
    if not ancestors:
        return None

    chain = None
    for node in ancestors:
        r = _span_to_range(node.span)
        chain = types.SelectionRange(range=r, parent=chain)

    return chain


@server.feature(types.TEXT_DOCUMENT_SELECTION_RANGE)
def selection_ranges(ls: LanguageServer, params: types.SelectionRangeParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None

    results: list[types.SelectionRange] = []

    for pos in params.positions:
        line = pos.line + 1
        col = pos.character + 1

        ancestors: list = []
        for fn in _local_functions(result.program):
            if _sel_collect_ancestors(fn, line, col, ancestors):
                break

        if not ancestors:
            for sd in result.program.struct_defs:
                if hasattr(sd, "span") and _span_contains(sd.span, line, col):
                    ancestors.append(sd)
                    break

        chain = _sel_build_chain(ancestors)
        if chain is None:
            chain = types.SelectionRange(
                range=types.Range(start=pos, end=pos),
                parent=None,
            )

        results.append(chain)

    return results


# ---------------------------------------------------------------------------
# Document Highlight
# ---------------------------------------------------------------------------


def _spans_to_highlights(result, name, kind, fn_scope=None):
    """Convert symbol spans to DocumentHighlight items with Read/Write classification."""
    all_spans = _refs_collect_all(result, name, kind,
                                  include_declaration=True, fn_scope=fn_scope)
    usage_spans = _refs_collect_all(result, name, kind,
                                    include_declaration=False, fn_scope=fn_scope)
    usage_set = {id(s) for s in usage_spans}
    highlights = []
    for s in all_spans:
        hk = (types.DocumentHighlightKind.Read
               if id(s) in usage_set
               else types.DocumentHighlightKind.Write)
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
# Go-to-Type-Definition
# ---------------------------------------------------------------------------


def _goto_type_definition(node, fn, result):
    """Find the struct definition span for the type of the given node."""
    program = result.program
    if program is None:
        return None

    # For Param nodes, check the type annotation directly.
    if isinstance(node, Param):
        base = node.type_annotation.base
        for sdef in program.struct_defs:
            if sdef.name == base:
                return sdef.span
        return None

    # For LetStmt, look up the type of the value expression.
    if isinstance(node, LetStmt):
        typ = result.type_map.get(id(node.value))
        if isinstance(typ, StructType):
            for sdef in program.struct_defs:
                if sdef.name == typ.name:
                    return sdef.span
        return None

    # For Identifier and other expressions, look up type_map.
    node_id = id(node)
    typ = result.type_map.get(node_id)
    if isinstance(typ, StructType):
        for sdef in program.struct_defs:
            if sdef.name == typ.name:
                return sdef.span
# Call Hierarchy
# ---------------------------------------------------------------------------


def _collect_calls_to(node, target_name: str, spans: list):
    """Recursively walk AST collecting spans of CallExpr nodes that call target_name."""
    if isinstance(node, CallExpr) and node.callee == target_name:
        spans.append(node.span)
    for child in _children_of(node):
        _collect_calls_to(child, target_name, spans)


def _collect_all_calls(node, calls_dict: dict):
    """Recursively walk AST collecting all CallExpr nodes grouped by callee name."""
    if isinstance(node, CallExpr):
        calls_dict.setdefault(node.callee, []).append(node.span)
    for child in _children_of(node):
        _collect_all_calls(child, calls_dict)


def _make_hierarchy_item(fn: FnDef, uri: str) -> types.CallHierarchyItem:
    """Build a CallHierarchyItem for a FnDef."""
    return types.CallHierarchyItem(
        name=fn.name,
        kind=types.SymbolKind.Function,
        uri=uri,
        range=_span_to_range(fn.span),
        selection_range=_span_to_range(fn.span),
    )


def _call_hierarchy_prepare(
    result: AnalysisResult, uri: str, line_0: int, col_0: int,
) -> list[types.CallHierarchyItem] | None:
    """Core logic for textDocument/prepareCallHierarchy."""
    if not result or not result.program:
        return None

    line = line_0 + 1
    col = col_0 + 1
    local_fns = _local_functions(result.program)
    fn_names = {fn.name for fn in local_fns}

    # Find the node at cursor
    target_name = None
    for fn in local_fns:
        node = _find_node_at(fn, line, col)
        if node is not None:
            if isinstance(node, FnDef):
                target_name = node.name
            elif isinstance(node, CallExpr):
                target_name = node.callee
            elif isinstance(node, Identifier) and node.name in fn_names:
                target_name = node.name
            break

    if target_name is None:
        return None

    # Find the FnDef for that name
    for fn in local_fns:
        if fn.name == target_name:
            return [_make_hierarchy_item(fn, uri)]

    return None


@server.feature(types.TEXT_DOCUMENT_DOCUMENT_HIGHLIGHT)
def document_highlight(ls: LanguageServer, params: types.DocumentHighlightParams):
# Code Lens
# ---------------------------------------------------------------------------

def _build_code_lenses(result: AnalysisResult, uri: str) -> list[types.CodeLens]:
    """Build code lenses for each function: '▶ Run' and 'N references'."""
    lenses: list[types.CodeLens] = []

    for fn in _local_functions(result.program):
        line = fn.span.line_start - 1  # 1-indexed → 0-indexed
        lens_range = types.Range(
            start=types.Position(line=line, character=0),
            end=types.Position(line=line, character=0),
        )

        # "▶ Run" lens — only if all params have concrete types (no symbolic dims)
        sig = result.fn_table.get(fn.name)
        if sig is not None:
            all_concrete = all(
                not isinstance(pt, WildcardArrayType)
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
def _call_hierarchy_incoming(
    result: AnalysisResult, uri: str, fn_name: str,
) -> list[types.CallHierarchyIncomingCall]:
    """Find all functions that call fn_name."""
    if not result or not result.program:
        return []

    incoming: list[types.CallHierarchyIncomingCall] = []
    for fn in _local_functions(result.program):
        spans: list = []
        _collect_calls_to(fn.body, fn_name, spans)
        if spans:
            incoming.append(types.CallHierarchyIncomingCall(
                from_=_make_hierarchy_item(fn, uri),
                from_ranges=[_span_to_range(s) for s in spans],
            ))

    return incoming


def _call_hierarchy_outgoing(
    result: AnalysisResult, uri: str, fn_name: str,
) -> list[types.CallHierarchyOutgoingCall]:
    """Find all user-defined functions called by fn_name."""
    if not result or not result.program:
        return []

    local_fns = _local_functions(result.program)
    fn_by_name = {fn.name: fn for fn in local_fns}

    target_fn = fn_by_name.get(fn_name)
    if target_fn is None:
        return []

    # Collect all calls from this function's body
    calls_dict: dict[str, list] = {}
    _collect_all_calls(target_fn.body, calls_dict)

    outgoing: list[types.CallHierarchyOutgoingCall] = []
    for callee_name, spans in calls_dict.items():
        callee_fn = fn_by_name.get(callee_name)
        if callee_fn is None:
            continue
        outgoing.append(types.CallHierarchyOutgoingCall(
            to=_make_hierarchy_item(callee_fn, uri),
            from_ranges=[_span_to_range(s) for s in spans],
        ))

    return outgoing


@server.feature(types.TEXT_DOCUMENT_PREPARE_CALL_HIERARCHY)
def prepare_call_hierarchy(ls: LanguageServer, params: types.CallHierarchyPrepareParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None
    line = params.position.line + 1
    col = params.position.character + 1
    return _build_document_highlights(result, line, col)


@server.feature(types.TEXT_DOCUMENT_TYPE_DEFINITION)
def goto_type_definition(ls: LanguageServer, params: types.TypeDefinitionParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None
    line = params.position.line + 1
    col = params.position.character + 1
    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            span = _goto_type_definition(node, fn, result)
            if span is not None:
                return types.Location(uri=uri, range=_span_to_range(span))
    return None
# Workspace Symbols
# ---------------------------------------------------------------------------

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


@server.feature(types.WORKSPACE_SYMBOL)
def workspace_symbols(ls: LanguageServer, params: types.WorkspaceSymbolParams):
    return _workspace_symbols(params.query)
    return _build_code_lenses(result, uri)
    line = params.position.line
    col = params.position.character
    return _call_hierarchy_prepare(result, uri, line, col)


@server.feature(types.CALL_HIERARCHY_INCOMING_CALLS)
def incoming_calls(ls: LanguageServer, params: types.CallHierarchyIncomingCallsParams):
    uri = params.item.uri
    fn_name = params.item.name
    result = _cache.get(uri)
    return _call_hierarchy_incoming(result, uri, fn_name)


@server.feature(types.CALL_HIERARCHY_OUTGOING_CALLS)
def outgoing_calls(ls: LanguageServer, params: types.CallHierarchyOutgoingCallsParams):
    uri = params.item.uri
    fn_name = params.item.name
    result = _cache.get(uri)
    return _call_hierarchy_outgoing(result, uri, fn_name)
# On-Type Formatting
# ---------------------------------------------------------------------------

def _compute_brace_depth(lines: list[str], target_line: int) -> int:
    """Count net { minus } through all lines before *target_line*."""
    depth = 0
    for i in range(target_line):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    return max(0, depth)


def _indent_edit(
    line_0: int, current_line: str, expected_indent: str
) -> types.TextEdit | None:
    """Return a TextEdit replacing leading whitespace, or *None* if already correct."""
    stripped = current_line.lstrip()
    actual_len = len(current_line) - len(stripped)
    if current_line[:actual_len] == expected_indent:
        return None
    return types.TextEdit(
        range=types.Range(
            start=types.Position(line=line_0, character=0),
            end=types.Position(line=line_0, character=actual_len),
        ),
        new_text=expected_indent,
    )


def _on_type_format(
    source: str, line_0: int, col_0: int, ch: str
) -> list[types.TextEdit]:
    lines = source.splitlines()
    if line_0 >= len(lines):
        return []
    current_line = lines[line_0]
    edits: list[types.TextEdit] = []

    if ch == "}":
        # Fix indentation of closing brace.
        stripped = current_line.lstrip()
        if stripped.startswith("}"):
            depth = max(0, _compute_brace_depth(lines, line_0) - 1)
            edit = _indent_edit(line_0, current_line, "    " * depth)
            if edit is not None:
                edits.append(edit)

    elif ch == ";":
        # Remove trailing whitespace.
        rstripped = current_line.rstrip()
        if len(rstripped) < len(current_line):
            edits.append(
                types.TextEdit(
                    range=types.Range(
                        start=types.Position(
                            line=line_0, character=len(rstripped)
                        ),
                        end=types.Position(
                            line=line_0, character=len(current_line)
                        ),
                    ),
                    new_text="",
                )
            )

    elif ch == "\n":
        # Auto-indent new line based on brace depth.
        if line_0 > 0:
            depth = _compute_brace_depth(lines, line_0)
            edit = _indent_edit(line_0, current_line, "    " * depth)
            if edit is not None:
                edits.append(edit)

    return edits


@server.feature(
    types.TEXT_DOCUMENT_ON_TYPE_FORMATTING,
    types.DocumentOnTypeFormattingOptions(
        first_trigger_character="}",
        more_trigger_character=[";", "\n"],
    ),
)
def on_type_formatting(
    ls: LanguageServer, params: types.DocumentOnTypeFormattingParams
):
    uri = params.text_document.uri
    doc = ls.workspace.get_text_document(uri)
    line = params.position.line  # already 0-indexed
    col = params.position.character
    return _on_type_format(doc.source, line, col, params.ch)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start_server():
    server.start_io()
