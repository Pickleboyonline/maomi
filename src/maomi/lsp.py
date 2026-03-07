from __future__ import annotations

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
    BinOp, UnaryOp, IfExpr, CallExpr, ScanExpr, MapExpr,
    GradExpr, Identifier, IntLiteral, FloatLiteral, BoolLiteral,
    StructLiteral, FieldAccess, WithExpr, IndexExpr, StructDef,
    _ScanGrad, _IndexGrad, _GatherGrad, _Conv2dGrad,
    _MaxPoolGrad, _AvgPoolGrad, _BroadcastExpr,
)
from .types import MaomiType, StructType


@dataclass
class AnalysisResult:
    program: Program | None
    type_map: dict[int, MaomiType]
    fn_table: dict
    struct_defs: dict


_EMPTY_RESULT = AnalysisResult(None, {}, {}, {})

server = LanguageServer("maomi-lsp", "0.1.0")
_cache: dict[str, AnalysisResult] = {}


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

    for fn in result.program.functions:
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

    # FnDef: show full signature
    if isinstance(node, FnDef):
        sig = result.fn_table.get(node.name)
        if sig is not None:
            params = ", ".join(f"{n}: {t}" for n, t in zip(sig.param_names, sig.param_types))
            return f"```maomi\nfn {node.name}({params}) -> {sig.return_type}\n```"
        return None

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
    "fn", "let", "if", "else", "scan", "map", "grad",
    "struct", "with", "import", "from", "in", "true", "false",
]

_TYPE_NAMES = ["f32", "f64", "i32", "i64", "bool"]

_BUILTINS = [
    "mean", "sum", "exp", "log", "tanh", "sqrt", "abs",
    "reshape", "concat", "iota", "transpose", "callback",
    "rng_key", "rng_split", "rng_uniform", "rng_normal",
    "conv2d", "max_pool", "avg_pool",
]

_BUILTIN_SET = set(_BUILTINS)


@server.feature(types.TEXT_DOCUMENT_COMPLETION)
def completions(ls: LanguageServer, params: types.CompletionParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    doc = ls.workspace.get_text_document(uri)

    lines = doc.source.splitlines()
    if params.position.line >= len(lines):
        return None
    line_text = lines[params.position.line]
    col = params.position.character

    # Dot context: struct field completion
    if col > 0 and line_text[col - 1] == ".":
        return _complete_dot(result, params.position)

    return _complete_general(result, params.position)


def _complete_dot(result: AnalysisResult | None, position: types.Position):
    if not result or not result.program:
        return None

    line = position.line + 1
    # The dot is at position.character (0-indexed), so the expr before it
    # ends at position.character - 1 (0-indexed) = position.character (1-indexed)
    col = position.character

    for fn in result.program.functions:
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
        items.append(types.CompletionItem(
            label=b, kind=types.CompletionItemKind.Function, detail="builtin",
        ))

    if result and result.program:
        # User-defined functions
        for name, sig in result.fn_table.items():
            if name in _BUILTIN_SET or "." in name:
                continue
            params = ", ".join(
                f"{n}: {t}" for n, t in zip(sig.param_names, sig.param_types)
            )
            items.append(types.CompletionItem(
                label=name,
                kind=types.CompletionItemKind.Function,
                detail=f"({params}) -> {sig.return_type}",
            ))

        # Struct names
        for name in result.struct_defs:
            items.append(types.CompletionItem(
                label=name, kind=types.CompletionItemKind.Struct,
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

    for fn in result.program.functions:
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
# Entry point
# ---------------------------------------------------------------------------

def start_server():
    server.start_io()
