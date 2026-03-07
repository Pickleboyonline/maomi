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
from .types import MaomiType


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
# Entry point
# ---------------------------------------------------------------------------

def start_server():
    server.start_io()
