from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote, urlparse

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ..lexer import Lexer
from ..parser import Parser
from ..resolver import resolve
from ..type_checker import TypeChecker
from ..errors import MaomiError, LexerError, ParseError
from ..ast_nodes import Program, FnDef
from ..types import MaomiType


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
# LSP document lifecycle handlers
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start_server():
    server.start_io()
