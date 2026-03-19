from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ..lexer import Lexer
from ..parser import Parser
from ..resolver import resolve
from ..type_checker import TypeChecker
from ..errors import MaomiError, MaomiTypeError, LexerError, ParseError
from ..ast_nodes import Program, FnDef
from ..types import MaomiType

logger = logging.getLogger("maomi-lsp")


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
    return [fn for fn in program.functions
            if "." not in fn.name and fn.source_file is None]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(source: str, filename: str) -> tuple[list[types.Diagnostic], AnalysisResult]:
    logger.debug("Validating %s (%d chars)", filename, len(source))
    diagnostics: list[types.Diagnostic] = []

    try:
        tokens = Lexer(source, filename=filename).tokenize()
    except LexerError as e:
        logger.debug("Lexer error in %s: %s", filename, e.message)
        diagnostics.append(_error_to_diagnostic(e))
        return diagnostics, _EMPTY_RESULT

    try:
        program = Parser(tokens, filename=filename).parse()
    except ParseError as e:
        logger.debug("Parse error in %s: %s", filename, e.message)
        diagnostics.append(_error_to_diagnostic(e))
        return diagnostics, _EMPTY_RESULT

    if not program.functions and not program.struct_defs and not program.type_aliases:
        return diagnostics, AnalysisResult(program, {}, {}, {})

    try:
        program = resolve(program, filename)
    except MaomiError as e:
        logger.debug("Resolve error in %s: %s", filename, e.message)
        diagnostics.append(_error_to_diagnostic(e))
        return diagnostics, _EMPTY_RESULT
    except Exception:
        logger.debug("Unexpected resolver error in %s", filename, exc_info=True)
        return diagnostics, _EMPTY_RESULT

    checker = TypeChecker(filename=filename)
    try:
        type_errors = checker.check(program)
        for e in type_errors:
            diagnostics.append(_error_to_diagnostic(e))
    except MaomiTypeError as e:
        diagnostics.append(_error_to_diagnostic(e))
    except Exception:
        logger.debug("Unexpected type-checker error in %s", filename, exc_info=True)

    logger.info("Validated %s: %d diagnostics, %d types, %d functions",
                filename, len(diagnostics), len(checker.type_map), len(checker.fn_table))

    return diagnostics, AnalysisResult(
        program, checker.type_map, dict(checker.fn_table), dict(checker.struct_defs),
    )


def _error_to_diagnostic(e: MaomiError) -> types.Diagnostic:
    line = max(0, e.line - 1)
    col = max(0, e.col - 1)
    severity = types.DiagnosticSeverity.Error
    if getattr(e, "severity", None) == "warning":
        severity = types.DiagnosticSeverity.Warning
    return types.Diagnostic(
        range=types.Range(
            start=types.Position(line=line, character=col),
            end=types.Position(line=line, character=col + 1),
        ),
        message=e.message,
        severity=severity,
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
    logger.debug("Cache updated for %s: program=%s",
                 uri, "OK" if result.program else "NONE")
    ls.text_document_publish_diagnostics(types.PublishDiagnosticsParams(
        uri=uri, diagnostics=diags,
    ))


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams):
    logger.info("Document opened: %s", params.text_document.uri)
    _do_validate(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams):
    logger.debug("Document changed: %s (version %s)",
                 params.text_document.uri, params.text_document.version)
    _do_validate(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams):
    logger.debug("Document saved: %s", params.text_document.uri)
    _do_validate(ls, params.text_document.uri)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start_server():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Check for MAOMI_LSP_LOG env var to enable verbose logging
    log_level = os.environ.get("MAOMI_LSP_LOG", "").upper()
    if log_level in ("DEBUG", "INFO", "WARNING"):
        logging.getLogger("maomi-lsp").setLevel(getattr(logging, log_level))
    server.start_io()
