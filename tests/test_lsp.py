"""Tests for the Maomi LSP server validation and hover logic."""

import pytest
from lsprotocol import types

from maomi.lsp import validate, _find_node_at, _error_to_diagnostic, _span_contains
from maomi.ast_nodes import (
    Identifier, IntLiteral, FloatLiteral, BinOp, CallExpr,
    LetStmt, FnDef, Block, Span,
)
from maomi.errors import MaomiError


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_source_no_diagnostics(self):
        source = "fn f(x: f32) -> f32 { x }"
        diags, result = validate(source, "<test>")
        assert diags == []
        assert result.program is not None
        assert len(result.type_map) > 0

    def test_lexer_error(self):
        source = "fn f() -> f32 { ! }"  # '!' alone is invalid
        diags, result = validate(source, "<test>")
        assert len(diags) == 1
        assert "unexpected" in diags[0].message.lower() or "!" in diags[0].message
        assert result.program is None

    def test_parse_error(self):
        source = "fn f( -> f32 { x }"  # missing ')'
        diags, result = validate(source, "<test>")
        assert len(diags) == 1
        assert result.program is None

    def test_type_error(self):
        source = "fn f(x: f32) -> i32 { x }"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        assert result.program is not None  # AST still available

    def test_multiple_type_errors(self):
        source = """
fn f(x: f32) -> i32 { x }
fn g(y: i32) -> f32 { y }
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 2

    def test_type_map_populated_despite_errors(self):
        source = "fn f(x: f32) -> i32 { x + 1.0 }"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        assert len(result.type_map) > 0


# ---------------------------------------------------------------------------
# Error to diagnostic conversion
# ---------------------------------------------------------------------------

class TestErrorConversion:
    def test_1indexed_to_0indexed(self):
        err = MaomiError("test error", "<test>", line=1, col=1)
        diag = _error_to_diagnostic(err)
        assert diag.range.start.line == 0
        assert diag.range.start.character == 0
        assert diag.message == "test error"
        assert diag.severity == types.DiagnosticSeverity.Error
        assert diag.source == "maomi"

    def test_multiline_position(self):
        err = MaomiError("bad thing", "<test>", line=10, col=5)
        diag = _error_to_diagnostic(err)
        assert diag.range.start.line == 9
        assert diag.range.start.character == 4


# ---------------------------------------------------------------------------
# AST node finding (hover support)
# ---------------------------------------------------------------------------

class TestSpanContains:
    def test_inside(self):
        span = Span(1, 1, 1, 10)
        assert _span_contains(span, 1, 5) is True

    def test_at_start(self):
        span = Span(1, 1, 1, 10)
        assert _span_contains(span, 1, 1) is True

    def test_at_end(self):
        span = Span(1, 1, 1, 10)
        assert _span_contains(span, 1, 10) is True

    def test_before(self):
        span = Span(2, 5, 2, 10)
        assert _span_contains(span, 1, 5) is False

    def test_after(self):
        span = Span(1, 1, 1, 10)
        assert _span_contains(span, 1, 11) is False

    def test_multiline(self):
        span = Span(1, 5, 3, 10)
        assert _span_contains(span, 2, 1) is True  # middle line, any col


class TestFindNodeAt:
    def test_find_identifier_in_function(self):
        source = "fn f(x: f32) -> f32 { x }"
        from maomi.lexer import Lexer
        from maomi.parser import Parser
        tokens = Lexer(source, "<test>").tokenize()
        program = Parser(tokens, "<test>").parse()
        fn = program.functions[0]

        # The trailing expr is an Identifier
        body_expr = fn.body.expr
        assert isinstance(body_expr, Identifier)

        # Find node at the identifier's position
        node = _find_node_at(fn, body_expr.span.line_start, body_expr.span.col_start)
        assert node is not None
        assert isinstance(node, Identifier)
        assert node.name == "x"

    def test_find_innermost_in_binop(self):
        source = "fn f(a: f32, b: f32) -> f32 { a + b }"
        from maomi.lexer import Lexer
        from maomi.parser import Parser
        tokens = Lexer(source, "<test>").tokenize()
        program = Parser(tokens, "<test>").parse()
        fn = program.functions[0]

        body_expr = fn.body.expr
        assert isinstance(body_expr, BinOp)

        # Cursor on 'a' should find Identifier("a"), not the BinOp
        left = body_expr.left
        assert isinstance(left, Identifier)
        node = _find_node_at(fn, left.span.line_start, left.span.col_start)
        assert isinstance(node, Identifier)
        assert node.name == "a"

    def test_returns_none_outside_span(self):
        source = "fn f(x: f32) -> f32 { x }"
        from maomi.lexer import Lexer
        from maomi.parser import Parser
        tokens = Lexer(source, "<test>").tokenize()
        program = Parser(tokens, "<test>").parse()
        fn = program.functions[0]

        # Line 99 is way outside the function
        node = _find_node_at(fn, 99, 1)
        assert node is None
