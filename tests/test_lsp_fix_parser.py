"""Tests for silent _error() bugs in parser.py (C1: missing raise).

The parser's parse() method catches ParseError and appends to .errors,
so we verify the errors are collected (not silently dropped).
"""

import pytest

from maomi.ast_nodes import Program
from maomi.errors import ParseError
from maomi.lexer import Lexer
from maomi.parser import Parser


def parse(source: str) -> tuple[Program, list[ParseError]]:
    tokens = Lexer(source).tokenize()
    p = Parser(tokens)
    prog = p.parse()
    return prog, p.errors


class TestCastInvalidType:
    def test_cast_with_string_type_raises_parse_error(self):
        """cast(x, string) should produce a ParseError, not a KeyError crash."""
        _prog, errors = parse("fn f(x: f32) -> f32 { cast(x, string) }")
        assert any("cast: expected a type" in e.message for e in errors)

    def test_cast_with_identifier_raises_parse_error(self):
        """cast(x, foo) should produce a ParseError."""
        _prog, errors = parse("fn f(x: f32) -> f32 { cast(x, foo) }")
        assert any("cast: expected a type" in e.message for e in errors)

    def test_cast_with_valid_types_still_works(self):
        """Ensure valid cast types are unaffected by the fix."""
        for ty in ("f32", "f64", "bf16", "i32", "i64", "bool"):
            prog, errors = parse(f"fn f(x: f32) -> {ty} {{ cast(x, {ty}) }}")
            assert prog.functions[0].name == "f"
            # No errors for valid types
            assert not any("cast: expected a type" in e.message for e in errors)


class TestWhileLimitNonPositive:
    def test_while_limit_zero_raises_parse_error(self):
        """while ... limit 0 should produce a ParseError."""
        _prog, errors = parse(
            "fn f(x: f32) -> f32 {"
            "  while s in x limit 0 { true } do { s }"
            "}"
        )
        assert any("limit must be positive" in e.message for e in errors)

    def test_while_limit_negative_raises_parse_error(self):
        """while ... limit -1 should produce a ParseError.
        -1 is lexed as MINUS then INT_LIT, so _expect(INT_LIT) raises
        ParseError when it sees MINUS. The point is it doesn't silently succeed."""
        _prog, errors = parse(
            "fn f(x: f32) -> f32 {"
            "  while s in x limit -1 { true } do { s }"
            "}"
        )
        assert len(errors) > 0

    def test_while_limit_positive_still_works(self):
        """Ensure valid positive limits are unaffected."""
        prog, errors = parse(
            "fn f(x: f32) -> f32 {"
            "  while s in x limit 100 { true } do { s }"
            "}"
        )
        assert prog.functions[0].name == "f"
