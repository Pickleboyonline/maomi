"""Tests for silent _error() bugs in parser.py (C1: missing raise)."""

import pytest

from maomi.ast_nodes import Program
from maomi.errors import ParseError
from maomi.lexer import Lexer
from maomi.parser import Parser


def parse(source: str) -> Program:
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


class TestCastInvalidType:
    def test_cast_with_string_type_raises_parse_error(self):
        """cast(x, string) should produce a ParseError, not a KeyError crash."""
        with pytest.raises(ParseError, match="cast: expected a type"):
            parse("fn f(x: f32) -> f32 { cast(x, string) }")

    def test_cast_with_identifier_raises_parse_error(self):
        """cast(x, foo) should produce a ParseError."""
        with pytest.raises(ParseError, match="cast: expected a type"):
            parse("fn f(x: f32) -> f32 { cast(x, foo) }")

    def test_cast_with_valid_types_still_works(self):
        """Ensure valid cast types are unaffected by the fix."""
        for ty in ("f32", "f64", "bf16", "i32", "i64", "bool"):
            prog = parse(f"fn f(x: f32) -> {ty} {{ cast(x, {ty}) }}")
            assert prog.functions[0].name == "f"


class TestWhileLimitNonPositive:
    def test_while_limit_zero_raises_parse_error(self):
        """while ... limit 0 should produce a ParseError."""
        with pytest.raises(ParseError, match="limit must be positive"):
            parse(
                "fn f(x: f32) -> f32 {"
                "  while s in x limit 0 { true } do { s }"
                "}"
            )

    def test_while_limit_negative_raises_parse_error(self):
        """while ... limit -1 should produce a ParseError.
        -1 is lexed as MINUS then INT_LIT, so _expect(INT_LIT) raises
        ParseError when it sees MINUS. The point is it doesn't silently succeed."""
        with pytest.raises(ParseError):
            parse(
                "fn f(x: f32) -> f32 {"
                "  while s in x limit -1 { true } do { s }"
                "}"
            )

    def test_while_limit_positive_still_works(self):
        """Ensure valid positive limits are unaffected."""
        prog = parse(
            "fn f(x: f32) -> f32 {"
            "  while s in x limit 100 { true } do { s }"
            "}"
        )
        assert prog.functions[0].name == "f"
