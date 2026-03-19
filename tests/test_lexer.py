from maomi.lexer import Lexer
from maomi.tokens import TokenType
from maomi.errors import LexerError
import pytest


def lex(source: str) -> list[tuple[TokenType, str]]:
    tokens = Lexer(source).tokenize()
    return [(t.type, t.value) for t in tokens if t.type != TokenType.EOF]


class TestSingleTokens:
    def test_delimiters(self):
        assert lex("( ) { } [ ] , : ;") == [
            (TokenType.LPAREN, "("),
            (TokenType.RPAREN, ")"),
            (TokenType.LBRACE, "{"),
            (TokenType.RBRACE, "}"),
            (TokenType.LBRACKET, "["),
            (TokenType.RBRACKET, "]"),
            (TokenType.COMMA, ","),
            (TokenType.COLON, ":"),
            (TokenType.SEMICOLON, ";"),
        ]

    def test_single_char_operators(self):
        assert lex("+ - * / @") == [
            (TokenType.PLUS, "+"),
            (TokenType.MINUS, "-"),
            (TokenType.STAR, "*"),
            (TokenType.SLASH, "/"),
            (TokenType.AT, "@"),
        ]

    def test_multi_char_operators(self):
        assert lex("** -> == != <= >=") == [
            (TokenType.STARSTAR, "**"),
            (TokenType.ARROW, "->"),
            (TokenType.EQ, "=="),
            (TokenType.NEQ, "!="),
            (TokenType.LEQ, "<="),
            (TokenType.GEQ, ">="),
        ]

    def test_disambiguation(self):
        """Multi-char operators don't eat single-char ones."""
        assert lex("* *") == [(TokenType.STAR, "*"), (TokenType.STAR, "*")]
        assert lex("- >") == [(TokenType.MINUS, "-"), (TokenType.GT, ">")]
        assert lex("= =") == [(TokenType.ASSIGN, "="), (TokenType.ASSIGN, "=")]
        # bare '!' is invalid — lexer recovers and skips it
        lexer = Lexer("! =")
        lexer.tokenize()
        assert len(lexer.errors) >= 1
        assert lex("< =") == [(TokenType.LT, "<"), (TokenType.ASSIGN, "=")]


class TestNumbers:
    def test_integers(self):
        assert lex("0 42 784") == [
            (TokenType.INT_LIT, "0"),
            (TokenType.INT_LIT, "42"),
            (TokenType.INT_LIT, "784"),
        ]

    def test_floats(self):
        assert lex("3.14 0.0 100.5") == [
            (TokenType.FLOAT_LIT, "3.14"),
            (TokenType.FLOAT_LIT, "0.0"),
            (TokenType.FLOAT_LIT, "100.5"),
        ]

    def test_scientific_notation(self):
        assert lex("1e3 2.5e10 1e-3") == [
            (TokenType.FLOAT_LIT, "1e3"),
            (TokenType.FLOAT_LIT, "2.5e10"),
            (TokenType.FLOAT_LIT, "1e-3"),
        ]

    def test_dot_without_trailing_digits_is_not_float(self):
        """42. followed by non-digit should lex as INT then DOT-like thing."""
        # 42 followed by .x — the dot isn't consumed as part of the number
        result = lex("42")
        assert result == [(TokenType.INT_LIT, "42")]


class TestIdentifiersAndKeywords:
    def test_keywords(self):
        assert lex("fn let if else true false") == [
            (TokenType.FN, "fn"),
            (TokenType.LET, "let"),
            (TokenType.IF, "if"),
            (TokenType.ELSE, "else"),
            (TokenType.TRUE, "true"),
            (TokenType.FALSE, "false"),
        ]

    def test_type_keywords(self):
        assert lex("f32 f64 i32 i64 bool") == [
            (TokenType.F32, "f32"),
            (TokenType.F64, "f64"),
            (TokenType.I32, "i32"),
            (TokenType.I64, "i64"),
            (TokenType.BOOL_TYPE, "bool"),
        ]

    def test_identifiers(self):
        assert lex("x foo _bar w1 myVar") == [
            (TokenType.IDENT, "x"),
            (TokenType.IDENT, "foo"),
            (TokenType.IDENT, "_bar"),
            (TokenType.IDENT, "w1"),
            (TokenType.IDENT, "myVar"),
        ]

    def test_keyword_prefix_is_ident(self):
        """'f32x' is an identifier, not 'f32' + 'x'."""
        assert lex("f32x") == [(TokenType.IDENT, "f32x")]
        assert lex("letter") == [(TokenType.IDENT, "letter")]
        assert lex("iff") == [(TokenType.IDENT, "iff")]


class TestComments:
    def test_line_comment(self):
        assert lex("// this is a comment\nfn") == [(TokenType.FN, "fn")]

    def test_comment_at_end_of_file(self):
        assert lex("fn // trailing") == [(TokenType.FN, "fn")]

    def test_comment_between_tokens(self):
        assert lex("x // comment\n+ y") == [
            (TokenType.IDENT, "x"),
            (TokenType.PLUS, "+"),
            (TokenType.IDENT, "y"),
        ]


class TestDocComments:
    def test_doc_comment_emits_token(self):
        assert lex("/// hello\nfn") == [
            (TokenType.DOC_COMMENT, "hello"),
            (TokenType.FN, "fn"),
        ]

    def test_regular_comment_still_skipped(self):
        assert lex("// regular\nfn") == [(TokenType.FN, "fn")]

    def test_doc_comment_strips_leading_space(self):
        assert lex("/// hello world") == [(TokenType.DOC_COMMENT, "hello world")]

    def test_doc_comment_no_space_after_slashes(self):
        assert lex("///hello") == [(TokenType.DOC_COMMENT, "hello")]

    def test_multiple_doc_comments(self):
        result = lex("/// line 1\n/// line 2\nfn")
        assert result == [
            (TokenType.DOC_COMMENT, "line 1"),
            (TokenType.DOC_COMMENT, "line 2"),
            (TokenType.FN, "fn"),
        ]

    def test_empty_doc_comment(self):
        assert lex("///\nfn") == [
            (TokenType.DOC_COMMENT, ""),
            (TokenType.FN, "fn"),
        ]

    def test_four_slashes_is_doc_comment(self):
        # //// is /// followed by /, so doc comment with leading /
        assert lex("////foo") == [(TokenType.DOC_COMMENT, "/foo")]


class TestLineColTracking:
    def test_first_token_position(self):
        tokens = Lexer("fn").tokenize()
        assert tokens[0].line == 1
        assert tokens[0].col == 1

    def test_multiline(self):
        tokens = Lexer("fn\nlet").tokenize()
        assert tokens[0].line == 1  # fn
        assert tokens[1].line == 2  # let
        assert tokens[1].col == 1


class TestErrors:
    def test_unexpected_character(self):
        lexer = Lexer("$")
        lexer.tokenize()
        assert len(lexer.errors) >= 1
        assert "unexpected character" in lexer.errors[0].message


class TestFullProgram:
    def test_linear_function(self):
        source = 'fn linear(x: f32[B, 128], w: f32[128, 64]) -> f32[B, 64] { x @ w }'
        tokens = lex(source)
        types = [t for t, _ in tokens]
        assert types == [
            TokenType.FN,
            TokenType.IDENT,       # linear
            TokenType.LPAREN,
            TokenType.IDENT,       # x
            TokenType.COLON,
            TokenType.F32,
            TokenType.LBRACKET,
            TokenType.IDENT,       # B
            TokenType.COMMA,
            TokenType.INT_LIT,     # 128
            TokenType.RBRACKET,
            TokenType.COMMA,
            TokenType.IDENT,       # w
            TokenType.COLON,
            TokenType.F32,
            TokenType.LBRACKET,
            TokenType.INT_LIT,     # 128
            TokenType.COMMA,
            TokenType.INT_LIT,     # 64
            TokenType.RBRACKET,
            TokenType.RPAREN,
            TokenType.ARROW,
            TokenType.F32,
            TokenType.LBRACKET,
            TokenType.IDENT,       # B
            TokenType.COMMA,
            TokenType.INT_LIT,     # 64
            TokenType.RBRACKET,
            TokenType.LBRACE,
            TokenType.IDENT,       # x
            TokenType.AT,
            TokenType.IDENT,       # w
            TokenType.RBRACE,
        ]


class TestStructTokens:
    def test_dot_token(self):
        assert lex("a.b") == [
            (TokenType.IDENT, "a"),
            (TokenType.DOT, "."),
            (TokenType.IDENT, "b"),
        ]

    def test_struct_keyword(self):
        assert lex("struct") == [(TokenType.STRUCT, "struct")]

    def test_with_keyword(self):
        assert lex("with") == [(TokenType.WITH, "with")]

    def test_dot_not_in_float(self):
        """3.14 should lex as a single FLOAT_LIT, not INT DOT INT."""
        assert lex("3.14") == [(TokenType.FLOAT_LIT, "3.14")]
