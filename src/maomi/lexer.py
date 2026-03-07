from .tokens import Token, TokenType, KEYWORDS
from .errors import LexerError


class Lexer:
    def __init__(self, source: str, filename: str = "<stdin>"):
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []

    def tokenize(self) -> list[Token]:
        while self.pos < len(self.source):
            if self._skip_whitespace_and_comments():
                continue  # doc comment emitted, re-enter skip loop
            if self.pos >= len(self.source):
                break
            self._read_token()
        self.tokens.append(Token(TokenType.EOF, "", self.line, self.col))
        return self.tokens

    # -- Helpers --

    def _peek(self) -> str:
        return self.source[self.pos]

    def _peek_next(self) -> str | None:
        if self.pos + 1 < len(self.source):
            return self.source[self.pos + 1]
        return None

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _add(self, token_type: TokenType, value: str, line: int, col: int):
        self.tokens.append(Token(token_type, value, line, col))

    def _error(self, msg: str):
        raise LexerError(msg, self.filename, self.line, self.col)

    # -- Whitespace and comments --

    def _skip_whitespace_and_comments(self) -> bool:
        while self.pos < len(self.source):
            ch = self._peek()
            if ch in " \t\r\n":
                self._advance()
            elif ch == "/" and self._peek_next() == "/":
                if self.pos + 2 < len(self.source) and self.source[self.pos + 2] == "/":
                    self._read_doc_comment()
                    return True
                self._skip_line_comment()
            else:
                break
        return False

    def _skip_line_comment(self):
        while self.pos < len(self.source) and self.source[self.pos] != "\n":
            self._advance()

    def _read_doc_comment(self):
        line, col = self.line, self.col
        self._advance()  # /
        self._advance()  # /
        self._advance()  # /
        # skip one optional leading space
        if self.pos < len(self.source) and self.source[self.pos] == " ":
            self._advance()
        start = self.pos
        while self.pos < len(self.source) and self.source[self.pos] != "\n":
            self._advance()
        text = self.source[start:self.pos]
        self._add(TokenType.DOC_COMMENT, text, line, col)

    # -- Token dispatch --

    def _read_token(self):
        ch = self._peek()
        line, col = self.line, self.col

        if ch.isdigit():
            self._read_number()
        elif ch.isalpha() or ch == "_":
            self._read_identifier()
        else:
            self._read_operator_or_delimiter()

    # -- Numbers --

    def _read_number(self):
        start = self.pos
        line, col = self.line, self.col
        is_float = False

        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            self._advance()

        if self.pos < len(self.source) and self.source[self.pos] == ".":
            next_after_dot = self.pos + 1
            if next_after_dot < len(self.source) and self.source[next_after_dot].isdigit():
                is_float = True
                self._advance()  # consume '.'
                while self.pos < len(self.source) and self.source[self.pos].isdigit():
                    self._advance()

        if self.pos < len(self.source) and self.source[self.pos] in "eE":
            is_float = True
            self._advance()  # consume 'e'/'E'
            if self.pos < len(self.source) and self.source[self.pos] in "+-":
                self._advance()
            if self.pos >= len(self.source) or not self.source[self.pos].isdigit():
                self._error("expected digit after exponent")
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                self._advance()

        value = self.source[start : self.pos]
        token_type = TokenType.FLOAT_LIT if is_float else TokenType.INT_LIT
        self._add(token_type, value, line, col)

    # -- Identifiers and keywords --

    def _read_identifier(self):
        start = self.pos
        line, col = self.line, self.col

        while self.pos < len(self.source) and (
            self.source[self.pos].isalnum() or self.source[self.pos] == "_"
        ):
            self._advance()

        value = self.source[start : self.pos]
        token_type = KEYWORDS.get(value, TokenType.IDENT)
        self._add(token_type, value, line, col)

    # -- Operators and delimiters --

    def _read_operator_or_delimiter(self):
        line, col = self.line, self.col
        ch = self._advance()

        match ch:
            case ".":
                if self.pos < len(self.source) and self.source[self.pos] == ".":
                    self._advance()
                    self._add(TokenType.DOTDOT, "..", line, col)
                else:
                    self._add(TokenType.DOT, ch, line, col)
            case '"':
                self._read_string(line, col)
            case "(":
                self._add(TokenType.LPAREN, ch, line, col)
            case ")":
                self._add(TokenType.RPAREN, ch, line, col)
            case "{":
                self._add(TokenType.LBRACE, ch, line, col)
            case "}":
                self._add(TokenType.RBRACE, ch, line, col)
            case "[":
                self._add(TokenType.LBRACKET, ch, line, col)
            case "]":
                self._add(TokenType.RBRACKET, ch, line, col)
            case ",":
                self._add(TokenType.COMMA, ch, line, col)
            case ":":
                self._add(TokenType.COLON, ch, line, col)
            case ";":
                self._add(TokenType.SEMICOLON, ch, line, col)
            case "@":
                self._add(TokenType.AT, ch, line, col)
            case "+":
                self._add(TokenType.PLUS, ch, line, col)
            case "/":
                self._add(TokenType.SLASH, ch, line, col)
            case "*":
                if self.pos < len(self.source) and self.source[self.pos] == "*":
                    self._advance()
                    self._add(TokenType.STARSTAR, "**", line, col)
                else:
                    self._add(TokenType.STAR, ch, line, col)
            case "-":
                if self.pos < len(self.source) and self.source[self.pos] == ">":
                    self._advance()
                    self._add(TokenType.ARROW, "->", line, col)
                else:
                    self._add(TokenType.MINUS, ch, line, col)
            case "=":
                if self.pos < len(self.source) and self.source[self.pos] == "=":
                    self._advance()
                    self._add(TokenType.EQ, "==", line, col)
                else:
                    self._add(TokenType.ASSIGN, ch, line, col)
            case "!":
                if self.pos < len(self.source) and self.source[self.pos] == "=":
                    self._advance()
                    self._add(TokenType.NEQ, "!=", line, col)
                else:
                    self._error(f"unexpected character '!'")
            case "<":
                if self.pos < len(self.source) and self.source[self.pos] == "=":
                    self._advance()
                    self._add(TokenType.LEQ, "<=", line, col)
                else:
                    self._add(TokenType.LT, ch, line, col)
            case ">":
                if self.pos < len(self.source) and self.source[self.pos] == "=":
                    self._advance()
                    self._add(TokenType.GEQ, ">=", line, col)
                else:
                    self._add(TokenType.GT, ch, line, col)
            case "|":
                if self.pos < len(self.source) and self.source[self.pos] == ">":
                    self._advance()
                    self._add(TokenType.PIPE, "|>", line, col)
                else:
                    self._error("unexpected character '|'")
            case _:
                self._error(f"unexpected character: {ch!r}")

    def _read_string(self, line: int, col: int):
        chars: list[str] = []
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch == '"':
                self._advance()
                self._add(TokenType.STRING_LIT, "".join(chars), line, col)
                return
            if ch == "\n":
                self._error("unterminated string literal")
            chars.append(ch)
            self._advance()
        self._error("unterminated string literal")
