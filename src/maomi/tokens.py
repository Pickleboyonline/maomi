from enum import StrEnum, auto
from dataclasses import dataclass


class TokenType(StrEnum):
    # Literals
    INT_LIT = auto()
    FLOAT_LIT = auto()

    # Identifiers
    IDENT = auto()

    # Keywords
    FN = auto()
    LET = auto()
    IF = auto()
    ELSE = auto()
    TRUE = auto()
    FALSE = auto()
    SCAN = auto()
    MAP = auto()
    GRAD = auto()
    IN = auto()
    STRUCT = auto()
    WITH = auto()

    # Type keywords
    F32 = auto()
    F64 = auto()
    I32 = auto()
    I64 = auto()
    BOOL_TYPE = auto()

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    DOT = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    AT = auto()
    STARSTAR = auto()
    ARROW = auto()
    ASSIGN = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LEQ = auto()
    GEQ = auto()

    # Special
    EOF = auto()


@dataclass(frozen=True, slots=True)
class Token:
    type: TokenType
    value: str
    line: int
    col: int


KEYWORDS: dict[str, TokenType] = {
    "fn": TokenType.FN,
    "let": TokenType.LET,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "f32": TokenType.F32,
    "f64": TokenType.F64,
    "i32": TokenType.I32,
    "i64": TokenType.I64,
    "bool": TokenType.BOOL_TYPE,
    "scan": TokenType.SCAN,
    "map": TokenType.MAP,
    "grad": TokenType.GRAD,
    "in": TokenType.IN,
    "struct": TokenType.STRUCT,
    "with": TokenType.WITH,
}
