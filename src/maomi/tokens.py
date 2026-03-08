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
    VALUE_AND_GRAD = auto()
    IN = auto()
    WHILE = auto()
    DO = auto()
    LIMIT = auto()
    CAST = auto()
    FOLD = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    STRUCT = auto()
    WITH = auto()
    AND = auto()
    OR = auto()
    NOT = auto()

    # Type keywords
    F32 = auto()
    F64 = auto()
    BF16 = auto()
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
    DOTDOT = auto()

    # String literal
    STRING_LIT = auto()

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
    PIPE = auto()

    # Comments
    DOC_COMMENT = auto()

    # Compile-time
    COMPTIME = auto()

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
    "bf16": TokenType.BF16,
    "i32": TokenType.I32,
    "i64": TokenType.I64,
    "bool": TokenType.BOOL_TYPE,
    "scan": TokenType.SCAN,
    "map": TokenType.MAP,
    "grad": TokenType.GRAD,
    "value_and_grad": TokenType.VALUE_AND_GRAD,
    "in": TokenType.IN,
    "import": TokenType.IMPORT,
    "from": TokenType.FROM,
    "as": TokenType.AS,
    "struct": TokenType.STRUCT,
    "with": TokenType.WITH,
    "while": TokenType.WHILE,
    "do": TokenType.DO,
    "limit": TokenType.LIMIT,
    "cast": TokenType.CAST,
    "fold": TokenType.FOLD,
    "comptime": TokenType.COMPTIME,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
}
