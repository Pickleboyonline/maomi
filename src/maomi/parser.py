from __future__ import annotations

from .tokens import Token, TokenType
from .ast_nodes import (
    Span,
    Program,
    FnDef,
    Param,
    Block,
    TypeAnnotation,
    Dim,
    LetStmt,
    ExprStmt,
    IntLiteral,
    FloatLiteral,
    BoolLiteral,
    Identifier,
    UnaryOp,
    BinOp,
    IfExpr,
    CallExpr,
    ScanExpr,
    MapExpr,
    GradExpr,
    StructDef,
    StructLiteral,
    FieldAccess,
    WithExpr,
    Expr,
)
from .errors import ParseError


COMPARISON_OPS = {TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.GT, TokenType.LEQ, TokenType.GEQ}
ADDITION_OPS = {TokenType.PLUS, TokenType.MINUS}
MULTIPLICATION_OPS = {TokenType.STAR, TokenType.SLASH}
BASE_TYPES = {TokenType.F32, TokenType.F64, TokenType.I32, TokenType.I64, TokenType.BOOL_TYPE}


class Parser:
    def __init__(self, tokens: list[Token], filename: str = "<stdin>"):
        self.tokens = tokens
        self.filename = filename
        self.pos = 0

    # -- Helpers --

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _at_end(self) -> bool:
        return self._current().type == TokenType.EOF

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _check(self, token_type: TokenType) -> bool:
        return self._current().type == token_type

    def _match(self, *types: TokenType) -> Token | None:
        if self._current().type in types:
            return self._advance()
        return None

    def _expect(self, token_type: TokenType) -> Token:
        tok = self._current()
        if tok.type != token_type:
            raise ParseError(
                f"expected {token_type.value}, got {tok.type.value} ({tok.value!r})",
                self.filename,
                tok.line,
                tok.col,
            )
        return self._advance()

    def _span_from(self, start: Token) -> Span:
        prev = self.tokens[self.pos - 1]
        return Span(start.line, start.col, prev.line, prev.col + len(prev.value))

    def _error(self, msg: str) -> ParseError:
        tok = self._current()
        return ParseError(msg, self.filename, tok.line, tok.col)

    # -- Top level --

    def parse(self) -> Program:
        start = self._current()
        struct_defs: list[StructDef] = []
        functions: list[FnDef] = []
        while not self._at_end():
            if self._check(TokenType.STRUCT):
                struct_defs.append(self._parse_struct_def())
            else:
                functions.append(self._parse_fn_def())
        return Program(struct_defs, functions, self._span_from(start))

    # -- Struct definition --

    def _parse_struct_def(self) -> StructDef:
        start = self._expect(TokenType.STRUCT)
        name = self._expect(TokenType.IDENT).value
        self._expect(TokenType.LBRACE)
        fields: list[tuple[str, TypeAnnotation]] = []
        if not self._check(TokenType.RBRACE):
            field_name = self._expect(TokenType.IDENT).value
            self._expect(TokenType.COLON)
            field_type = self._parse_type()
            fields.append((field_name, field_type))
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RBRACE):
                    break  # trailing comma
                field_name = self._expect(TokenType.IDENT).value
                self._expect(TokenType.COLON)
                field_type = self._parse_type()
                fields.append((field_name, field_type))
        self._expect(TokenType.RBRACE)
        return StructDef(name, fields, self._span_from(start))

    # -- Function definition --

    def _parse_fn_def(self) -> FnDef:
        start = self._expect(TokenType.FN)
        name = self._expect(TokenType.IDENT).value
        self._expect(TokenType.LPAREN)
        params = self._parse_param_list()
        self._expect(TokenType.RPAREN)
        self._expect(TokenType.ARROW)
        return_type = self._parse_type()
        body = self._parse_block()
        return FnDef(name, params, return_type, body, self._span_from(start))

    def _parse_param_list(self) -> list[Param]:
        params: list[Param] = []
        if self._check(TokenType.RPAREN):
            return params
        params.append(self._parse_param())
        while self._match(TokenType.COMMA):
            params.append(self._parse_param())
        return params

    def _parse_param(self) -> Param:
        start = self._current()
        name = self._expect(TokenType.IDENT).value
        self._expect(TokenType.COLON)
        type_ann = self._parse_type()
        return Param(name, type_ann, self._span_from(start))

    # -- Types --

    def _parse_type(self) -> TypeAnnotation:
        start = self._current()
        if start.type in BASE_TYPES:
            base = self._advance().value
            dims = None
            if self._match(TokenType.LBRACKET):
                dims = [self._parse_dim()]
                while self._match(TokenType.COMMA):
                    dims.append(self._parse_dim())
                self._expect(TokenType.RBRACKET)
            return TypeAnnotation(base, dims, self._span_from(start))
        if start.type == TokenType.IDENT:
            base = self._advance().value
            return TypeAnnotation(base, None, self._span_from(start))
        raise self._error(f"expected type, got {start.type.value}")

    def _parse_dim(self) -> Dim:
        tok = self._current()
        if tok.type == TokenType.INT_LIT:
            self._advance()
            return Dim(int(tok.value), Span(tok.line, tok.col, tok.line, tok.col + len(tok.value)))
        if tok.type == TokenType.IDENT:
            self._advance()
            return Dim(tok.value, Span(tok.line, tok.col, tok.line, tok.col + len(tok.value)))
        raise self._error(f"expected dimension (integer or name), got {tok.type.value}")

    # -- Blocks --

    def _parse_block(self) -> Block:
        start = self._expect(TokenType.LBRACE)
        stmts: list[LetStmt | ExprStmt] = []
        trailing_expr: Expr | None = None

        while not self._check(TokenType.RBRACE):
            if self._check(TokenType.LET):
                stmts.append(self._parse_let_stmt())
            else:
                expr = self._parse_expr()
                if self._match(TokenType.SEMICOLON):
                    stmts.append(ExprStmt(expr, expr.span))
                elif self._check(TokenType.RBRACE):
                    trailing_expr = expr
                else:
                    raise self._error("expected ';' or '}'")

        self._expect(TokenType.RBRACE)
        return Block(stmts, trailing_expr, self._span_from(start))

    # -- Statements --

    def _parse_let_stmt(self) -> LetStmt:
        start = self._expect(TokenType.LET)
        name = self._expect(TokenType.IDENT).value
        type_ann = None
        if self._match(TokenType.COLON):
            type_ann = self._parse_type()
        self._expect(TokenType.ASSIGN)
        value = self._parse_expr()
        self._expect(TokenType.SEMICOLON)
        return LetStmt(name, type_ann, value, self._span_from(start))

    # -- Expressions (precedence climbing) --

    def _parse_expr(self) -> Expr:
        if self._check(TokenType.SCAN):
            return self._parse_scan()
        if self._check(TokenType.MAP):
            return self._parse_map()
        if self._check(TokenType.GRAD):
            return self._parse_grad()
        if self._check(TokenType.IF):
            return self._parse_if_expr()
        expr = self._parse_comparison()
        if self._check(TokenType.WITH):
            return self._parse_with(expr)
        return expr

    def _parse_scan(self) -> ScanExpr:
        start = self._expect(TokenType.SCAN)
        self._expect(TokenType.LPAREN)
        carry_var = self._expect(TokenType.IDENT).value
        self._expect(TokenType.COMMA)

        # Multi-sequence: (acc, (x, y)) or single: (acc, x)
        if self._check(TokenType.LPAREN):
            self._advance()
            elem_vars = [self._expect(TokenType.IDENT).value]
            while self._match(TokenType.COMMA):
                elem_vars.append(self._expect(TokenType.IDENT).value)
            self._expect(TokenType.RPAREN)
        else:
            elem_vars = [self._expect(TokenType.IDENT).value]

        self._expect(TokenType.RPAREN)
        self._expect(TokenType.IN)
        self._expect(TokenType.LPAREN)
        init = self._parse_expr()
        self._expect(TokenType.COMMA)

        # Multi-sequence: (init, (xs, ys)) or single: (init, xs)
        if len(elem_vars) > 1:
            self._expect(TokenType.LPAREN)
            sequences = [self._parse_expr()]
            while self._match(TokenType.COMMA):
                sequences.append(self._parse_expr())
            self._expect(TokenType.RPAREN)
        else:
            sequences = [self._parse_expr()]

        self._expect(TokenType.RPAREN)
        body = self._parse_block()
        return ScanExpr(carry_var, elem_vars, init, sequences, body, self._span_from(start))

    def _parse_map(self) -> MapExpr:
        start = self._expect(TokenType.MAP)
        elem_var = self._expect(TokenType.IDENT).value
        self._expect(TokenType.IN)
        sequence = self._parse_comparison()
        body = self._parse_block()
        return MapExpr(elem_var, sequence, body, self._span_from(start))

    def _parse_grad(self) -> GradExpr:
        start = self._expect(TokenType.GRAD)
        self._expect(TokenType.LPAREN)
        expr = self._parse_expr()
        self._expect(TokenType.COMMA)
        wrt = self._expect(TokenType.IDENT).value
        self._expect(TokenType.RPAREN)
        return GradExpr(expr, wrt, self._span_from(start))

    def _parse_if_expr(self) -> IfExpr:
        start = self._expect(TokenType.IF)
        condition = self._parse_comparison()
        then_block = self._parse_block()
        self._expect(TokenType.ELSE)
        else_block = self._parse_block()
        return IfExpr(condition, then_block, else_block, self._span_from(start))

    def _parse_with(self, base: Expr) -> WithExpr:
        start = self._expect(TokenType.WITH)
        self._expect(TokenType.LBRACE)
        updates: list[tuple[list[str], Expr]] = []
        if not self._check(TokenType.RBRACE):
            path, value = self._parse_with_field()
            updates.append((path, value))
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RBRACE):
                    break  # trailing comma
                path, value = self._parse_with_field()
                updates.append((path, value))
        self._expect(TokenType.RBRACE)
        return WithExpr(base, updates, Span(base.span.line_start, base.span.col_start,
                        self.tokens[self.pos - 1].line, self.tokens[self.pos - 1].col + 1))

    def _parse_with_field(self) -> tuple[list[str], Expr]:
        path = [self._expect(TokenType.IDENT).value]
        while self._match(TokenType.DOT):
            path.append(self._expect(TokenType.IDENT).value)
        self._expect(TokenType.ASSIGN)
        value = self._parse_expr()
        return path, value

    def _parse_comparison(self) -> Expr:
        left = self._parse_addition()
        if self._current().type in COMPARISON_OPS:
            op = self._advance().value
            right = self._parse_addition()
            left = BinOp(op, left, right, Span(left.span.line_start, left.span.col_start, right.span.line_end, right.span.col_end))
        return left

    def _parse_addition(self) -> Expr:
        left = self._parse_multiplication()
        while self._current().type in ADDITION_OPS:
            op = self._advance().value
            right = self._parse_multiplication()
            left = BinOp(op, left, right, Span(left.span.line_start, left.span.col_start, right.span.line_end, right.span.col_end))
        return left

    def _parse_multiplication(self) -> Expr:
        left = self._parse_matmul()
        while self._current().type in MULTIPLICATION_OPS:
            op = self._advance().value
            right = self._parse_matmul()
            left = BinOp(op, left, right, Span(left.span.line_start, left.span.col_start, right.span.line_end, right.span.col_end))
        return left

    def _parse_matmul(self) -> Expr:
        left = self._parse_power()
        while self._check(TokenType.AT):
            self._advance()
            right = self._parse_power()
            left = BinOp("@", left, right, Span(left.span.line_start, left.span.col_start, right.span.line_end, right.span.col_end))
        return left

    def _parse_power(self) -> Expr:
        base = self._parse_unary()
        if self._check(TokenType.STARSTAR):
            self._advance()
            exp = self._parse_power()  # right-associative
            return BinOp("**", base, exp, Span(base.span.line_start, base.span.col_start, exp.span.line_end, exp.span.col_end))
        return base

    def _parse_unary(self) -> Expr:
        if self._check(TokenType.MINUS):
            start = self._advance()
            operand = self._parse_unary()
            return UnaryOp("-", operand, self._span_from(start))
        return self._parse_postfix()

    def _parse_postfix(self) -> Expr:
        expr = self._parse_primary()
        while True:
            if self._check(TokenType.LPAREN) and isinstance(expr, Identifier):
                # Function call
                self._advance()
                args: list[Expr] = []
                if not self._check(TokenType.RPAREN):
                    args.append(self._parse_expr())
                    while self._match(TokenType.COMMA):
                        args.append(self._parse_expr())
                self._expect(TokenType.RPAREN)
                expr = CallExpr(expr.name, args, Span(expr.span.line_start, expr.span.col_start, self.tokens[self.pos - 1].line, self.tokens[self.pos - 1].col + 1))
            elif self._check(TokenType.LBRACE) and isinstance(expr, Identifier) and self._is_struct_literal():
                # Struct literal: Name { field: expr, ... }
                expr = self._parse_struct_literal(expr)
            elif self._check(TokenType.DOT):
                # Field access: expr.field
                self._advance()
                field_name = self._expect(TokenType.IDENT)
                expr = FieldAccess(expr, field_name.value, Span(expr.span.line_start, expr.span.col_start, field_name.line, field_name.col + len(field_name.value)))
            else:
                break
        return expr

    def _is_struct_literal(self) -> bool:
        """Lookahead: check if LBRACE starts a struct literal (IDENT COLON) or a block."""
        # Current token is LBRACE. Check tokens after it.
        p = self.pos + 1  # skip LBRACE
        if p < len(self.tokens) and self.tokens[p].type == TokenType.RBRACE:
            return True  # empty struct literal: Name {}
        if p + 1 < len(self.tokens):
            return (self.tokens[p].type == TokenType.IDENT and
                    self.tokens[p + 1].type == TokenType.COLON)
        return False

    def _parse_struct_literal(self, name_expr: Identifier) -> StructLiteral:
        self._expect(TokenType.LBRACE)
        fields: list[tuple[str, Expr]] = []
        if not self._check(TokenType.RBRACE):
            field_name = self._expect(TokenType.IDENT).value
            self._expect(TokenType.COLON)
            value = self._parse_expr()
            fields.append((field_name, value))
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RBRACE):
                    break  # trailing comma
                field_name = self._expect(TokenType.IDENT).value
                self._expect(TokenType.COLON)
                value = self._parse_expr()
                fields.append((field_name, value))
        self._expect(TokenType.RBRACE)
        return StructLiteral(name_expr.name, fields, Span(name_expr.span.line_start, name_expr.span.col_start, self.tokens[self.pos - 1].line, self.tokens[self.pos - 1].col + 1))

    def _parse_primary(self) -> Expr:
        tok = self._current()

        if tok.type == TokenType.INT_LIT:
            self._advance()
            span = Span(tok.line, tok.col, tok.line, tok.col + len(tok.value))
            return IntLiteral(int(tok.value), span)

        if tok.type == TokenType.FLOAT_LIT:
            self._advance()
            span = Span(tok.line, tok.col, tok.line, tok.col + len(tok.value))
            return FloatLiteral(float(tok.value), span)

        if tok.type == TokenType.TRUE:
            self._advance()
            span = Span(tok.line, tok.col, tok.line, tok.col + 4)
            return BoolLiteral(True, span)

        if tok.type == TokenType.FALSE:
            self._advance()
            span = Span(tok.line, tok.col, tok.line, tok.col + 5)
            return BoolLiteral(False, span)

        if tok.type == TokenType.IDENT:
            self._advance()
            span = Span(tok.line, tok.col, tok.line, tok.col + len(tok.value))
            return Identifier(tok.value, span)

        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expr()
            self._expect(TokenType.RPAREN)
            return expr

        raise self._error(f"expected expression, got {tok.type.value} ({tok.value!r})")
