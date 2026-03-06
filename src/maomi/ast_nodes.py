from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Span:
    line_start: int
    col_start: int
    line_end: int
    col_end: int


# ---------- Types ----------


@dataclass
class Dim:
    value: int | str  # concrete int or symbolic name
    span: Span


@dataclass
class TypeAnnotation:
    base: str  # "f32", "f64", "i32", "i64", "bool"
    dims: list[Dim] | None  # None = scalar
    span: Span


# ---------- Function ----------


@dataclass
class Param:
    name: str
    type_annotation: TypeAnnotation
    span: Span


@dataclass
class FnDef:
    name: str
    params: list[Param]
    return_type: TypeAnnotation
    effect: str | None
    body: Block
    span: Span


# ---------- Blocks ----------


@dataclass
class Block:
    stmts: list[LetStmt | ExprStmt]
    expr: Expr | None  # trailing expression = implicit return
    span: Span


# ---------- Statements ----------


@dataclass
class LetStmt:
    name: str
    type_annotation: TypeAnnotation | None
    value: Expr
    span: Span


@dataclass
class ExprStmt:
    expr: Expr
    span: Span


# ---------- Expressions ----------


@dataclass
class IntLiteral:
    value: int
    span: Span


@dataclass
class FloatLiteral:
    value: float
    span: Span


@dataclass
class BoolLiteral:
    value: bool
    span: Span


@dataclass
class Identifier:
    name: str
    span: Span


@dataclass
class UnaryOp:
    op: str
    operand: Expr
    span: Span


@dataclass
class BinOp:
    op: str
    left: Expr
    right: Expr
    span: Span


@dataclass
class IfExpr:
    condition: Expr
    then_block: Block
    else_block: Block
    span: Span


@dataclass
class CallExpr:
    callee: str
    args: list[Expr]
    span: Span


@dataclass
class ScanExpr:
    carry_var: str
    elem_var: str
    init: Expr
    sequence: Expr
    body: Block
    span: Span


@dataclass
class MapExpr:
    elem_var: str
    sequence: Expr
    body: Block
    span: Span


@dataclass
class GradExpr:
    expr: Expr
    wrt: str  # variable name to differentiate with respect to
    span: Span


# Union types for convenience
Expr = IntLiteral | FloatLiteral | BoolLiteral | Identifier | UnaryOp | BinOp | IfExpr | CallExpr | ScanExpr | MapExpr | GradExpr
Stmt = LetStmt | ExprStmt


# ---------- Program ----------


@dataclass
class Program:
    functions: list[FnDef]
    span: Span
