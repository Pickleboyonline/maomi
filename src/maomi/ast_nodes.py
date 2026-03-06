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
    base: str  # "f32", "f64", "i32", "i64", "bool", or struct name
    dims: list[Dim] | None  # None = scalar (or struct)
    span: Span


# ---------- Struct ----------


@dataclass
class StructDef:
    name: str
    fields: list[tuple[str, TypeAnnotation]]
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
    elem_vars: list[str]
    init: Expr
    sequences: list[Expr]
    body: Block
    span: Span
    reverse: bool = False


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


@dataclass
class StructLiteral:
    name: str
    fields: list[tuple[str, Expr]]
    span: Span


@dataclass
class FieldAccess:
    object: Expr
    field: str
    span: Span


@dataclass
class WithExpr:
    base: Expr
    updates: list[tuple[list[str], Expr]]  # [(path, value)] — path like ["pcn", "w"]
    span: Span


@dataclass
class _ScanGrad:
    """Internal: backward pass of scan. Created by AD, compiled by codegen."""
    d_body_d_carry: Expr
    d_body_d_elems: list[Expr]
    carry_var: str
    elem_vars: list[str]
    init: Expr
    sequences: list[Expr]
    forward_result: Expr
    adj: Expr
    wrt: str
    span: Span


# Union types for convenience
Expr = IntLiteral | FloatLiteral | BoolLiteral | Identifier | UnaryOp | BinOp | IfExpr | CallExpr | ScanExpr | MapExpr | GradExpr | StructLiteral | FieldAccess | WithExpr | _ScanGrad
Stmt = LetStmt | ExprStmt


# ---------- Program ----------


@dataclass
class Program:
    struct_defs: list[StructDef]
    functions: list[FnDef]
    span: Span
