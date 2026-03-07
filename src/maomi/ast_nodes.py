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
    wildcard: bool = False  # True for f32[..] shape wildcard


# ---------- Struct ----------


@dataclass
class StructDef:
    name: str
    fields: list[tuple[str, TypeAnnotation]]
    span: Span
    doc: str | None = field(default=None, compare=False)


# ---------- Function ----------


@dataclass
class Param:
    name: str
    type_annotation: TypeAnnotation
    span: Span
    comptime: bool = False


@dataclass
class FnDef:
    name: str
    params: list[Param]
    return_type: TypeAnnotation
    body: Block
    span: Span
    doc: str | None = field(default=None, compare=False)


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
class StringLiteral:
    value: str
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
    named_args: list[tuple[str, Expr]] = field(default_factory=list)


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
class WhileExpr:
    state_var: str          # loop state variable name
    init: Expr              # initial state value
    max_iters: int | None   # None = non-differentiable, int = bounded + differentiable
    cond: Block             # condition block (must return bool)
    body: Block             # body block (must return state type)
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


@dataclass
class CastExpr:
    expr: Expr
    target_type: str  # "f32", "f64", "i32", "i64", "bool"
    span: Span


@dataclass
class FoldExpr:
    carry_var: str
    elem_vars: list[str]
    init: Expr
    sequences: list[Expr]
    body: Block
    span: Span


@dataclass
class ArrayLiteral:
    elements: list[Expr]
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
class IndexComponent:
    kind: str              # "single" | "slice" | "full"
    value: Expr | None     # the index expr (kind == "single")
    start: Expr | None     # range start (kind == "slice")
    end: Expr | None       # range end (kind == "slice")
    span: Span
    static_size: int | None = None  # slice size, set by type checker


@dataclass
class IndexExpr:
    base: Expr
    indices: list[IndexComponent]
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


@dataclass
class _WhileGrad:
    """Internal: backward pass of bounded while. Pre-allocated trajectory."""
    d_body_d_state: Expr    # symbolic derivative of body w.r.t. state
    state_var: str
    init: Expr              # original init
    max_iters: int          # trajectory buffer size
    cond: Block             # original condition (for forward augmented loop)
    body: Block             # original body (for forward augmented loop)
    forward_result: Expr    # reference to forward WhileExpr
    adj: Expr               # upstream adjoint
    span: Span


@dataclass
class _IndexGrad:
    """Internal: backward pass of indexing. Created by AD, compiled by codegen."""
    base_expr: Expr               # original array being indexed (for shape)
    adj: Expr                     # adjoint of the indexed result
    indices: list[IndexComponent] # same indices as forward pass
    span: Span


@dataclass
class _GatherGrad:
    """Internal: backward pass of array-based indexing (gather).
    Created by AD, compiled by codegen as stablehlo.scatter."""
    base_expr: Expr               # original array (for shape, e.g. f32[V, D])
    adj: Expr                     # adjoint of gathered result (e.g. f32[B, D])
    indices: Expr                 # the index array (e.g. i32[B])
    gather_axis: int              # which operand axis was gathered
    span: Span


@dataclass
class _Conv2dGrad:
    """Internal: backward pass of conv2d. Created by AD, compiled by codegen."""
    input_expr: Expr              # original input (for grad w.r.t. kernel, and for shape)
    kernel_expr: Expr             # original kernel (for grad w.r.t. input, and for shape)
    adj: Expr                     # upstream adjoint
    wrt: str                      # "lhs" (input grad) or "rhs" (kernel grad)
    strides: tuple[int, int]
    padding: tuple[int, int]
    span: Span


@dataclass
class _MaxPoolGrad:
    """Internal: backward pass of max_pool. Created by AD, compiled by codegen."""
    input_expr: Expr              # original input (to find max positions)
    adj: Expr                     # upstream adjoint
    window: tuple[int, int]
    strides: tuple[int, int]
    span: Span


@dataclass
class _AvgPoolGrad:
    """Internal: backward pass of avg_pool. Created by AD, compiled by codegen."""
    input_expr: Expr              # original input (for shape)
    adj: Expr                     # upstream adjoint
    window: tuple[int, int]
    strides: tuple[int, int]
    span: Span


@dataclass
class _FoldGrad:
    """Internal: backward pass of fold. Created by AD, compiled by codegen."""
    d_body_d_carry: Expr              # symbolic derivative of body w.r.t. carry
    d_body_d_elems: list[Expr]        # symbolic derivatives w.r.t. each element var
    carry_var: str
    elem_vars: list[str]
    init: Expr                        # original init expression
    sequences: list[Expr]             # original sequence expressions
    body: Block                       # original fold body (for augmented forward in codegen)
    adj: Expr                         # upstream adjoint (carry-typed, not stacked)
    wrt: str                          # "__init__" or sequence var name
    span: Span


@dataclass
class _BroadcastExpr:
    """Internal: broadcast scalar/lower-rank to array shape. Created by AD for sum/mean backprop."""
    expr: Expr                    # expression to broadcast
    target_dims: tuple[int, ...]  # target shape dimensions
    span: Span
    broadcast_dims: tuple[int, ...] | None = None  # explicit dim mapping (None = right-align)


@dataclass
class _ReduceSum:
    """Internal: reduce-sum over specific dimensions. Created by AD for map free var gradients."""
    expr: Expr                    # array to reduce
    axes: tuple[int, ...]         # dimensions to reduce over
    span: Span


# Union types for convenience
Expr = IntLiteral | FloatLiteral | BoolLiteral | StringLiteral | Identifier | UnaryOp | BinOp | IfExpr | CallExpr | ScanExpr | WhileExpr | MapExpr | GradExpr | CastExpr | FoldExpr | ArrayLiteral | StructLiteral | FieldAccess | WithExpr | IndexExpr | _ScanGrad | _WhileGrad | _IndexGrad | _GatherGrad | _Conv2dGrad | _MaxPoolGrad | _AvgPoolGrad | _FoldGrad | _BroadcastExpr
Stmt = LetStmt | ExprStmt


# ---------- Imports ----------


@dataclass
class ImportDecl:
    module_path: str           # "math" or "../lib/nn"
    alias: str | None          # from "as nn", None = derive from module_path
    names: list[str] | None    # { relu, linear } or None = qualified import
    span: Span


# ---------- Program ----------


@dataclass
class Program:
    imports: list[ImportDecl]
    struct_defs: list[StructDef]
    functions: list[FnDef]
    span: Span
