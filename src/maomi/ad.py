"""Reverse-mode automatic differentiation as an AST-to-AST transform.

Rewrites GradExpr nodes into ordinary expression trees that compute
gradients via the chain rule. After this pass, the AST has no GradExpr
nodes — codegen never sees them.

Supported operations inside grad:
  arithmetic (+, -, *, /, **)
  matmul (@)
  elementwise builtins (exp, log, tanh, sqrt, abs)
  reductions (mean, sum)
  shape builtins (reshape, concat)
  if/else (condition not differentiated; both branches differentiated)
  user function calls (inlined then differentiated)
  map (adjoint distributes over map body)
  let bindings
  identifiers, float/int literals

  grad-of-grad (nested grad expressions, e.g. grad(grad(x**3, x), x))

  grad-of-grad through indexing, scan (constant derivatives), broadcast, reduce

Not supported (emits compile error):
  grad-of-grad through scan (non-constant derivatives), conv2d, max_pool, avg_pool, while
"""
from __future__ import annotations

from dataclasses import dataclass
from .ast_nodes import (
    Program,
    FnDef,
    Block,
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
    WhileExpr,
    MapExpr,
    GradExpr,
    CastExpr,
    FoldExpr,
    ArrayLiteral,
    StructLiteral,
    FieldAccess,
    WithExpr,
    IndexExpr,
    IndexComponent,
    _ScanGrad,
    _WhileGrad,
    _IndexGrad,
    _GatherGrad,
    _Conv2dGrad,
    _MaxPoolGrad,
    _AvgPoolGrad,
    _FoldGrad,
    _BroadcastExpr,
    _ReduceSum,
    Span,
    Expr,
)
from .types import MaomiType, ScalarType, ArrayType, StructType
from .errors import MaomiError

_DUMMY_SPAN = Span(0, 0, 0, 0)

_ELEMENTWISE_BUILTINS = {"exp", "log", "tanh", "sqrt", "abs"}
_REDUCTION_BUILTINS = {"mean", "sum", "max", "min"}
_SHAPE_BUILTINS = {"reshape", "concat"}
_NONDIFF_BUILTINS = {"callback"}
_IOTA_BUILTINS = {"iota"}
_CONV_POOL_BUILTINS = {"conv2d", "max_pool", "avg_pool"}
_RNG_BUILTINS = {"random.key", "random.split", "random.uniform", "random.normal"}
_STOP_GRAD_BUILTINS = {"stop_gradient"}
_WHERE_BUILTINS = {"where"}
_ARGMAX_BUILTINS = {"argmax", "argmin"}
_MAX_GRAD_DEPTH = 10


def _collect_free_vars(expr: Expr) -> set[str]:
    """Collect all Identifier names referenced in an expression."""
    result: set[str] = set()
    _collect_free_vars_inner(expr, result)
    return result


def _collect_free_vars_inner(expr: Expr, result: set[str]):
    match expr:
        case Identifier(name=name):
            result.add(name)
        case UnaryOp(operand=operand):
            _collect_free_vars_inner(operand, result)
        case BinOp(left=left, right=right):
            _collect_free_vars_inner(left, result)
            _collect_free_vars_inner(right, result)
        case CallExpr(args=args):
            for a in args:
                _collect_free_vars_inner(a, result)
        case IfExpr(condition=cond, then_block=then_b, else_block=else_b):
            _collect_free_vars_inner(cond, result)
            if then_b.expr:
                _collect_free_vars_inner(then_b.expr, result)
            if else_b.expr:
                _collect_free_vars_inner(else_b.expr, result)
        case MapExpr(elem_var=ev, sequence=seq, body=body):
            _collect_free_vars_inner(seq, result)
            if body.expr:
                inner = set[str]()
                _collect_free_vars_inner(body.expr, inner)
                inner.discard(ev)  # elem_var is bound, not free
                result.update(inner)
        case CastExpr(expr=inner):
            _collect_free_vars_inner(inner, result)
        case ScanExpr(carry_var=cv, elem_vars=evs, init=init, sequences=seqs, body=body):
            _collect_free_vars_inner(init, result)
            for s in seqs:
                _collect_free_vars_inner(s, result)
            if body.expr:
                inner = set[str]()
                _collect_free_vars_inner(body.expr, inner)
                inner.discard(cv)
                for ev in evs:
                    inner.discard(ev)
                result.update(inner)
        case FoldExpr(carry_var=cv, elem_vars=evs, init=init, sequences=seqs, body=body):
            _collect_free_vars_inner(init, result)
            for s in seqs:
                _collect_free_vars_inner(s, result)
            if body.expr:
                inner = set[str]()
                _collect_free_vars_inner(body.expr, inner)
                inner.discard(cv)
                for ev in evs:
                    inner.discard(ev)
                result.update(inner)
        case WhileExpr(state_var=sv, init=init, cond=cond, body=body):
            _collect_free_vars_inner(init, result)
            inner = set[str]()
            if cond.expr:
                _collect_free_vars_inner(cond.expr, inner)
            if body.expr:
                _collect_free_vars_inner(body.expr, inner)
            inner.discard(sv)
            result.update(inner)
        case ArrayLiteral(elements=elems):
            for e in elems:
                _collect_free_vars_inner(e, result)
        case StructLiteral(fields=fields):
            for _, fv in fields:
                _collect_free_vars_inner(fv, result)
        case FieldAccess(object=obj):
            _collect_free_vars_inner(obj, result)
        case WithExpr(base=base, updates=updates):
            _collect_free_vars_inner(base, result)
            for _, ve in updates:
                _collect_free_vars_inner(ve, result)
        case IndexExpr(base=base, indices=indices):
            _collect_free_vars_inner(base, result)
            for ic in indices:
                if ic.value is not None:
                    _collect_free_vars_inner(ic.value, result)
                if ic.start is not None:
                    _collect_free_vars_inner(ic.start, result)
                if ic.end is not None:
                    _collect_free_vars_inner(ic.end, result)
        case _IndexGrad(base_expr=base, adj=adj, indices=indices):
            _collect_free_vars_inner(base, result)
            _collect_free_vars_inner(adj, result)
            for ic in indices:
                if ic.value is not None:
                    _collect_free_vars_inner(ic.value, result)
                if ic.start is not None:
                    _collect_free_vars_inner(ic.start, result)
                if ic.end is not None:
                    _collect_free_vars_inner(ic.end, result)
        case _GatherGrad(base_expr=base, adj=adj, indices=indices):
            _collect_free_vars_inner(base, result)
            _collect_free_vars_inner(adj, result)
            _collect_free_vars_inner(indices, result)
        case _ScanGrad():
            _collect_free_vars_inner(expr.d_body_d_carry, result)
            for de in expr.d_body_d_elems:
                _collect_free_vars_inner(de, result)
            _collect_free_vars_inner(expr.init, result)
            for s in expr.sequences:
                _collect_free_vars_inner(s, result)
            _collect_free_vars_inner(expr.forward_result, result)
            _collect_free_vars_inner(expr.adj, result)
        case _WhileGrad():
            _collect_free_vars_inner(expr.d_body_d_state, result)
            _collect_free_vars_inner(expr.init, result)
            _collect_free_vars_inner(expr.forward_result, result)
            _collect_free_vars_inner(expr.adj, result)
        case GradExpr(expr=inner_expr):
            _collect_free_vars_inner(inner_expr, result)
        case _Conv2dGrad(input_expr=ie, kernel_expr=ke, adj=adj):
            _collect_free_vars_inner(ie, result)
            _collect_free_vars_inner(ke, result)
            _collect_free_vars_inner(adj, result)
        case _MaxPoolGrad(input_expr=ie, adj=adj):
            _collect_free_vars_inner(ie, result)
            _collect_free_vars_inner(adj, result)
        case _AvgPoolGrad(input_expr=ie, adj=adj):
            _collect_free_vars_inner(ie, result)
            _collect_free_vars_inner(adj, result)
        case _FoldGrad(d_body_d_carry=dc, d_body_d_elems=des, init=init, sequences=seqs, adj=adj):
            _collect_free_vars_inner(dc, result)
            for de in des:
                _collect_free_vars_inner(de, result)
            _collect_free_vars_inner(init, result)
            for s in seqs:
                _collect_free_vars_inner(s, result)
            _collect_free_vars_inner(adj, result)
        case _BroadcastExpr(expr=e):
            _collect_free_vars_inner(e, result)
        case _ReduceSum(expr=e):
            _collect_free_vars_inner(e, result)


def transform_grad(program: Program, type_map: dict[int, MaomiType]) -> Program:
    """Walk the program and replace all GradExpr nodes with gradient expressions."""
    fn_defs = {fn.name: fn for fn in program.functions}
    transformer = ADTransform(type_map, fn_defs)
    new_fns = [transformer.transform_fn(fn) for fn in program.functions]
    return Program(program.imports, program.struct_defs, new_fns, program.span)


class ADTransform:
    def __init__(self, type_map: dict[int, MaomiType], fn_defs: dict[str, FnDef]):
        self.type_map = type_map
        self.fn_defs = fn_defs
        self._name_counter = 0
        self._inlining_stack: set[str] = set()  # cycle detection for inlining
        self._tape_exprs: dict[str, Expr] = {}  # tape name -> original AST expression
        self._grad_depth: int = 0  # nesting depth for grad-of-grad

    def _fresh_name(self, prefix: str = "_ad") -> str:
        self._name_counter += 1
        return f"{prefix}_{self._name_counter}"

    def _type_of(self, expr: Expr) -> MaomiType | None:
        return self.type_map.get(id(expr))

    def transform_fn(self, fn: FnDef) -> FnDef:
        new_body = self._transform_block(fn.body)
        return FnDef(fn.name, fn.params, fn.return_type, new_body, fn.span)

    def _transform_block(self, block: Block) -> Block:
        # Collect let bindings so grad can see through them
        let_env: dict[str, Expr] = {}
        new_stmts = []
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                new_val = self._transform_expr(stmt.value)
                let_env[stmt.name] = new_val
                new_stmts.append(LetStmt(stmt.name, stmt.type_annotation, new_val, stmt.span))
            elif isinstance(stmt, ExprStmt):
                new_stmts.append(ExprStmt(self._transform_expr(stmt.expr), stmt.span))

        new_expr = None
        if block.expr is not None:
            new_expr = self._transform_expr_with_lets(block.expr, let_env)

        return Block(new_stmts, new_expr, block.span)

    def _transform_expr_with_lets(self, expr: Expr, let_env: dict[str, Expr]) -> Expr:
        """Transform an expression with access to enclosing let bindings."""
        match expr:
            case GradExpr(expr=inner, wrt=wrt):
                return self._differentiate(inner, wrt, expr, let_env)
            case _:
                return self._transform_expr(expr)

    def _transform_expr(self, expr: Expr) -> Expr:
        """Recursively transform, replacing GradExpr nodes."""
        match expr:
            case GradExpr(expr=inner, wrt=wrt):
                return self._differentiate(inner, wrt, expr, {})
            case BinOp(op=op, left=left, right=right):
                new_left = self._transform_expr(left)
                new_right = self._transform_expr(right)
                result = BinOp(op, new_left, new_right, expr.span)
                self._copy_type(expr, result)
                return result
            case UnaryOp(op=op, operand=operand):
                new_operand = self._transform_expr(operand)
                result = UnaryOp(op, new_operand, expr.span)
                self._copy_type(expr, result)
                return result
            case CallExpr(callee=callee, args=args):
                new_args = [self._transform_expr(a) for a in args]
                result = CallExpr(callee, new_args, expr.span)
                self._copy_type(expr, result)
                return result
            case IfExpr():
                result = IfExpr(
                    self._transform_expr(expr.condition),
                    self._transform_block(expr.then_block),
                    self._transform_block(expr.else_block),
                    expr.span,
                )
                self._copy_type(expr, result)
                return result
            case ScanExpr():
                result = ScanExpr(
                    expr.carry_var,
                    expr.elem_vars,
                    self._transform_expr(expr.init),
                    [self._transform_expr(s) for s in expr.sequences],
                    self._transform_block(expr.body),
                    expr.span,
                    expr.reverse,
                )
                self._copy_type(expr, result)
                return result
            case CastExpr(expr=inner, target_type=target):
                result = CastExpr(self._transform_expr(inner), target, expr.span)
                self._copy_type(expr, result)
                return result
            case FoldExpr():
                result = FoldExpr(
                    expr.carry_var,
                    expr.elem_vars,
                    self._transform_expr(expr.init),
                    [self._transform_expr(s) for s in expr.sequences],
                    self._transform_block(expr.body),
                    expr.span,
                )
                self._copy_type(expr, result)
                return result
            case WhileExpr():
                result = WhileExpr(
                    expr.state_var,
                    self._transform_expr(expr.init),
                    expr.max_iters,
                    self._transform_block(expr.cond),
                    self._transform_block(expr.body),
                    expr.span,
                )
                self._copy_type(expr, result)
                return result
            case ArrayLiteral(elements=elems):
                new_elems = [self._transform_expr(e) for e in elems]
                result = ArrayLiteral(new_elems, expr.span)
                self._copy_type(expr, result)
                return result
            case StructLiteral(name=name, fields=fields):
                new_fields = [(fn, self._transform_expr(fv)) for fn, fv in fields]
                result = StructLiteral(name, new_fields, expr.span)
                self._copy_type(expr, result)
                return result
            case FieldAccess(object=obj, field=field):
                new_obj = self._transform_expr(obj)
                result = FieldAccess(new_obj, field, expr.span)
                self._copy_type(expr, result)
                return result
            case WithExpr(base=base, updates=updates):
                new_base = self._transform_expr(base)
                new_updates = [(path, self._transform_expr(ve)) for path, ve in updates]
                result = WithExpr(new_base, new_updates, expr.span)
                self._copy_type(expr, result)
                return result
            case IndexExpr(base=base, indices=indices):
                new_base = self._transform_expr(base)
                new_indices = [self._transform_index_component(ic) for ic in indices]
                result = IndexExpr(new_base, new_indices, expr.span)
                self._copy_type(expr, result)
                return result
            case _:
                return expr

    def _transform_index_component(self, ic: IndexComponent) -> IndexComponent:
        new_value = self._transform_expr(ic.value) if ic.value is not None else None
        new_start = self._transform_expr(ic.start) if ic.start is not None else None
        new_end = self._transform_expr(ic.end) if ic.end is not None else None
        return IndexComponent(ic.kind, new_value, new_start, new_end, ic.span)

    def _copy_type(self, old: Expr, new: Expr):
        t = self.type_map.get(id(old))
        if t is not None:
            self.type_map[id(new)] = t

    # -- Reverse-mode AD core --

    def _differentiate(self, expr: Expr, wrt: str, grad_expr: GradExpr,
                        let_env: dict[str, Expr]) -> Expr:
        """Compute d(expr)/d(wrt) using reverse-mode AD."""
        # Collect the computation into a tape
        tape: list[tuple[str, Expr]] = []  # (name, expr)
        var_map: dict[int, str] = {}  # id(expr) -> tape variable name
        self._linearize(expr, tape, var_map, let_env)

        # Build lookup from tape names to original AST expressions so
        # backprop can reference intermediate values without creating
        # Identifiers for internal tape names that codegen doesn't know about.
        self._tape_exprs = {name: node for name, node in tape}

        # The output is the last tape entry
        output_name = var_map[id(expr)]

        # Initialize adjoints
        adjoints: dict[str, Expr] = {}
        result_type = self._type_of(grad_expr)
        adjoints[output_name] = self._make_float(1.0)

        # Walk tape backwards, accumulating adjoints
        for name, node in reversed(tape):
            if name not in adjoints:
                continue
            adj = adjoints[name]
            self._backprop(name, node, adj, adjoints, var_map)

        # Return the adjoint for wrt
        if wrt in adjoints:
            result = adjoints[wrt]
        else:
            # wrt doesn't appear in the expression — gradient is zero
            if result_type is not None:
                result = self._make_zero(result_type)
            else:
                result = self._make_float(0.0)

        # Register the result type in type_map
        if result_type is not None:
            self.type_map[id(result)] = result_type

        return result

    def _linearize(self, expr: Expr, tape: list[tuple[str, Expr]],
                    var_map: dict[int, str], let_env: dict[str, Expr]):
        """Flatten the expression into a tape of named operations."""
        if id(expr) in var_map:
            return

        match expr:
            case Identifier(name=name):
                # If this identifier refers to a let binding, expand it
                if name in let_env:
                    self._linearize(let_env[name], tape, var_map, let_env)
                    # Map this identifier to the same tape entry as the let value
                    var_map[id(expr)] = var_map[id(let_env[name])]
                    return
                var_map[id(expr)] = name
                return
            case IntLiteral() | FloatLiteral() | BoolLiteral():
                name = self._fresh_name("const")
                var_map[id(expr)] = name
                tape.append((name, expr))
                return
            case UnaryOp(operand=operand):
                self._linearize(operand, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case BinOp(left=left, right=right):
                self._linearize(left, tape, var_map, let_env)
                self._linearize(right, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case CallExpr(callee=callee, args=args):
                if callee in _NONDIFF_BUILTINS:
                    # Callback: no value, no gradient. Skip entirely.
                    return
                elif callee in _IOTA_BUILTINS | _ELEMENTWISE_BUILTINS | _REDUCTION_BUILTINS | _SHAPE_BUILTINS | _CONV_POOL_BUILTINS | _RNG_BUILTINS | _STOP_GRAD_BUILTINS | _WHERE_BUILTINS | _ARGMAX_BUILTINS | {"transpose"}:
                    # Built-in: put on tape as-is
                    for a in args:
                        self._linearize(a, tape, var_map, let_env)
                    name = self._fresh_name("v")
                    var_map[id(expr)] = name
                    tape.append((name, expr))
                elif callee in self.fn_defs:
                    # User function: inline body then linearize
                    self._linearize_user_call(expr, callee, args, tape, var_map, let_env)
                else:
                    raise MaomiError(
                        f"grad: unknown function '{callee}' inside grad",
                        "<ad>", expr.span.line_start, expr.span.col_start,
                    )
            case IfExpr(condition=cond, then_block=then_b, else_block=else_b):
                self._linearize(cond, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case MapExpr(sequence=seq):
                self._linearize(seq, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case ScanExpr(init=init, sequences=seqs):
                self._linearize(init, tape, var_map, let_env)
                for s in seqs:
                    self._linearize(s, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case WhileExpr(init=init):
                self._linearize(init, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case FieldAccess(object=obj):
                self._linearize(obj, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case ArrayLiteral(elements=elems):
                for e in elems:
                    self._linearize(e, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case StructLiteral(fields=fields):
                for _, fv in fields:
                    self._linearize(fv, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case WithExpr(base=base, updates=updates):
                self._linearize(base, tape, var_map, let_env)
                for _, ve in updates:
                    self._linearize(ve, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case IndexExpr(base=base, indices=indices):
                self._linearize(base, tape, var_map, let_env)
                for ic in indices:
                    if ic.value is not None:
                        self._linearize(ic.value, tape, var_map, let_env)
                    if ic.start is not None:
                        self._linearize(ic.start, tape, var_map, let_env)
                    if ic.end is not None:
                        self._linearize(ic.end, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case CastExpr(expr=inner):
                self._linearize(inner, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case FoldExpr(init=init, sequences=seqs):
                self._linearize(init, tape, var_map, let_env)
                for s in seqs:
                    self._linearize(s, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case _IndexGrad(base_expr=base, adj=adj_inner, indices=indices):
                self._linearize(base, tape, var_map, let_env)
                self._linearize(adj_inner, tape, var_map, let_env)
                for ic in indices:
                    if ic.value is not None:
                        self._linearize(ic.value, tape, var_map, let_env)
                    if ic.start is not None:
                        self._linearize(ic.start, tape, var_map, let_env)
                    if ic.end is not None:
                        self._linearize(ic.end, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case _GatherGrad(base_expr=base, adj=adj_inner, indices=indices):
                self._linearize(base, tape, var_map, let_env)
                self._linearize(adj_inner, tape, var_map, let_env)
                self._linearize(indices, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case _BroadcastExpr(expr=inner):
                self._linearize(inner, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case _ReduceSum(expr=inner):
                self._linearize(inner, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case _ScanGrad():
                self._linearize(expr.init, tape, var_map, let_env)
                for s in expr.sequences:
                    self._linearize(s, tape, var_map, let_env)
                self._linearize(expr.adj, tape, var_map, let_env)
                self._linearize(expr.forward_result, tape, var_map, let_env)
                self._linearize(expr.d_body_d_carry, tape, var_map, let_env)
                for de in expr.d_body_d_elems:
                    self._linearize(de, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case _WhileGrad():
                self._linearize(expr.init, tape, var_map, let_env)
                self._linearize(expr.adj, tape, var_map, let_env)
                self._linearize(expr.forward_result, tape, var_map, let_env)
                self._linearize(expr.d_body_d_state, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case _Conv2dGrad():
                self._linearize(expr.input_expr, tape, var_map, let_env)
                self._linearize(expr.kernel_expr, tape, var_map, let_env)
                self._linearize(expr.adj, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case _MaxPoolGrad():
                self._linearize(expr.input_expr, tape, var_map, let_env)
                self._linearize(expr.adj, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case _AvgPoolGrad():
                self._linearize(expr.input_expr, tape, var_map, let_env)
                self._linearize(expr.adj, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case GradExpr(expr=inner_expr, wrt=inner_wrt):
                self._grad_depth += 1
                if self._grad_depth > _MAX_GRAD_DEPTH:
                    raise MaomiError(
                        f"grad: maximum nesting depth exceeded (limit: {_MAX_GRAD_DEPTH})",
                        "<ad>", expr.span.line_start, expr.span.col_start,
                    )
                saved_tape_exprs = self._tape_exprs
                try:
                    inner_result = self._differentiate(inner_expr, inner_wrt, expr, let_env)
                finally:
                    self._tape_exprs = saved_tape_exprs
                    self._grad_depth -= 1
                self._linearize(inner_result, tape, var_map, let_env)
                var_map[id(expr)] = var_map[id(inner_result)]
            case _:
                raise MaomiError(
                    f"grad: unsupported expression type {type(expr).__name__} inside grad",
                    "<ad>", expr.span.line_start, expr.span.col_start,
                )

    def _linearize_user_call(self, call_expr: CallExpr, callee: str,
                              args: list[Expr], tape: list[tuple[str, Expr]],
                              var_map: dict[int, str], let_env: dict[str, Expr]):
        """Inline a user function call and linearize the inlined body."""
        if callee in self._inlining_stack:
            raise MaomiError(
                f"grad: recursive function '{callee}' not supported inside grad",
                "<ad>", call_expr.span.line_start, call_expr.span.col_start,
            )

        fn_def = self.fn_defs[callee]
        # Build substitution: param_name -> call arg expression
        subst = {}
        for param, arg in zip(fn_def.params, args):
            subst[param.name] = arg

        # Expand let bindings in the function body into the substitution
        body = fn_def.body
        body_let_env = dict(let_env)
        for stmt in body.stmts:
            if isinstance(stmt, LetStmt):
                expanded = self._substitute(stmt.value, subst)
                subst[stmt.name] = expanded
                body_let_env[stmt.name] = expanded

        # Substitute into the trailing expression
        if body.expr is None:
            raise MaomiError(
                f"grad: function '{callee}' has no return expression",
                "<ad>", call_expr.span.line_start, call_expr.span.col_start,
            )

        inlined = self._substitute(body.expr, subst)
        # Copy the type from the call to the inlined expression
        call_type = self._type_of(call_expr)
        if call_type is not None:
            self.type_map[id(inlined)] = call_type

        self._inlining_stack.add(callee)
        try:
            self._linearize(inlined, tape, var_map, body_let_env)
        finally:
            self._inlining_stack.discard(callee)

        var_map[id(call_expr)] = var_map[id(inlined)]

    def _substitute(self, expr: Expr, subst: dict[str, Expr]) -> Expr:
        """Deep-copy an expression, replacing identifiers per subst map."""
        match expr:
            case Identifier(name=name):
                if name in subst:
                    return subst[name]
                return expr
            case IntLiteral() | FloatLiteral() | BoolLiteral():
                return expr
            case UnaryOp(op=op, operand=operand):
                new_op = self._substitute(operand, subst)
                node = UnaryOp(op, new_op, expr.span)
                self._copy_type(expr, node)
                return node
            case BinOp(op=op, left=left, right=right):
                new_left = self._substitute(left, subst)
                new_right = self._substitute(right, subst)
                node = BinOp(op, new_left, new_right, expr.span)
                self._copy_type(expr, node)
                return node
            case CallExpr(callee=callee, args=args):
                new_args = [self._substitute(a, subst) for a in args]
                node = CallExpr(callee, new_args, expr.span)
                self._copy_type(expr, node)
                return node
            case IfExpr(condition=cond, then_block=then_b, else_block=else_b):
                new_cond = self._substitute(cond, subst)
                new_then = self._substitute_block(then_b, subst)
                new_else = self._substitute_block(else_b, subst)
                node = IfExpr(new_cond, new_then, new_else, expr.span)
                self._copy_type(expr, node)
                return node
            case MapExpr(elem_var=ev, sequence=seq, body=body):
                new_seq = self._substitute(seq, subst)
                # Don't substitute elem_var — it's a fresh binding
                inner_subst = {k: v for k, v in subst.items() if k != ev}
                new_body = self._substitute_block(body, inner_subst)
                node = MapExpr(ev, new_seq, new_body, expr.span)
                self._copy_type(expr, node)
                return node
            case CastExpr(expr=inner, target_type=target):
                new_inner = self._substitute(inner, subst)
                node = CastExpr(new_inner, target, expr.span)
                self._copy_type(expr, node)
                return node
            case ScanExpr(carry_var=cv, elem_vars=evs, init=init, sequences=seqs, body=body):
                new_init = self._substitute(init, subst)
                new_seqs = [self._substitute(s, subst) for s in seqs]
                # carry_var and elem_vars are fresh bindings
                inner_subst = {k: v for k, v in subst.items() if k != cv and k not in evs}
                new_body = self._substitute_block(body, inner_subst)
                node = ScanExpr(cv, evs, new_init, new_seqs, new_body, expr.span, expr.reverse)
                self._copy_type(expr, node)
                return node
            case FoldExpr(carry_var=cv, elem_vars=evs, init=init, sequences=seqs, body=body):
                new_init = self._substitute(init, subst)
                new_seqs = [self._substitute(s, subst) for s in seqs]
                inner_subst = {k: v for k, v in subst.items() if k != cv and k not in evs}
                new_body = self._substitute_block(body, inner_subst)
                node = FoldExpr(cv, evs, new_init, new_seqs, new_body, expr.span)
                self._copy_type(expr, node)
                return node
            case WhileExpr(state_var=sv, init=init, cond=cond, body=body):
                new_init = self._substitute(init, subst)
                inner_subst = {k: v for k, v in subst.items() if k != sv}
                new_cond = self._substitute_block(cond, inner_subst)
                new_body = self._substitute_block(body, inner_subst)
                node = WhileExpr(sv, new_init, expr.max_iters, new_cond, new_body, expr.span)
                self._copy_type(expr, node)
                return node
            case ArrayLiteral(elements=elems):
                new_elems = [self._substitute(e, subst) for e in elems]
                node = ArrayLiteral(new_elems, expr.span)
                self._copy_type(expr, node)
                return node
            case StructLiteral(name=name, fields=fields):
                new_fields = [(fn, self._substitute(fv, subst)) for fn, fv in fields]
                node = StructLiteral(name, new_fields, expr.span)
                self._copy_type(expr, node)
                return node
            case FieldAccess(object=obj, field=field):
                new_obj = self._substitute(obj, subst)
                node = FieldAccess(new_obj, field, expr.span)
                self._copy_type(expr, node)
                return node
            case WithExpr(base=base, updates=updates):
                new_base = self._substitute(base, subst)
                new_updates = [(path, self._substitute(ve, subst)) for path, ve in updates]
                node = WithExpr(new_base, new_updates, expr.span)
                self._copy_type(expr, node)
                return node
            case IndexExpr(base=base, indices=indices):
                new_base = self._substitute(base, subst)
                new_indices = [self._substitute_index_component(ic, subst) for ic in indices]
                node = IndexExpr(new_base, new_indices, expr.span)
                self._copy_type(expr, node)
                return node
            case GradExpr(expr=inner_expr, wrt=wrt):
                new_inner = self._substitute(inner_expr, subst)
                node = GradExpr(new_inner, wrt, expr.span)
                self._copy_type(expr, node)
                return node
            case _Conv2dGrad():
                new_input = self._substitute(expr.input_expr, subst)
                new_kernel = self._substitute(expr.kernel_expr, subst)
                new_adj = self._substitute(expr.adj, subst)
                node = _Conv2dGrad(new_input, new_kernel, new_adj, expr.wrt, expr.strides, expr.padding, expr.span)
                self._copy_type(expr, node)
                return node
            case _MaxPoolGrad():
                new_input = self._substitute(expr.input_expr, subst)
                new_adj = self._substitute(expr.adj, subst)
                node = _MaxPoolGrad(new_input, new_adj, expr.window, expr.strides, expr.span)
                self._copy_type(expr, node)
                return node
            case _AvgPoolGrad():
                new_input = self._substitute(expr.input_expr, subst)
                new_adj = self._substitute(expr.adj, subst)
                node = _AvgPoolGrad(new_input, new_adj, expr.window, expr.strides, expr.span)
                self._copy_type(expr, node)
                return node
            case _FoldGrad():
                new_dc = self._substitute(expr.d_body_d_carry, subst)
                new_des = [self._substitute(de, subst) for de in expr.d_body_d_elems]
                new_init = self._substitute(expr.init, subst)
                new_seqs = [self._substitute(s, subst) for s in expr.sequences]
                new_adj = self._substitute(expr.adj, subst)
                inner_subst = {k: v for k, v in subst.items() if k != expr.carry_var and k not in expr.elem_vars}
                new_body = self._substitute_block(expr.body, inner_subst)
                node = _FoldGrad(new_dc, new_des, expr.carry_var, expr.elem_vars,
                                 new_init, new_seqs, new_body, new_adj, expr.wrt, expr.span)
                self._copy_type(expr, node)
                return node
            case _BroadcastExpr():
                new_expr = self._substitute(expr.expr, subst)
                node = _BroadcastExpr(new_expr, expr.target_dims, expr.span, broadcast_dims=expr.broadcast_dims)
                self._copy_type(expr, node)
                return node
            case _IndexGrad():
                new_base = self._substitute(expr.base_expr, subst)
                new_adj = self._substitute(expr.adj, subst)
                new_indices = [self._substitute_index_component(ic, subst) for ic in expr.indices]
                node = _IndexGrad(new_base, new_adj, new_indices, expr.span)
                self._copy_type(expr, node)
                return node
            case _GatherGrad():
                new_base = self._substitute(expr.base_expr, subst)
                new_adj = self._substitute(expr.adj, subst)
                new_indices = self._substitute(expr.indices, subst)
                node = _GatherGrad(new_base, new_adj, new_indices, expr.gather_axis, expr.span)
                self._copy_type(expr, node)
                return node
            case _ScanGrad():
                new_dbc = self._substitute(expr.d_body_d_carry, subst)
                new_dbe = [self._substitute(de, subst) for de in expr.d_body_d_elems]
                new_init = self._substitute(expr.init, subst)
                new_seqs = [self._substitute(s, subst) for s in expr.sequences]
                new_fwd = self._substitute(expr.forward_result, subst)
                new_adj = self._substitute(expr.adj, subst)
                node = _ScanGrad(new_dbc, new_dbe, expr.carry_var, expr.elem_vars,
                                 new_init, new_seqs, new_fwd, new_adj, expr.wrt, expr.span)
                self._copy_type(expr, node)
                return node
            case _WhileGrad():
                new_dbs = self._substitute(expr.d_body_d_state, subst)
                new_init = self._substitute(expr.init, subst)
                new_fwd = self._substitute(expr.forward_result, subst)
                new_adj = self._substitute(expr.adj, subst)
                node = _WhileGrad(new_dbs, expr.state_var, new_init, expr.max_iters,
                                  expr.cond, expr.body, new_fwd, new_adj, expr.span)
                self._copy_type(expr, node)
                return node
            case _ReduceSum():
                new_expr = self._substitute(expr.expr, subst)
                node = _ReduceSum(new_expr, expr.axes, expr.span)
                self._copy_type(expr, node)
                return node
            case _:
                return expr

    def _substitute_index_component(self, ic: IndexComponent, subst: dict[str, Expr]) -> IndexComponent:
        new_value = self._substitute(ic.value, subst) if ic.value is not None else None
        new_start = self._substitute(ic.start, subst) if ic.start is not None else None
        new_end = self._substitute(ic.end, subst) if ic.end is not None else None
        return IndexComponent(ic.kind, new_value, new_start, new_end, ic.span)

    def _substitute_block(self, block: Block, subst: dict[str, Expr]) -> Block:
        """Substitute through a block, respecting let binding scopes."""
        new_subst = dict(subst)
        new_stmts = []
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                new_val = self._substitute(stmt.value, new_subst)
                # Let binding shadows — remove from subst
                new_subst.pop(stmt.name, None)
                new_stmts.append(LetStmt(stmt.name, stmt.type_annotation, new_val, stmt.span))
            elif isinstance(stmt, ExprStmt):
                new_stmts.append(ExprStmt(self._substitute(stmt.expr, new_subst), stmt.span))
        new_expr = None
        if block.expr is not None:
            new_expr = self._substitute(block.expr, new_subst)
        return Block(new_stmts, new_expr, block.span)

    def _backprop(self, name: str, node: Expr, adj: Expr,
                  adjoints: dict[str, Expr], var_map: dict[int, str]):
        """Apply adjoint rules for a single tape node."""
        match node:
            case IntLiteral() | FloatLiteral() | BoolLiteral():
                pass  # constants have no inputs to propagate to

            case UnaryOp(op="-", operand=operand):
                op_name = var_map[id(operand)]
                neg_adj = self._make_unary("-", adj)
                self._accumulate(adjoints, op_name, neg_adj)

            case BinOp(op="+", left=left, right=right):
                l_name = var_map[id(left)]
                r_name = var_map[id(right)]
                self._accumulate(adjoints, l_name, adj)
                self._accumulate(adjoints, r_name, adj)

            case BinOp(op="-", left=left, right=right):
                l_name = var_map[id(left)]
                r_name = var_map[id(right)]
                self._accumulate(adjoints, l_name, adj)
                self._accumulate(adjoints, r_name, self._make_unary("-", adj))

            case BinOp(op="*", left=left, right=right):
                l_name = var_map[id(left)]
                r_name = var_map[id(right)]
                # dx = dz * y, dy = dz * x
                l_ref = self._make_ref(l_name, self._type_of(left))
                r_ref = self._make_ref(r_name, self._type_of(right))
                self._accumulate(adjoints, l_name, self._make_binop("*", adj, r_ref))
                self._accumulate(adjoints, r_name, self._make_binop("*", adj, l_ref))

            case BinOp(op="/", left=left, right=right):
                l_name = var_map[id(left)]
                r_name = var_map[id(right)]
                l_ref = self._make_ref(l_name, self._type_of(left))
                r_ref = self._make_ref(r_name, self._type_of(right))
                # dx = dz / y
                self._accumulate(adjoints, l_name, self._make_binop("/", adj, r_ref))
                # dy = -dz * x / (y * y)
                neg_adj = self._make_unary("-", adj)
                num = self._make_binop("*", neg_adj, l_ref)
                denom = self._make_binop("*", r_ref, r_ref)
                self._accumulate(adjoints, r_name, self._make_binop("/", num, denom))

            case BinOp(op="**", left=left, right=right):
                l_name = var_map[id(left)]
                l_ref = self._make_ref(l_name, self._type_of(left))
                # dx = dz * n * x^(n-1)
                n_minus_1 = self._make_binop("-", right, self._make_float(1.0))
                x_pow = self._make_binop("**", l_ref, n_minus_1)
                dx = self._make_binop("*", self._make_binop("*", adj, right), x_pow)
                self._accumulate(adjoints, l_name, dx)

            case BinOp(op="@", left=left, right=right):
                l_name = var_map[id(left)]
                r_name = var_map[id(right)]
                l_ref = self._make_ref(l_name, self._type_of(left))
                r_ref = self._make_ref(r_name, self._type_of(right))
                lt = self._type_of(left)
                rt = self._type_of(right)
                l_is_1d = isinstance(lt, ArrayType) and len(lt.dims) == 1
                r_is_1d = isinstance(rt, ArrayType) and len(rt.dims) == 1

                # dL/dA = dL/dC @ B^T
                if r_is_1d:
                    # B is 1D: reshape adj and B for outer product, then extract
                    r_col = self._make_call("reshape", [r_ref, IntLiteral(rt.dims[0], _DUMMY_SPAN), IntLiteral(1, _DUMMY_SPAN)])
                    adj_row = self._make_call("reshape", [adj, IntLiteral(1, _DUMMY_SPAN), IntLiteral(rt.dims[0], _DUMMY_SPAN)])
                    self._accumulate(adjoints, l_name, self._make_binop("@", adj_row, r_col))
                else:
                    r_transposed = self._make_call("transpose", [r_ref])
                    self._accumulate(adjoints, l_name, self._make_binop("@", adj, r_transposed))

                # dL/dB = A^T @ dL/dC
                if l_is_1d:
                    # A is 1D [D]: reshape to [D,1], adj is 1D [K] reshape to [1,K]
                    adj_t = self._type_of(adj)
                    adj_k = adj_t.dims[0] if isinstance(adj_t, ArrayType) else 1
                    l_col = self._make_call("reshape", [l_ref, IntLiteral(lt.dims[0], _DUMMY_SPAN), IntLiteral(1, _DUMMY_SPAN)])
                    adj_row = self._make_call("reshape", [adj, IntLiteral(1, _DUMMY_SPAN), IntLiteral(adj_k, _DUMMY_SPAN)])
                    self._accumulate(adjoints, r_name, self._make_binop("@", l_col, adj_row))
                else:
                    l_transposed = self._make_call("transpose", [l_ref])
                    self._accumulate(adjoints, r_name, self._make_binop("@", l_transposed, adj))

            case CallExpr(callee=callee, args=args):
                if callee in _IOTA_BUILTINS:
                    pass  # iota produces integers — no gradient flows through
                elif callee in _RNG_BUILTINS:
                    pass  # RNG has zero gradient — non-differentiable
                elif callee in _ELEMENTWISE_BUILTINS:
                    self._backprop_elementwise(callee, args, adj, adjoints, var_map, node)
                elif callee in _REDUCTION_BUILTINS:
                    self._backprop_reduction(callee, args, adj, adjoints, var_map, node)
                elif callee == "reshape":
                    self._backprop_reshape(args, adj, adjoints, var_map)
                elif callee == "concat":
                    self._backprop_concat(args, adj, adjoints, var_map)
                elif callee == "transpose":
                    # transpose is self-adjoint: d/dx transpose(x) = transpose(adj)
                    arg = args[0]
                    arg_name = var_map[id(arg)]
                    self._accumulate(adjoints, arg_name, self._make_call("transpose", [adj]))
                elif callee in _CONV_POOL_BUILTINS:
                    self._backprop_conv_pool(callee, args, adj, adjoints, var_map, node)
                elif callee in _STOP_GRAD_BUILTINS:
                    pass  # gradient stops here — don't accumulate any adjoint
                elif callee in _WHERE_BUILTINS:
                    self._backprop_where(args, adj, adjoints, var_map, node)
                elif callee in _ARGMAX_BUILTINS:
                    pass  # non-differentiable (returns i32 indices)
                else:
                    raise MaomiError(
                        f"grad: unsupported function call '{callee}' inside grad",
                        "<ad>", node.span.line_start, node.span.col_start,
                    )

            case IfExpr(condition=cond, then_block=then_b, else_block=else_b):
                self._backprop_if(cond, then_b, else_b, adj, adjoints, var_map, node)

            case MapExpr(elem_var=elem_var, sequence=seq, body=body):
                self._backprop_map(elem_var, seq, body, adj, adjoints, var_map, node)

            case ScanExpr(carry_var=cv, elem_vars=evs, init=init, sequences=seqs, body=body):
                self._backprop_scan(cv, evs, init, seqs, body, adj, adjoints, var_map, node)

            case WhileExpr(state_var=sv, init=init, max_iters=mi, cond=cond, body=body):
                self._backprop_while(sv, init, mi, cond, body, adj, adjoints, var_map, node)

            case CastExpr(expr=inner, target_type=target):
                inner_name = var_map[id(inner)]
                inner_type = self._type_of(inner)
                inner_base = None
                if isinstance(inner_type, ScalarType):
                    inner_base = inner_type.base
                elif isinstance(inner_type, ArrayType):
                    inner_base = inner_type.base
                if inner_base in ("f32", "f64") and target in ("f32", "f64"):
                    cast_back = CastExpr(adj, inner_base, _DUMMY_SPAN)
                    self.type_map[id(cast_back)] = inner_type
                    self._accumulate(adjoints, inner_name, cast_back)
                # else: non-differentiable conversion (to int/bool), zero gradient

            case FoldExpr(carry_var=cv, elem_vars=evs, init=init, sequences=seqs, body=body):
                self._backprop_fold(cv, evs, init, seqs, body, adj, adjoints, var_map, node)

            case FieldAccess(object=obj, field=field):
                # adj of obj += struct with only 'field' set to adj
                obj_name = var_map[id(obj)]
                obj_type = self._type_of(obj)
                if isinstance(obj_type, StructType):
                    partial = self._make_struct_with_field(obj_type, field, adj)
                    self._accumulate(adjoints, obj_name, partial)

            case ArrayLiteral(elements=elems):
                # adj of element i = adj[i]
                for i, e in enumerate(elems):
                    if id(e) in var_map:
                        e_name = var_map[id(e)]
                        idx_node = IntLiteral(i, _DUMMY_SPAN)
                        self.type_map[id(idx_node)] = ScalarType("i32")
                        ic = IndexComponent(kind="single", value=idx_node, start=None, end=None, span=_DUMMY_SPAN)
                        elem_adj = IndexExpr(adj, [ic], _DUMMY_SPAN)
                        et = self._type_of(e)
                        if et is not None:
                            self.type_map[id(elem_adj)] = et
                        self._accumulate(adjoints, e_name, elem_adj)

            case StructLiteral(name=sname, fields=fields):
                # adj of each field value = extract that field from adj
                stype = self._type_of(node)
                if isinstance(stype, StructType):
                    for i, (fn, fv) in enumerate(fields):
                        if id(fv) in var_map:
                            fv_name = var_map[id(fv)]
                            field_adj = FieldAccess(adj, fn, _DUMMY_SPAN)
                            ft = dict(stype.fields)[fn]
                            self.type_map[id(field_adj)] = ft
                            self._accumulate(adjoints, fv_name, field_adj)

            case WithExpr():
                pass  # WithExpr gradients are complex; skip for now

            case IndexExpr(base=base, indices=indices):
                base_name = var_map[id(base)]
                base_type = self._type_of(base)

                # Check if any index is an array (gather case)
                gather_axis = None
                gather_indices = None
                for ic_i, ic in enumerate(indices):
                    if ic.kind == "single":
                        idx_type = self._type_of(ic.value)
                        if isinstance(idx_type, ArrayType):
                            gather_axis = ic_i
                            gather_indices = ic.value
                            break

                if gather_axis is not None:
                    grad_node = _GatherGrad(base, adj, gather_indices, gather_axis, _DUMMY_SPAN)
                else:
                    grad_node = _IndexGrad(base, adj, indices, _DUMMY_SPAN)

                if base_type is not None:
                    self.type_map[id(grad_node)] = base_type
                self._accumulate(adjoints, base_name, grad_node)

            # -- Grad-of-grad: backprop through internal gradient nodes --

            case _IndexGrad(base_expr=_base, adj=adj_inner, indices=indices):
                # _IndexGrad(base, adj_inner, indices) = dynamic_update_slice(zeros, adj_inner, indices)
                # Linear in adj_inner. d/d(adj_inner) = upstream[indices] (an IndexExpr).
                if id(adj_inner) in var_map:
                    adj_inner_name = var_map[id(adj_inner)]
                    grad_adj = IndexExpr(adj, indices, _DUMMY_SPAN)
                    adj_inner_type = self._type_of(adj_inner)
                    if adj_inner_type is not None:
                        self.type_map[id(grad_adj)] = adj_inner_type
                    self._accumulate(adjoints, adj_inner_name, grad_adj)

            case _GatherGrad(base_expr=_base, adj=adj_inner, indices=idx, gather_axis=axis):
                # _GatherGrad(base, adj_inner, idx, axis) = scatter_add(zeros, idx, adj_inner)
                # Linear in adj_inner. d/d(adj_inner) = gather(upstream, idx, axis).
                if id(adj_inner) in var_map:
                    adj_inner_name = var_map[id(adj_inner)]
                    adj_type = self._type_of(adj)
                    if isinstance(adj_type, ArrayType):
                        ndim = len(adj_type.dims)
                        ics = []
                        for i in range(ndim):
                            if i == axis:
                                ics.append(IndexComponent("single", idx, None, None, _DUMMY_SPAN))
                            else:
                                ics.append(IndexComponent("full", None, None, None, _DUMMY_SPAN))
                        grad_adj = IndexExpr(adj, ics, _DUMMY_SPAN)
                        adj_inner_type = self._type_of(adj_inner)
                        if adj_inner_type is not None:
                            self.type_map[id(grad_adj)] = adj_inner_type
                        self._accumulate(adjoints, adj_inner_name, grad_adj)

            case _BroadcastExpr(expr=inner):
                # _BroadcastExpr(inner, target_dims) broadcasts inner to target_dims.
                # d/d(inner) = reduce_sum(upstream) over broadcast dimensions.
                if id(inner) in var_map:
                    inner_name = var_map[id(inner)]
                    inner_type = self._type_of(inner)
                    if isinstance(inner_type, ScalarType):
                        # Scalar was broadcast — gradient is sum of all upstream elements
                        grad_inner = self._make_call("sum", [adj])
                        self.type_map[id(grad_inner)] = inner_type
                        self._accumulate(adjoints, inner_name, grad_inner)
                    elif isinstance(inner_type, ArrayType):
                        # Lower-rank broadcast — reduce over the broadcast axes
                        target_ndim = len(node.target_dims)
                        inner_ndim = len(inner_type.dims)
                        if node.broadcast_dims is not None:
                            mapped = set(node.broadcast_dims)
                        else:
                            mapped = set(range(target_ndim - inner_ndim, target_ndim))
                        reduce_axes = tuple(i for i in range(target_ndim) if i not in mapped)
                        if reduce_axes:
                            grad_inner = _ReduceSum(adj, reduce_axes, _DUMMY_SPAN)
                            self.type_map[id(grad_inner)] = inner_type
                            self._accumulate(adjoints, inner_name, grad_inner)
                        else:
                            self._accumulate(adjoints, inner_name, adj)

            case _ReduceSum(expr=inner):
                # _ReduceSum(inner, axes) reduces inner by summing over axes.
                # d/d(inner) = broadcast(upstream) back to inner's shape along reduced axes.
                if id(inner) in var_map:
                    inner_name = var_map[id(inner)]
                    inner_type = self._type_of(inner)
                    if isinstance(inner_type, ArrayType):
                        # Broadcast upstream back to inner's shape
                        broadcast_dims = tuple(i for i in range(len(inner_type.dims)) if i not in node.axes)
                        grad_inner = _BroadcastExpr(adj, tuple(inner_type.dims), _DUMMY_SPAN, broadcast_dims=broadcast_dims)
                        self.type_map[id(grad_inner)] = inner_type
                        self._accumulate(adjoints, inner_name, grad_inner)

            case _ScanGrad():
                raise MaomiError(
                    "grad-of-grad through scan with non-constant body derivatives is not yet supported. "
                    "Use scans with linear bodies (e.g. carry + x) for grad-of-grad.",
                    "<ad>", node.span.line_start, node.span.col_start,
                )

            case _WhileGrad():
                raise MaomiError(
                    "grad-of-grad through while loops is not yet supported.",
                    "<ad>", node.span.line_start, node.span.col_start,
                )

            case _Conv2dGrad():
                raise MaomiError(
                    "grad-of-grad through conv2d is not yet supported.",
                    "<ad>", node.span.line_start, node.span.col_start,
                )

            case _MaxPoolGrad():
                raise MaomiError(
                    "grad-of-grad through max_pool is not yet supported.",
                    "<ad>", node.span.line_start, node.span.col_start,
                )

            case _AvgPoolGrad():
                raise MaomiError(
                    "grad-of-grad through avg_pool is not yet supported.",
                    "<ad>", node.span.line_start, node.span.col_start,
                )

            case _:
                raise MaomiError(
                    f"grad: unsupported expression in backward pass: {type(node).__name__}",
                    "<ad>", node.span.line_start, node.span.col_start,
                )

    def _backprop_elementwise(self, callee: str, args: list[Expr], adj: Expr,
                               adjoints: dict[str, Expr], var_map: dict[int, str],
                               node: Expr):
        arg = args[0]
        arg_name = var_map[id(arg)]
        arg_ref = self._make_ref(arg_name, self._type_of(arg))

        if callee == "exp":
            # d/dx exp(x) = exp(x) * dz
            exp_x = self._make_call("exp", [arg_ref])
            self._accumulate(adjoints, arg_name, self._make_binop("*", adj, exp_x))

        elif callee == "log":
            # d/dx log(x) = dz / x
            self._accumulate(adjoints, arg_name, self._make_binop("/", adj, arg_ref))

        elif callee == "tanh":
            # d/dx tanh(x) = dz * (1 - tanh(x)^2)
            tanh_x = self._make_call("tanh", [arg_ref])
            tanh_sq = self._make_binop("*", tanh_x, tanh_x)
            one_minus = self._make_binop("-", self._make_float(1.0), tanh_sq)
            self._accumulate(adjoints, arg_name, self._make_binop("*", adj, one_minus))

        elif callee == "sqrt":
            # d/dx sqrt(x) = dz / (2 * sqrt(x))
            sqrt_x = self._make_call("sqrt", [arg_ref])
            two_sqrt = self._make_binop("*", self._make_float(2.0), sqrt_x)
            self._accumulate(adjoints, arg_name, self._make_binop("/", adj, two_sqrt))

        elif callee == "abs":
            # d/dx |x| = dz * sign(x) — approximate as x / |x|
            abs_x = self._make_call("abs", [arg_ref])
            sign = self._make_binop("/", arg_ref, abs_x)
            self._accumulate(adjoints, arg_name, self._make_binop("*", adj, sign))

    def _backprop_reduction(self, callee: str, args: list[Expr], adj: Expr,
                             adjoints: dict[str, Expr], var_map: dict[int, str],
                             node: Expr):
        arg = args[0]
        arg_name = var_map[id(arg)]
        arg_type = self._type_of(arg)

        has_axis = len(args) == 2
        axis = args[1].value if has_axis else None

        if callee == "mean":
            if isinstance(arg_type, ArrayType):
                if has_axis:
                    # Axis-specific mean: adj / axis_size, broadcast back with explicit dims
                    axis_size = arg_type.dims[axis]
                    if isinstance(axis_size, str):
                        raise MaomiError(f"grad: cannot differentiate mean with symbolic dim '{axis_size}'", "<ad>", 0, 0)
                    scaled = self._make_binop("/", adj, self._make_float(float(axis_size)))
                    ndim = len(arg_type.dims)
                    broadcast_dims = tuple(i for i in range(ndim) if i != axis)
                    broadcast = _BroadcastExpr(scaled, tuple(arg_type.dims), _DUMMY_SPAN, broadcast_dims=broadcast_dims)
                    self.type_map[id(broadcast)] = arg_type
                    self._accumulate(adjoints, arg_name, broadcast)
                else:
                    # All-dims mean: broadcast(dz / numel)
                    numel = 1
                    for d in arg_type.dims:
                        if isinstance(d, int):
                            numel *= d
                        else:
                            raise MaomiError(f"grad: cannot differentiate mean with symbolic dim '{d}'", "<ad>", 0, 0)
                    scaled = self._make_binop("/", adj, self._make_float(float(numel)))
                    broadcast = _BroadcastExpr(scaled, tuple(arg_type.dims), _DUMMY_SPAN)
                    self.type_map[id(broadcast)] = arg_type
                    self._accumulate(adjoints, arg_name, broadcast)
            else:
                self._accumulate(adjoints, arg_name, adj)

        elif callee == "sum":
            if isinstance(arg_type, ArrayType):
                if has_axis:
                    # Axis-specific sum: broadcast adj back by inserting reduced dim
                    ndim = len(arg_type.dims)
                    broadcast_dims = tuple(i for i in range(ndim) if i != axis)
                    broadcast = _BroadcastExpr(adj, tuple(arg_type.dims), _DUMMY_SPAN, broadcast_dims=broadcast_dims)
                    self.type_map[id(broadcast)] = arg_type
                    self._accumulate(adjoints, arg_name, broadcast)
                else:
                    # All-dims sum: broadcast scalar adj to input shape
                    broadcast = _BroadcastExpr(adj, tuple(arg_type.dims), _DUMMY_SPAN)
                    self.type_map[id(broadcast)] = arg_type
                    self._accumulate(adjoints, arg_name, broadcast)
            else:
                self._accumulate(adjoints, arg_name, adj)

        elif callee in ("max", "min"):
            if not isinstance(arg_type, ArrayType):
                self._accumulate(adjoints, arg_name, adj)
                return

            # JAX indicator rule: grad = adj_bc * indicators / counts_bc
            # indicators = cast(operand == broadcast(result), f32)
            # counts = sum(indicators, axes)
            tape_name = var_map.get(id(node))
            node_ref = self._make_ref(tape_name, self._type_of(node)) if tape_name else node
            arg_ref = self._make_ref(arg_name, arg_type)
            ndim = len(arg_type.dims)

            # Broadcast result back to input shape
            if has_axis:
                broadcast_dims = tuple(i for i in range(ndim) if i != axis)
                result_bc = _BroadcastExpr(node_ref, tuple(arg_type.dims), _DUMMY_SPAN, broadcast_dims=broadcast_dims)
            else:
                result_bc = _BroadcastExpr(node_ref, tuple(arg_type.dims), _DUMMY_SPAN)
            self.type_map[id(result_bc)] = arg_type

            # indicators = cast(arg == result_bc, arg_type.base)
            eq = BinOp("==", arg_ref, result_bc, _DUMMY_SPAN)
            bool_type = ArrayType("bool", arg_type.dims)
            self.type_map[id(eq)] = bool_type
            indicators = CastExpr(eq, arg_type.base, _DUMMY_SPAN)
            self.type_map[id(indicators)] = arg_type

            # counts = sum(indicators) or sum(indicators, axis)
            if has_axis:
                axis_lit = IntLiteral(axis, _DUMMY_SPAN)
                self.type_map[id(axis_lit)] = ScalarType("i32")
                counts = CallExpr("sum", [indicators, axis_lit], _DUMMY_SPAN)
                count_type = self._type_of(node)  # same shape as reduced result
            else:
                counts = CallExpr("sum", [indicators], _DUMMY_SPAN)
                count_type = ScalarType(arg_type.base)
            self.type_map[id(counts)] = count_type

            # Broadcast adj and counts back to input shape
            if has_axis:
                adj_bc = _BroadcastExpr(adj, tuple(arg_type.dims), _DUMMY_SPAN, broadcast_dims=broadcast_dims)
                counts_bc = _BroadcastExpr(counts, tuple(arg_type.dims), _DUMMY_SPAN, broadcast_dims=broadcast_dims)
            else:
                adj_bc = _BroadcastExpr(adj, tuple(arg_type.dims), _DUMMY_SPAN)
                counts_bc = _BroadcastExpr(counts, tuple(arg_type.dims), _DUMMY_SPAN)
            self.type_map[id(adj_bc)] = arg_type
            self.type_map[id(counts_bc)] = arg_type

            # grad = adj_bc * indicators / counts_bc
            grad = BinOp("*", adj_bc, indicators, _DUMMY_SPAN)
            self.type_map[id(grad)] = arg_type
            grad = BinOp("/", grad, counts_bc, _DUMMY_SPAN)
            self.type_map[id(grad)] = arg_type
            self._accumulate(adjoints, arg_name, grad)

    def _backprop_where(self, args: list[Expr], adj: Expr,
                         adjoints: dict[str, Expr], var_map: dict[int, str],
                         node: Expr):
        """Backprop through where(cond, x, y): adj_x = where(cond, adj, 0), adj_y = where(cond, 0, adj)."""
        cond = args[0]
        x_arg = args[1]
        y_arg = args[2]

        # cond has no gradient (boolean)
        # For x: gradient flows where cond is true
        if id(x_arg) in var_map:
            x_name = var_map[id(x_arg)]
            x_type = self._type_of(x_arg)
            zero = self._make_float(0.0)
            if isinstance(x_type, ArrayType):
                zero_broadcast = _BroadcastExpr(zero, tuple(x_type.dims), _DUMMY_SPAN)
                self.type_map[id(zero_broadcast)] = x_type
                adj_x = self._make_call("where", [cond, adj, zero_broadcast])
            else:
                adj_x = self._make_call("where", [cond, adj, zero])
            self.type_map[id(adj_x)] = self._type_of(x_arg)
            self._accumulate(adjoints, x_name, adj_x)

        # For y: gradient flows where cond is false
        if id(y_arg) in var_map:
            y_name = var_map[id(y_arg)]
            y_type = self._type_of(y_arg)
            zero = self._make_float(0.0)
            if isinstance(y_type, ArrayType):
                zero_broadcast = _BroadcastExpr(zero, tuple(y_type.dims), _DUMMY_SPAN)
                self.type_map[id(zero_broadcast)] = y_type
                adj_y = self._make_call("where", [cond, zero_broadcast, adj])
            else:
                adj_y = self._make_call("where", [cond, zero, adj])
            self.type_map[id(adj_y)] = self._type_of(y_arg)
            self._accumulate(adjoints, y_name, adj_y)

    def _backprop_reshape(self, args: list[Expr], adj: Expr,
                           adjoints: dict[str, Expr], var_map: dict[int, str]):
        """Backprop through reshape: reshape adjoint back to original shape."""
        arg = args[0]
        if id(arg) not in var_map:
            return
        arg_name = var_map[id(arg)]
        arg_type = self._type_of(arg)
        if not isinstance(arg_type, ArrayType):
            self._accumulate(adjoints, arg_name, adj)
            return

        # Build reshape(adj, *original_dims)
        dim_literals = []
        for d in arg_type.dims:
            lit = IntLiteral(d, _DUMMY_SPAN)
            self.type_map[id(lit)] = ScalarType("i32")
            dim_literals.append(lit)
        reshape_call = CallExpr("reshape", [adj] + dim_literals, _DUMMY_SPAN)
        self.type_map[id(reshape_call)] = arg_type
        self._accumulate(adjoints, arg_name, reshape_call)

    def _backprop_concat(self, args: list[Expr], adj: Expr,
                          adjoints: dict[str, Expr], var_map: dict[int, str]):
        """Backprop through concat: slice adjoint into pieces for each input."""
        # Detect axis
        if (isinstance(args[-1], IntLiteral)
                and isinstance(self.type_map.get(id(args[-1])), ScalarType)):
            axis = args[-1].value
            array_args = args[:-1]
        else:
            axis = 0
            array_args = args

        adj_type = self._type_of(adj)

        # If adj is scalar (e.g. from sum backprop), broadcast handles it —
        # just accumulate the scalar adj to each input (codegen broadcasts).
        if not isinstance(adj_type, ArrayType):
            for arg in array_args:
                if id(arg) in var_map:
                    self._accumulate(adjoints, var_map[id(arg)], adj)
            return

        rank = len(adj_type.dims)

        offset = 0
        for arg in array_args:
            arg_type = self._type_of(arg)
            if not isinstance(arg_type, ArrayType):
                continue
            size = arg_type.dims[axis]
            if not isinstance(size, int):
                continue

            if id(arg) in var_map:
                arg_name = var_map[id(arg)]

                # Build IndexExpr: adj sliced along concat axis
                components = []
                for d in range(rank):
                    if d == axis:
                        start = IntLiteral(offset, _DUMMY_SPAN)
                        end = IntLiteral(offset + size, _DUMMY_SPAN)
                        self.type_map[id(start)] = ScalarType("i32")
                        self.type_map[id(end)] = ScalarType("i32")
                        components.append(IndexComponent("slice", None, start, end, _DUMMY_SPAN))
                    else:
                        components.append(IndexComponent("full", None, None, None, _DUMMY_SPAN))

                slice_expr = IndexExpr(adj, components, _DUMMY_SPAN)
                self.type_map[id(slice_expr)] = arg_type
                self._accumulate(adjoints, arg_name, slice_expr)

            offset += size

    def _backprop_conv_pool(self, callee: str, args: list[Expr], adj: Expr,
                             adjoints: dict[str, Expr], var_map: dict[int, str],
                             node: Expr):
        """Backprop through conv2d, max_pool, avg_pool."""
        if callee == "conv2d":
            # conv2d(input, kernel, ...) — bilinear
            input_expr = args[0]
            kernel_expr = args[1]
            input_type = self._type_of(input_expr)
            kernel_type = self._type_of(kernel_expr)
            assert isinstance(input_type, ArrayType) and isinstance(kernel_type, ArrayType)

            # Extract stride/pad from literal args
            nargs = len(args)
            if nargs == 2:
                sh, sw, ph, pw = 1, 1, 0, 0
            elif nargs == 4:
                sh = sw = args[2].value
                ph = pw = args[3].value
            else:
                sh, sw = args[2].value, args[3].value
                ph, pw = args[4].value, args[5].value

            strides = (sh, sw)
            padding = (ph, pw)

            # grad w.r.t. input
            if id(input_expr) in var_map:
                input_name = var_map[id(input_expr)]
                grad_node = _Conv2dGrad(
                    input_expr, kernel_expr, adj, "lhs", strides, padding, _DUMMY_SPAN,
                )
                self.type_map[id(grad_node)] = input_type
                self._accumulate(adjoints, input_name, grad_node)

            # grad w.r.t. kernel
            if id(kernel_expr) in var_map:
                kernel_name = var_map[id(kernel_expr)]
                grad_node = _Conv2dGrad(
                    input_expr, kernel_expr, adj, "rhs", strides, padding, _DUMMY_SPAN,
                )
                self.type_map[id(grad_node)] = kernel_type
                self._accumulate(adjoints, kernel_name, grad_node)

        elif callee == "max_pool":
            # max_pool(input, wh, ww, sh, sw)
            input_expr = args[0]
            input_type = self._type_of(input_expr)
            wh, ww = args[1].value, args[2].value
            sh, sw = args[3].value, args[4].value

            if id(input_expr) in var_map:
                input_name = var_map[id(input_expr)]
                grad_node = _MaxPoolGrad(
                    input_expr, adj, (wh, ww), (sh, sw), _DUMMY_SPAN,
                )
                self.type_map[id(grad_node)] = input_type
                self._accumulate(adjoints, input_name, grad_node)

        elif callee == "avg_pool":
            # avg_pool(input, wh, ww, sh, sw)
            input_expr = args[0]
            input_type = self._type_of(input_expr)
            wh, ww = args[1].value, args[2].value
            sh, sw = args[3].value, args[4].value

            if id(input_expr) in var_map:
                input_name = var_map[id(input_expr)]
                grad_node = _AvgPoolGrad(
                    input_expr, adj, (wh, ww), (sh, sw), _DUMMY_SPAN,
                )
                self.type_map[id(grad_node)] = input_type
                self._accumulate(adjoints, input_name, grad_node)

    def _backprop_if(self, cond: Expr, then_b: Block, else_b: Block,
                      adj: Expr, adjoints: dict[str, Expr],
                      var_map: dict[int, str], node: Expr):
        """Backprop through if/else: differentiate both branches, select with condition."""
        then_expr = then_b.expr
        else_expr = else_b.expr
        if then_expr is None or else_expr is None:
            return

        # Find all free variables referenced in both branches
        free_vars = _collect_free_vars(then_expr) | _collect_free_vars(else_expr)
        # Only propagate to variables that are on the tape
        tape_vars = set(var_map.values())

        for v_name in free_vars:
            if v_name not in tape_vars:
                continue

            # Differentiate each branch w.r.t. this variable
            then_grad = self._differentiate_branch(then_expr, v_name)
            else_grad = self._differentiate_branch(else_expr, v_name)

            # Build: if cond { then_grad } else { else_grad }
            then_block = Block([], then_grad, _DUMMY_SPAN)
            else_block = Block([], else_grad, _DUMMY_SPAN)
            grad_if = IfExpr(cond, then_block, else_block, _DUMMY_SPAN)

            # Type the if expression
            gt = self.type_map.get(id(then_grad))
            if gt is not None:
                self.type_map[id(grad_if)] = gt

            # contribution = adj * if cond { d_then } else { d_else }
            contribution = self._make_binop("*", adj, grad_if)
            self._accumulate(adjoints, v_name, contribution)

    def _differentiate_branch(self, expr: Expr, wrt: str) -> Expr:
        """Compute d(expr)/d(wrt) for a branch expression using a fresh tape."""
        saved_tape_exprs = self._tape_exprs

        tape: list[tuple[str, Expr]] = []
        var_map: dict[int, str] = {}
        self._linearize(expr, tape, var_map, {})

        if id(expr) not in var_map:
            self._tape_exprs = saved_tape_exprs
            return self._make_float(0.0)

        self._tape_exprs = {name: node for name, node in tape}

        output_name = var_map[id(expr)]
        adjoints: dict[str, Expr] = {output_name: self._make_float(1.0)}

        for name, node in reversed(tape):
            if name not in adjoints:
                continue
            adj = adjoints[name]
            self._backprop(name, node, adj, adjoints, var_map)

        self._tape_exprs = saved_tape_exprs

        if wrt in adjoints:
            return adjoints[wrt]
        return self._make_float(0.0)

    def _backprop_map(self, elem_var: str, seq: Expr, body: Block,
                       adj: Expr, adjoints: dict[str, Expr],
                       var_map: dict[int, str], node: Expr):
        """Backprop through map: adj_seq = map elem in seq { d(body)/d(elem) * adj_elem }."""
        seq_name = var_map[id(seq)]
        body_expr = body.expr
        if body_expr is None:
            return

        # Differentiate the body w.r.t. the element variable
        body_grad = self._differentiate_branch(body_expr, elem_var)

        # Build: map elem_var in seq { body_grad * adj_elem }
        # Since adj is the adjoint of the whole map output (an array),
        # and body_grad gives per-element derivative, we want:
        #   adj_seq = adj * map elem_var in seq { body_grad }
        # This works because both are same-shape arrays and * is elementwise.
        grad_map = MapExpr(elem_var, self._make_ref(seq_name, self._type_of(seq)),
                          Block([], body_grad, _DUMMY_SPAN), _DUMMY_SPAN)
        # Type the grad map — same type as the original map
        map_type = self._type_of(node)
        if map_type is not None:
            self.type_map[id(grad_map)] = map_type

        contribution = self._make_binop("*", adj, grad_map)
        self._accumulate(adjoints, seq_name, contribution)

        # Propagate to free variables in the body (other than elem_var)
        # For free var w: d(map x in xs { f(x, w) })/dw = sum_i(adj[i] * d(f(x_i, w))/dw)
        free_vars = _collect_free_vars(body_expr) - {elem_var}
        seq_type = self._type_of(seq)
        for v_name in free_vars:

            # Per-element gradient d(body)/d(w)
            grad_w = self._differentiate_branch(body_expr, v_name)

            # Wrap in map: map elem_var in seq { d(body)/d(w) }
            seq_ref = self._make_ref(seq_name, seq_type)
            grad_map_w = MapExpr(elem_var, seq_ref,
                                 Block([], grad_w, _DUMMY_SPAN), _DUMMY_SPAN)

            # Type the grad map: seq_len prepended to w's type
            w_type = self._type_of(grad_w)
            if isinstance(seq_type, ArrayType) and w_type is not None:
                seq_len = seq_type.dims[0]
                if isinstance(w_type, ScalarType):
                    grad_map_type = ArrayType(w_type.base, (seq_len,))
                elif isinstance(w_type, ArrayType):
                    grad_map_type = ArrayType(w_type.base, (seq_len,) + w_type.dims)
                else:
                    grad_map_type = w_type
                self.type_map[id(grad_map_w)] = grad_map_type
            else:
                grad_map_type = map_type

            # Scale by adjoint: adj * grad_map_w (broadcasts adj over non-batch dims)
            scaled = self._make_binop("*", adj, grad_map_w)

            # Reduce over dimension 0 (the map/batch dimension) to get w-shaped gradient
            reduced = _ReduceSum(expr=scaled, axes=(0,), span=_DUMMY_SPAN)
            if w_type is not None:
                self.type_map[id(reduced)] = w_type

            self._accumulate(adjoints, v_name, reduced)

    def _backprop_while(self, state_var: str, init: Expr,
                         max_iters: int | None, cond: Block, body: Block,
                         adj: Expr, adjoints: dict[str, Expr],
                         var_map: dict[int, str], node: Expr):
        """Backprop through while: emit _WhileGrad for bounded, error for unbounded."""
        if max_iters is None:
            raise MaomiError(
                "reverse-mode AD is not supported through while loops without a limit. "
                "Use 'while state in init limit N { cond } do { body }' for differentiable while loops, "
                "or use scan for fixed-iteration differentiable loops.",
                "<ad>", node.span.line_start, node.span.col_start,
            )

        body_expr = body.expr
        if body_expr is None:
            return

        # Compute symbolic derivative of body w.r.t. state
        d_body_d_state = self._differentiate_branch(body_expr, state_var)

        fwd_ref = node

        if id(init) in var_map:
            init_name = var_map[id(init)]
            grad_node = _WhileGrad(
                d_body_d_state=d_body_d_state,
                state_var=state_var,
                init=init,
                max_iters=max_iters,
                cond=cond,
                body=body,
                forward_result=fwd_ref,
                adj=adj,
                span=node.span,
            )
            state_type = self._type_of(init)
            self.type_map[id(grad_node)] = state_type
            self._accumulate(adjoints, init_name, grad_node)

    def _backprop_scan(self, carry_var: str, elem_vars: list[str],
                        init: Expr, sequences: list[Expr], body: Block,
                        adj: Expr, adjoints: dict[str, Expr],
                        var_map: dict[int, str], node: Expr):
        """Backprop through scan: emit reverse ScanExpr or _ScanGrad nodes."""
        body_expr = body.expr
        if body_expr is None:
            return

        # Compute symbolic derivatives of body w.r.t. carry and each elem
        d_body_d_carry = self._differentiate_branch(body_expr, carry_var)
        d_body_d_elems = [self._differentiate_branch(body_expr, ev) for ev in elem_vars]

        # The forward result is the scan node itself — codegen will generate it
        fwd_ref = node

        # Check if derivatives are constant (don't reference carry/elem vars).
        # When constant, we can emit standard ScanExpr nodes (JAX-style)
        # that support grad-of-grad. Non-constant falls back to _ScanGrad.
        all_deriv_vars = _collect_free_vars(d_body_d_carry)
        for de in d_body_d_elems:
            all_deriv_vars |= _collect_free_vars(de)
        constant_derivs = not (all_deriv_vars & {carry_var, *elem_vars})

        # Propagate to init
        if id(init) in var_map:
            init_name = var_map[id(init)]
            init_type = self._type_of(init)
            if constant_derivs:
                # Build reverse scan then extract final carry via indexing.
                # The reverse scan accumulates: carry * d_carry + adj_elem
                # The final carry (last stacked output) is the init gradient.
                rev_scan = self._build_reverse_scan_grad(
                    d_body_d_carry, self._make_float(1.0), adj, sequences[0], node)
                # Get sequence length from type to index the last element
                seq_type = self._type_of(sequences[0])
                if isinstance(seq_type, ArrayType):
                    seq_len = seq_type.dims[0]
                    assert isinstance(seq_len, int)
                    last_idx = IntLiteral(seq_len - 1, _DUMMY_SPAN)
                    self.type_map[id(last_idx)] = ScalarType("i32")
                    ic = IndexComponent("single", last_idx, None, None, _DUMMY_SPAN)
                    grad_init = IndexExpr(rev_scan, [ic], _DUMMY_SPAN)
                    if init_type is not None:
                        self.type_map[id(grad_init)] = init_type
                    rev_scan_type = seq_type  # reverse scan stacks carries
                    self.type_map[id(rev_scan)] = rev_scan_type
                else:
                    grad_init = rev_scan
                    if init_type is not None:
                        self.type_map[id(grad_init)] = init_type
            else:
                # Fallback: _ScanGrad for non-constant derivatives
                grad_init = _ScanGrad(
                    d_body_d_carry=d_body_d_carry,
                    d_body_d_elems=d_body_d_elems,
                    carry_var=carry_var,
                    elem_vars=elem_vars,
                    init=init,
                    sequences=sequences,
                    forward_result=fwd_ref,
                    adj=adj,
                    wrt="__init__",
                    span=_DUMMY_SPAN,
                )
                if init_type is not None:
                    self.type_map[id(grad_init)] = init_type
            self._accumulate(adjoints, init_name, grad_init)

        # Propagate to each sequence
        for i, seq in enumerate(sequences):
            if id(seq) not in var_map:
                continue
            seq_name = var_map[id(seq)]

            if constant_derivs:
                # JAX-style: backward pass is just a reverse scan
                grad_seq = self._build_reverse_scan_grad(
                    d_body_d_carry, d_body_d_elems[i], adj, seq, node)
            else:
                # Fallback: _ScanGrad for non-constant derivatives
                wrt_name = seq.name if isinstance(seq, Identifier) else f"__seq_{i}__"
                grad_seq = _ScanGrad(
                    d_body_d_carry=d_body_d_carry,
                    d_body_d_elems=d_body_d_elems,
                    carry_var=carry_var,
                    elem_vars=elem_vars,
                    init=init,
                    sequences=sequences,
                    forward_result=fwd_ref,
                    adj=adj,
                    wrt=wrt_name,
                    span=_DUMMY_SPAN,
                )

            seq_type = self._type_of(seq)
            if seq_type is not None:
                self.type_map[id(grad_seq)] = seq_type
            self._accumulate(adjoints, seq_name, grad_seq)

    def _build_reverse_scan_grad(self, d_carry: Expr, d_elem: Expr,
                                  adj: Expr, seq: Expr, scan_node: Expr) -> ScanExpr:
        """Build a reverse ScanExpr for the backward pass (constant derivative case).

        The backward scan accumulates: new_carry = carry * d_carry + adj_elem * d_elem
        and stacks the carries as the gradient array.
        """
        fresh_carry = self._fresh_name("_adj_c")
        fresh_elem = self._fresh_name("_adj_e")

        # Carry type = element type of scan result (what each step produces)
        scan_type = self._type_of(scan_node)
        if isinstance(scan_type, ArrayType):
            if len(scan_type.dims) == 1:
                carry_type = ScalarType(scan_type.base)
            else:
                carry_type = ArrayType(scan_type.base, scan_type.dims[1:])
        else:
            carry_type = scan_type

        # Create typed carry reference
        carry_ref = Identifier(fresh_carry, _DUMMY_SPAN)
        self.type_map[id(carry_ref)] = carry_type

        # Determine adj element and backward sequence
        adj_type = self._type_of(adj)
        if isinstance(adj_type, ArrayType):
            # adj is an array — use it as the sequence, elem_var gets sliced adj
            back_seq = adj
            adj_element = Identifier(fresh_elem, _DUMMY_SPAN)
            if len(adj_type.dims) == 1:
                self.type_map[id(adj_element)] = ScalarType(adj_type.base)
            else:
                self.type_map[id(adj_element)] = ArrayType(adj_type.base, adj_type.dims[1:])
        else:
            # adj is scalar — use original seq for iteration count, adj directly in body
            back_seq = seq
            adj_element = adj

        # Build body: carry * d_carry + adj_element * d_elem
        # Simplify: skip * 1.0 when derivative is FloatLiteral(1.0)
        def is_one(e):
            return isinstance(e, FloatLiteral) and e.value == 1.0

        term1 = carry_ref if is_one(d_carry) else self._make_binop("*", carry_ref, d_carry)
        term2 = adj_element if is_one(d_elem) else self._make_binop("*", adj_element, d_elem)
        body_expr = self._make_binop("+", term1, term2)
        body_block = Block([], body_expr, _DUMMY_SPAN)

        # Init: zero with carry type
        back_init = self._make_zero(carry_type)

        return ScanExpr(
            carry_var=fresh_carry,
            elem_vars=[fresh_elem],
            init=back_init,
            sequences=[back_seq],
            body=body_block,
            span=_DUMMY_SPAN,
            reverse=True,
        )

    def _backprop_fold(self, carry_var: str, elem_vars: list[str],
                        init: Expr, sequences: list[Expr], body: Block,
                        adj: Expr, adjoints: dict[str, Expr],
                        var_map: dict[int, str], node: Expr):
        """Backprop through fold: emit _FoldGrad nodes."""
        body_expr = body.expr
        if body_expr is None:
            return

        # Compute symbolic derivatives of body w.r.t. carry and each elem
        d_body_d_carry = self._differentiate_branch(body_expr, carry_var)
        d_body_d_elems = [self._differentiate_branch(body_expr, ev) for ev in elem_vars]

        # Propagate to init
        if id(init) in var_map:
            init_name = var_map[id(init)]
            grad_init = _FoldGrad(
                d_body_d_carry=d_body_d_carry,
                d_body_d_elems=d_body_d_elems,
                carry_var=carry_var,
                elem_vars=elem_vars,
                init=init,
                sequences=sequences,
                body=body,
                adj=adj,
                wrt="__init__",
                span=_DUMMY_SPAN,
            )
            init_type = self._type_of(init)
            if init_type is not None:
                self.type_map[id(grad_init)] = init_type
            self._accumulate(adjoints, init_name, grad_init)

        # Propagate to each sequence
        for i, seq in enumerate(sequences):
            if id(seq) not in var_map:
                continue
            seq_name = var_map[id(seq)]

            wrt_name = seq.name if isinstance(seq, Identifier) else f"__seq_{i}__"
            grad_seq = _FoldGrad(
                d_body_d_carry=d_body_d_carry,
                d_body_d_elems=d_body_d_elems,
                carry_var=carry_var,
                elem_vars=elem_vars,
                init=init,
                sequences=sequences,
                body=body,
                adj=adj,
                wrt=wrt_name,
                span=_DUMMY_SPAN,
            )

            seq_type = self._type_of(seq)
            if seq_type is not None:
                self.type_map[id(grad_seq)] = seq_type
            self._accumulate(adjoints, seq_name, grad_seq)

    # -- AST construction helpers --

    def _make_float(self, v: float) -> Expr:
        node = FloatLiteral(v, _DUMMY_SPAN)
        self.type_map[id(node)] = ScalarType("f32")
        return node

    def _make_ref(self, name: str, t: MaomiType | None) -> Expr:
        # If the name is a tape-internal name, return the original expression
        # so codegen sees valid AST instead of undefined internal variables.
        if name in self._tape_exprs:
            return self._tape_exprs[name]
        node = Identifier(name, _DUMMY_SPAN)
        if t is not None:
            self.type_map[id(node)] = t
        return node

    def _make_binop(self, op: str, left: Expr, right: Expr) -> Expr:
        node = BinOp(op, left, right, _DUMMY_SPAN)
        # Infer result type from operands
        lt = self.type_map.get(id(left))
        rt = self.type_map.get(id(right))
        if op == "@":
            # Matmul result type
            if isinstance(lt, ArrayType) and isinstance(rt, ArrayType):
                result_dims = lt.dims[:-1] + rt.dims[1:]
                if result_dims:
                    self.type_map[id(node)] = ArrayType(lt.base, tuple(result_dims))
                else:
                    self.type_map[id(node)] = ScalarType(lt.base)
            elif isinstance(lt, ArrayType):
                self.type_map[id(node)] = lt
            elif isinstance(rt, ArrayType):
                self.type_map[id(node)] = rt
            elif lt is not None:
                self.type_map[id(node)] = lt
        elif lt is not None and rt is not None:
            # Use the "larger" type (broadcast)
            if isinstance(lt, ArrayType):
                self.type_map[id(node)] = lt
            elif isinstance(rt, ArrayType):
                self.type_map[id(node)] = rt
            else:
                self.type_map[id(node)] = lt
        elif lt is not None:
            self.type_map[id(node)] = lt
        elif rt is not None:
            self.type_map[id(node)] = rt
        return node

    def _make_unary(self, op: str, operand: Expr) -> Expr:
        node = UnaryOp(op, operand, _DUMMY_SPAN)
        t = self.type_map.get(id(operand))
        if t is not None:
            self.type_map[id(node)] = t
        return node

    def _make_call(self, name: str, args: list[Expr]) -> Expr:
        node = CallExpr(name, args, _DUMMY_SPAN)
        # Infer type from first arg for elementwise, or special-case transpose/reshape
        if name == "transpose" and args:
            arg_t = self.type_map.get(id(args[0]))
            if isinstance(arg_t, ArrayType) and len(arg_t.dims) == 2:
                self.type_map[id(node)] = ArrayType(arg_t.base, (arg_t.dims[1], arg_t.dims[0]))
            elif arg_t is not None:
                self.type_map[id(node)] = arg_t
        elif name == "reshape" and len(args) > 1:
            arg_t = self.type_map.get(id(args[0]))
            dims = tuple(a.value for a in args[1:] if isinstance(a, IntLiteral))
            if dims and arg_t is not None:
                base = arg_t.base if isinstance(arg_t, ArrayType) else arg_t.base
                self.type_map[id(node)] = ArrayType(base, dims)
        elif args:
            t = self.type_map.get(id(args[0]))
            if t is not None:
                self.type_map[id(node)] = t
        return node

    def _make_zero(self, t: MaomiType) -> Expr:
        if isinstance(t, StructType):
            return self._make_zero_struct(t)
        node = FloatLiteral(0.0, _DUMMY_SPAN)
        self.type_map[id(node)] = t
        return node

    def _make_zero_struct(self, t: StructType) -> Expr:
        """Create a struct literal with all fields zeroed."""
        fields = []
        for fn, ft in t.fields:
            fields.append((fn, self._make_zero(ft)))
        node = StructLiteral(t.name, fields, _DUMMY_SPAN)
        self.type_map[id(node)] = t
        return node

    def _make_struct_with_field(self, stype: StructType, field_name: str, value: Expr) -> Expr:
        """Create a struct literal with one field set to value, rest zeroed."""
        fields = []
        for fn, ft in stype.fields:
            if fn == field_name:
                fields.append((fn, value))
            else:
                fields.append((fn, self._make_zero(ft)))
        node = StructLiteral(stype.name, fields, _DUMMY_SPAN)
        self.type_map[id(node)] = stype
        return node

    def _accumulate(self, adjoints: dict[str, Expr], name: str, contribution: Expr):
        """Add contribution to the adjoint of name."""
        if name in adjoints:
            existing = adjoints[name]
            # For struct-typed adjoints, do field-wise addition
            existing_type = self.type_map.get(id(existing))
            if isinstance(existing_type, StructType):
                adjoints[name] = self._struct_add(existing, contribution, existing_type)
            else:
                adjoints[name] = self._make_binop("+", existing, contribution)
        else:
            adjoints[name] = contribution

    def _struct_add(self, a: Expr, b: Expr, stype: StructType) -> Expr:
        """Field-wise addition of two struct-typed expressions."""
        fields = []
        for fn, ft in stype.fields:
            fa = FieldAccess(a, fn, _DUMMY_SPAN)
            fb = FieldAccess(b, fn, _DUMMY_SPAN)
            self.type_map[id(fa)] = ft
            self.type_map[id(fb)] = ft
            if isinstance(ft, StructType):
                fields.append((fn, self._struct_add(fa, fb, ft)))
            else:
                fields.append((fn, self._make_binop("+", fa, fb)))
        node = StructLiteral(stype.name, fields, _DUMMY_SPAN)
        self.type_map[id(node)] = stype
        return node
