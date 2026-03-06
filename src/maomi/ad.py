"""Reverse-mode automatic differentiation as an AST-to-AST transform.

Rewrites GradExpr nodes into ordinary expression trees that compute
gradients via the chain rule. After this pass, the AST has no GradExpr
nodes — codegen never sees them.

Supported operations inside grad:
  arithmetic (+, -, *, /, **)
  matmul (@)
  elementwise builtins (exp, log, tanh, sqrt, abs)
  reductions (mean, sum)
  let bindings
  identifiers, float/int literals

Not supported (emits compile error):
  if/else, function calls (non-builtin), scan, map, grad-of-grad
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
    MapExpr,
    GradExpr,
    Span,
    Expr,
)
from .types import MaomiType, ScalarType, ArrayType
from .errors import MaomiError

_DUMMY_SPAN = Span(0, 0, 0, 0)

_ELEMENTWISE_BUILTINS = {"exp", "log", "tanh", "sqrt", "abs"}
_REDUCTION_BUILTINS = {"mean", "sum"}


def transform_grad(program: Program, type_map: dict[int, MaomiType]) -> Program:
    """Walk the program and replace all GradExpr nodes with gradient expressions."""
    transformer = ADTransform(type_map)
    new_fns = [transformer.transform_fn(fn) for fn in program.functions]
    return Program(new_fns, program.span)


class ADTransform:
    def __init__(self, type_map: dict[int, MaomiType]):
        self.type_map = type_map
        self._name_counter = 0

    def _fresh_name(self, prefix: str = "_ad") -> str:
        self._name_counter += 1
        return f"{prefix}_{self._name_counter}"

    def _type_of(self, expr: Expr) -> MaomiType | None:
        return self.type_map.get(id(expr))

    def transform_fn(self, fn: FnDef) -> FnDef:
        new_body = self._transform_block(fn.body)
        return FnDef(fn.name, fn.params, fn.return_type, fn.effect, new_body, fn.span)

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
            case _:
                return expr

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
            case CallExpr(args=args):
                for a in args:
                    self._linearize(a, tape, var_map, let_env)
                name = self._fresh_name("v")
                var_map[id(expr)] = name
                tape.append((name, expr))
            case _:
                raise MaomiError(
                    f"grad: unsupported expression type {type(expr).__name__} inside grad",
                    "<ad>", expr.span.line_start, expr.span.col_start,
                )

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
                # dL/dA = dL/dC @ B^T
                r_transposed = self._make_call("transpose", [r_ref])
                self._accumulate(adjoints, l_name, self._make_binop("@", adj, r_transposed))
                # dL/dB = A^T @ dL/dC
                l_transposed = self._make_call("transpose", [l_ref])
                self._accumulate(adjoints, r_name, self._make_binop("@", l_transposed, adj))

            case CallExpr(callee=callee, args=args):
                if callee in _ELEMENTWISE_BUILTINS:
                    self._backprop_elementwise(callee, args, adj, adjoints, var_map, node)
                elif callee in _REDUCTION_BUILTINS:
                    self._backprop_reduction(callee, args, adj, adjoints, var_map, node)
                else:
                    raise MaomiError(
                        f"grad: unsupported function call '{callee}' inside grad",
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

        if callee == "mean":
            # d/dx mean(x) = broadcast(dz / numel)
            if isinstance(arg_type, ArrayType):
                numel = 1
                for d in arg_type.dims:
                    if isinstance(d, int):
                        numel *= d
                    else:
                        raise MaomiError(f"grad: cannot differentiate mean with symbolic dim '{d}'", "<ad>", 0, 0)
                scaled = self._make_binop("/", adj, self._make_float(float(numel)))
                # The scalar adjoint needs to be broadcast back — but since we're
                # building AST nodes, the type checker/codegen will handle broadcasting
                self._accumulate(adjoints, arg_name, scaled)
            else:
                self._accumulate(adjoints, arg_name, adj)

        elif callee == "sum":
            # d/dx sum(x) = broadcast(dz)
            self._accumulate(adjoints, arg_name, adj)

    # -- AST construction helpers --

    def _make_float(self, v: float) -> Expr:
        node = FloatLiteral(v, _DUMMY_SPAN)
        self.type_map[id(node)] = ScalarType("f32")
        return node

    def _make_ref(self, name: str, t: MaomiType | None) -> Expr:
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
        # Infer type from first arg for elementwise, or special-case transpose
        if name == "transpose" and args:
            arg_t = self.type_map.get(id(args[0]))
            if isinstance(arg_t, ArrayType) and len(arg_t.dims) == 2:
                self.type_map[id(node)] = ArrayType(arg_t.base, (arg_t.dims[1], arg_t.dims[0]))
        elif args:
            t = self.type_map.get(id(args[0]))
            if t is not None:
                self.type_map[id(node)] = t
        return node

    def _make_zero(self, t: MaomiType) -> Expr:
        node = FloatLiteral(0.0, _DUMMY_SPAN)
        self.type_map[id(node)] = t
        return node

    def _accumulate(self, adjoints: dict[str, Expr], name: str, contribution: Expr):
        """Add contribution to the adjoint of name."""
        if name in adjoints:
            existing = adjoints[name]
            adjoints[name] = self._make_binop("+", existing, contribution)
        else:
            adjoints[name] = contribution
