"""Central builtin registry — single source of truth for all Maomi builtins.

Each builtin is defined once here with its name, StableHLO op mapping,
AD gradient rule, documentation, and LSP signature info. The type checker,
AD transformer, codegen, and LSP all import from this module instead of
maintaining their own parallel lists.

Two tiers:
  - ElementwiseBuiltin: fully data-driven (single StableHLO op or compound codegen)
  - ComplexBuiltin: registry entry + bespoke handlers in type_checker/ad/codegen
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Any

if TYPE_CHECKING:
    from .ast_nodes import Expr

# Type aliases for grad rule and codegen function signatures.
# GradRuleFn: (ad_ctx, arg_ref, adj) -> gradient_expr
# CodegenFn: (codegen_ctx, call_expr, env) -> ssa_var
GradRuleFn = Callable[..., Any]
CodegenFn = Callable[..., str]


@dataclass(frozen=True)
class ElementwiseBuiltin:
    """A builtin that maps scalar -> scalar, lifts to arrays/structs."""
    name: str
    stablehlo_op: str | None       # None for compound ops (sigmoid, etc.)
    grad_rule: GradRuleFn          # (ad_ctx, arg_ref, adj) -> grad_expr
    doc: str
    codegen_fn: CodegenFn | None = None  # Only for compound ops


@dataclass(frozen=True)
class ComplexBuiltin:
    """A builtin with bespoke type checking, codegen, and/or AD logic."""
    name: str
    category: str                   # "reduction", "shape", "conv_pool", "rng", etc.
    ad_behavior: str                # "zero_grad" | "has_rule" | "nondiff"
    doc: str
    lsp_params: tuple[list[str], list[str], str]  # (param_names, param_types, return_type)


# ---------------------------------------------------------------------------
# Elementwise gradient rules
# ---------------------------------------------------------------------------
# Each takes (ctx, arg_ref, adj) where ctx is the AD transformer instance,
# arg_ref is the SSA reference to the input, adj is the adjoint (upstream grad).
# Returns the gradient expression (Expr).

def _grad_exp(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx exp(x) = exp(x) * dz
    exp_x = ctx._make_call("exp", [arg_ref])
    return ctx._make_binop("*", adj, exp_x)


def _grad_log(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx log(x) = dz / x
    return ctx._make_binop("/", adj, arg_ref)


def _grad_tanh(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx tanh(x) = dz * (1 - tanh(x)^2)
    tanh_x = ctx._make_call("tanh", [arg_ref])
    tanh_sq = ctx._make_binop("*", tanh_x, tanh_x)
    one_minus = ctx._make_binop("-", ctx._make_float(1.0), tanh_sq)
    return ctx._make_binop("*", adj, one_minus)


def _grad_sqrt(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx sqrt(x) = dz / (2 * sqrt(x))
    sqrt_x = ctx._make_call("sqrt", [arg_ref])
    two_sqrt = ctx._make_binop("*", ctx._make_float(2.0), sqrt_x)
    return ctx._make_binop("/", adj, two_sqrt)


def _grad_abs(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx |x| = dz * sign(x) — approximate as x / |x|
    abs_x = ctx._make_call("abs", [arg_ref])
    sign = ctx._make_binop("/", arg_ref, abs_x)
    return ctx._make_binop("*", adj, sign)


def _grad_cos(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx cos(x) = -sin(x) * dz
    sin_x = ctx._make_call("sin", [arg_ref])
    neg_sin = ctx._make_unary("-", sin_x)
    return ctx._make_binop("*", adj, neg_sin)


def _grad_sin(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx sin(x) = cos(x) * dz
    cos_x = ctx._make_call("cos", [arg_ref])
    return ctx._make_binop("*", adj, cos_x)


# -- New builtins --

def _grad_neg(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx (-x) = -dz
    return ctx._make_unary("-", adj)


def _grad_sigmoid(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) * dz
    sig = ctx._make_call("sigmoid", [arg_ref])
    one_minus_sig = ctx._make_binop("-", ctx._make_float(1.0), sig)
    return ctx._make_binop("*", adj, ctx._make_binop("*", sig, one_minus_sig))


def _grad_log2(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx log2(x) = dz / (x * ln(2))
    # ln(2) ≈ 0.6931471805599453
    x_ln2 = ctx._make_binop("*", arg_ref, ctx._make_float(0.6931471805599453))
    return ctx._make_binop("/", adj, x_ln2)


def _grad_rsqrt(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx rsqrt(x) = -0.5 * x^(-3/2) * dz = -0.5 * rsqrt(x)^3 * dz
    rsqrt_x = ctx._make_call("rsqrt", [arg_ref])
    rsqrt_cubed = ctx._make_binop("*", rsqrt_x, ctx._make_binop("*", rsqrt_x, rsqrt_x))
    neg_half = ctx._make_float(-0.5)
    return ctx._make_binop("*", adj, ctx._make_binop("*", neg_half, rsqrt_cubed))


def _grad_floor(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # floor is piecewise constant — gradient is zero
    return ctx._make_float(0.0)


def _grad_ceil(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # ceil is piecewise constant — gradient is zero
    return ctx._make_float(0.0)


def _grad_sign(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # sign is piecewise constant — gradient is zero
    return ctx._make_float(0.0)


def _grad_reciprocal(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx (1/x) = -1/x^2 * dz = -reciprocal(x)^2 * dz
    recip = ctx._make_call("reciprocal", [arg_ref])
    recip_sq = ctx._make_binop("*", recip, recip)
    return ctx._make_binop("*", adj, ctx._make_unary("-", recip_sq))


def _grad_log1p(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx log(1+x) = 1/(1+x) * dz
    one_plus_x = ctx._make_binop("+", ctx._make_float(1.0), arg_ref)
    return ctx._make_binop("/", adj, one_plus_x)


def _grad_square(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx x^2 = 2x * dz
    two_x = ctx._make_binop("*", ctx._make_float(2.0), arg_ref)
    return ctx._make_binop("*", two_x, adj)


def _grad_softplus(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx log(1 + exp(x)) = sigmoid(x) * dz
    sig = ctx._make_call("sigmoid", [arg_ref])
    return ctx._make_binop("*", sig, adj)


def _grad_relu(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx relu(x) = (x > 0) * dz
    # Compare x > 0, cast bool to f32, multiply by adj
    from .ast_nodes import BinOp, CastExpr
    from .types import ScalarType, ArrayType

    zero = ctx._make_float(0.0)
    arg_type = ctx.type_map.get(id(arg_ref))
    if isinstance(arg_type, ArrayType):
        bool_type = ArrayType("bool", arg_type.dims)
    else:
        bool_type = ScalarType("bool")
        if not isinstance(arg_type, ScalarType):
            arg_type = ScalarType("f32")

    cmp = BinOp(">", arg_ref, zero, arg_ref.span)
    ctx.type_map[id(cmp)] = bool_type
    mask = CastExpr(cmp, arg_type.base, arg_ref.span)
    ctx.type_map[id(mask)] = arg_type
    return ctx._make_binop("*", mask, adj)


def _grad_silu(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx x*sigmoid(x) = sigmoid(x) + x*sigmoid(x)*(1 - sigmoid(x))
    #                    = sigmoid(x) * (1 + x*(1 - sigmoid(x))) * dz
    sig = ctx._make_call("sigmoid", [arg_ref])
    one = ctx._make_float(1.0)
    one_minus_sig = ctx._make_binop("-", one, sig)
    x_sig_1ms = ctx._make_binop("*", arg_ref, ctx._make_binop("*", sig, one_minus_sig))
    inner = ctx._make_binop("+", sig, x_sig_1ms)
    return ctx._make_binop("*", adj, inner)


def _grad_gelu(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # GELU (tanh approx): 0.5 * x * (1 + tanh(c * (x + k*x^3)))
    # where c = sqrt(2/pi) ≈ 0.7978845608, k = 0.044715
    # Derivative:
    # gelu'(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * c * (1 + 3*k*x^2)
    # where t = tanh(c * (x + k*x^3))
    c = ctx._make_float(0.7978845608)
    k = ctx._make_float(0.044715)
    three_k = ctx._make_float(3.0 * 0.044715)
    half = ctx._make_float(0.5)
    one = ctx._make_float(1.0)

    x_sq = ctx._make_binop("*", arg_ref, arg_ref)
    x_cu = ctx._make_binop("*", x_sq, arg_ref)
    inner = ctx._make_binop("*", c, ctx._make_binop("+", arg_ref, ctx._make_binop("*", k, x_cu)))
    t = ctx._make_call("tanh", [inner])
    t_sq = ctx._make_binop("*", t, t)

    # 0.5 * (1 + t)
    term1 = ctx._make_binop("*", half, ctx._make_binop("+", one, t))

    # 0.5 * x * (1 - t^2) * c * (1 + 3*k*x^2)
    one_minus_tsq = ctx._make_binop("-", one, t_sq)
    one_plus_3kx2 = ctx._make_binop("+", one, ctx._make_binop("*", three_k, x_sq))
    term2 = ctx._make_binop("*", half,
                ctx._make_binop("*", arg_ref,
                    ctx._make_binop("*", one_minus_tsq,
                        ctx._make_binop("*", c, one_plus_3kx2))))

    grad_val = ctx._make_binop("+", term1, term2)
    return ctx._make_binop("*", adj, grad_val)


def _grad_expm1(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx expm1(x) = exp(x) * dz  (same as exp)
    exp_x = ctx._make_call("exp", [arg_ref])
    return ctx._make_binop("*", adj, exp_x)


# -- Trig/math builtins --

def _grad_tan(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx tan(x) = (1 + tan(x)^2) * dz  (JAX: g*(1+ans^2))
    tan_x = ctx._make_call("tan", [arg_ref])
    tan_sq = ctx._make_binop("*", tan_x, tan_x)
    sec_sq = ctx._make_binop("+", ctx._make_float(1.0), tan_sq)
    return ctx._make_binop("*", adj, sec_sq)


def _grad_sinh(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx sinh(x) = cosh(x) * dz
    cosh_x = ctx._make_call("cosh", [arg_ref])
    return ctx._make_binop("*", adj, cosh_x)


def _grad_cosh(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx cosh(x) = sinh(x) * dz
    sinh_x = ctx._make_call("sinh", [arg_ref])
    return ctx._make_binop("*", adj, sinh_x)


def _grad_asin(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx asin(x) = dz * rsqrt(1 - x^2)  (JAX uses rsqrt)
    x_sq = ctx._make_binop("*", arg_ref, arg_ref)
    one_minus_xsq = ctx._make_binop("-", ctx._make_float(1.0), x_sq)
    inv_denom = ctx._make_call("rsqrt", [one_minus_xsq])
    return ctx._make_binop("*", adj, inv_denom)


def _grad_acos(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx acos(x) = -dz * rsqrt(1 - x^2)
    x_sq = ctx._make_binop("*", arg_ref, arg_ref)
    one_minus_xsq = ctx._make_binop("-", ctx._make_float(1.0), x_sq)
    inv_denom = ctx._make_call("rsqrt", [one_minus_xsq])
    neg_inv = ctx._make_unary("-", inv_denom)
    return ctx._make_binop("*", adj, neg_inv)


def _grad_atan(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx atan(x) = dz / (1 + x^2)
    x_sq = ctx._make_binop("*", arg_ref, arg_ref)
    denom = ctx._make_binop("+", ctx._make_float(1.0), x_sq)
    return ctx._make_binop("/", adj, denom)


def _grad_asinh(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx asinh(x) = dz * rsqrt(x^2 + 1)
    x_sq = ctx._make_binop("*", arg_ref, arg_ref)
    xsq_plus_1 = ctx._make_binop("+", x_sq, ctx._make_float(1.0))
    inv_denom = ctx._make_call("rsqrt", [xsq_plus_1])
    return ctx._make_binop("*", adj, inv_denom)


def _grad_acosh(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx acosh(x) = dz * rsqrt(x^2 - 1)  (JAX uses x^2-1, domain x>=1)
    x_sq = ctx._make_binop("*", arg_ref, arg_ref)
    xsq_minus_1 = ctx._make_binop("-", x_sq, ctx._make_float(1.0))
    inv_denom = ctx._make_call("rsqrt", [xsq_minus_1])
    return ctx._make_binop("*", adj, inv_denom)


def _grad_atanh(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx atanh(x) = dz / (1 - x^2)
    # JAX factors as reciprocal(1+x) * (g / (1-x)) for stability
    one_plus_x = ctx._make_binop("+", ctx._make_float(1.0), arg_ref)
    one_minus_x = ctx._make_binop("-", ctx._make_float(1.0), arg_ref)
    recip_1px = ctx._make_call("reciprocal", [one_plus_x])
    adj_over_1mx = ctx._make_binop("/", adj, one_minus_x)
    return ctx._make_binop("*", recip_1px, adj_over_1mx)


def _grad_exp2(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx 2^x = ln(2) * 2^x * dz  (JAX: log(2)*g*ans)
    exp2_x = ctx._make_call("exp2", [arg_ref])
    ln2 = ctx._make_float(0.6931471805599453)
    return ctx._make_binop("*", adj, ctx._make_binop("*", ln2, exp2_x))


def _grad_log10(ctx: Any, arg_ref: Expr, adj: Expr) -> Expr:
    # d/dx log10(x) = dz / (x * ln(10))
    ln10 = ctx._make_float(2.302585092994046)
    x_ln10 = ctx._make_binop("*", arg_ref, ln10)
    return ctx._make_binop("/", adj, x_ln10)


# ---------------------------------------------------------------------------
# Compound elementwise codegen functions
# ---------------------------------------------------------------------------
# For builtins without a single StableHLO op, these emit compound StableHLO.
# Each takes (codegen_ctx, call_expr, env) -> SSA variable name.
# Uses the codegen's internal API: _gen_expr, _type_of, _fresh, _emit.
# Uses the module-level _mlir_type function from codegen_stablehlo.

def _compound_codegen(name: str, inner_fn: Callable[..., str]) -> CodegenFn:
    """Create a compound codegen wrapper that handles StructType dispatch.

    Every compound elementwise builtin has the same outer pattern: evaluate
    the argument, check for StructType (field-by-field application), then
    delegate to the inner function that emits the actual StableHLO ops.
    """
    def codegen_fn(codegen: Any, expr: Any, env: dict[str, str]) -> str:
        from .codegen.stablehlo.utils import _mlir_type
        from .types import StructType

        arg = codegen._gen_expr(expr.args[0], env)
        result_type = codegen._type_of(expr)

        if isinstance(result_type, StructType):
            return codegen._gen_struct_compound_elementwise(
                name, inner_fn, arg, result_type,
            )

        mlir_t = _mlir_type(result_type)
        return inner_fn(codegen, arg, mlir_t)

    codegen_fn.__doc__ = f"Compound codegen for {name}"
    return codegen_fn


def _codegen_sigmoid_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    neg_x = codegen._fresh()
    codegen._emit(f"{neg_x} = stablehlo.negate {arg_ssa} : {mlir_t}")
    exp_neg = codegen._fresh()
    codegen._emit(f"{exp_neg} = stablehlo.exponential {neg_x} : {mlir_t}")
    one = codegen._fresh()
    codegen._emit(f"{one} = stablehlo.constant dense<1.000000e+00> : {mlir_t}")
    denom = codegen._fresh()
    codegen._emit(f"{denom} = stablehlo.add {one}, {exp_neg} : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.divide {one}, {denom} : {mlir_t}")
    return result

_codegen_sigmoid = _compound_codegen("sigmoid", _codegen_sigmoid_inner)


def _codegen_log2_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    log_x = codegen._fresh()
    codegen._emit(f"{log_x} = stablehlo.log {arg_ssa} : {mlir_t}")
    ln2 = codegen._fresh()
    codegen._emit(f"{ln2} = stablehlo.constant dense<6.931470e-01> : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.divide {log_x}, {ln2} : {mlir_t}")
    return result

_codegen_log2 = _compound_codegen("log2", _codegen_log2_inner)


def _codegen_reciprocal_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    one = codegen._fresh()
    codegen._emit(f"{one} = stablehlo.constant dense<1.000000e+00> : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.divide {one}, {arg_ssa} : {mlir_t}")
    return result

_codegen_reciprocal = _compound_codegen("reciprocal", _codegen_reciprocal_inner)


def _codegen_square_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.multiply {arg_ssa}, {arg_ssa} : {mlir_t}")
    return result

_codegen_square = _compound_codegen("square", _codegen_square_inner)


def _codegen_softplus_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    exp_x = codegen._fresh()
    codegen._emit(f"{exp_x} = stablehlo.exponential {arg_ssa} : {mlir_t}")
    one = codegen._fresh()
    codegen._emit(f"{one} = stablehlo.constant dense<1.000000e+00> : {mlir_t}")
    sum_val = codegen._fresh()
    codegen._emit(f"{sum_val} = stablehlo.add {one}, {exp_x} : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.log {sum_val} : {mlir_t}")
    return result

_codegen_softplus = _compound_codegen("softplus", _codegen_softplus_inner)


def _codegen_relu_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    # Build bool type matching the shape
    bool_t = mlir_t.replace("f32", "i1").replace("f64", "i1")
    zero = codegen._fresh()
    codegen._emit(f"{zero} = stablehlo.constant dense<0.000000e+00> : {mlir_t}")
    cmp = codegen._fresh()
    codegen._emit(f"{cmp} = stablehlo.compare GT, {arg_ssa}, {zero} : ({mlir_t}, {mlir_t}) -> {bool_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.select {cmp}, {arg_ssa}, {zero} : ({bool_t}, {mlir_t}, {mlir_t}) -> {mlir_t}")
    return result

_codegen_relu = _compound_codegen("relu", _codegen_relu_inner)


def _codegen_silu_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    sig = _codegen_sigmoid_inner(codegen, arg_ssa, mlir_t)
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.multiply {arg_ssa}, {sig} : {mlir_t}")
    return result

_codegen_silu = _compound_codegen("silu", _codegen_silu_inner)


def _codegen_gelu_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    # Constants
    half = codegen._fresh()
    codegen._emit(f"{half} = stablehlo.constant dense<5.000000e-01> : {mlir_t}")
    one = codegen._fresh()
    codegen._emit(f"{one} = stablehlo.constant dense<1.000000e+00> : {mlir_t}")
    c = codegen._fresh()
    codegen._emit(f"{c} = stablehlo.constant dense<7.978846e-01> : {mlir_t}")
    k = codegen._fresh()
    codegen._emit(f"{k} = stablehlo.constant dense<4.471500e-02> : {mlir_t}")

    # x^3 = x * x * x
    x_sq = codegen._fresh()
    codegen._emit(f"{x_sq} = stablehlo.multiply {arg_ssa}, {arg_ssa} : {mlir_t}")
    x_cu = codegen._fresh()
    codegen._emit(f"{x_cu} = stablehlo.multiply {x_sq}, {arg_ssa} : {mlir_t}")

    # k * x^3
    k_x3 = codegen._fresh()
    codegen._emit(f"{k_x3} = stablehlo.multiply {k}, {x_cu} : {mlir_t}")

    # x + k*x^3
    x_plus_kx3 = codegen._fresh()
    codegen._emit(f"{x_plus_kx3} = stablehlo.add {arg_ssa}, {k_x3} : {mlir_t}")

    # c * (x + k*x^3)
    inner = codegen._fresh()
    codegen._emit(f"{inner} = stablehlo.multiply {c}, {x_plus_kx3} : {mlir_t}")

    # tanh(inner)
    tanh_inner = codegen._fresh()
    codegen._emit(f"{tanh_inner} = stablehlo.tanh {inner} : {mlir_t}")

    # 1 + tanh(inner)
    one_plus_tanh = codegen._fresh()
    codegen._emit(f"{one_plus_tanh} = stablehlo.add {one}, {tanh_inner} : {mlir_t}")

    # x * (1 + tanh(inner))
    x_times = codegen._fresh()
    codegen._emit(f"{x_times} = stablehlo.multiply {arg_ssa}, {one_plus_tanh} : {mlir_t}")

    # 0.5 * x * (1 + tanh(inner))
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.multiply {half}, {x_times} : {mlir_t}")
    return result

_codegen_gelu = _compound_codegen("gelu", _codegen_gelu_inner)


# -- Trig/math compound codegen --

def _codegen_tan_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    # tan(x) = sin(x) / cos(x)
    sin_x = codegen._fresh()
    codegen._emit(f"{sin_x} = stablehlo.sine {arg_ssa} : {mlir_t}")
    cos_x = codegen._fresh()
    codegen._emit(f"{cos_x} = stablehlo.cosine {arg_ssa} : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.divide {sin_x}, {cos_x} : {mlir_t}")
    return result

_codegen_tan = _compound_codegen("tan", _codegen_tan_inner)


def _codegen_sinh_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    # sinh(x) = (exp(x) - exp(-x)) / 2
    exp_x = codegen._fresh()
    codegen._emit(f"{exp_x} = stablehlo.exponential {arg_ssa} : {mlir_t}")
    neg_x = codegen._fresh()
    codegen._emit(f"{neg_x} = stablehlo.negate {arg_ssa} : {mlir_t}")
    exp_neg = codegen._fresh()
    codegen._emit(f"{exp_neg} = stablehlo.exponential {neg_x} : {mlir_t}")
    diff = codegen._fresh()
    codegen._emit(f"{diff} = stablehlo.subtract {exp_x}, {exp_neg} : {mlir_t}")
    two = codegen._fresh()
    codegen._emit(f"{two} = stablehlo.constant dense<2.000000e+00> : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.divide {diff}, {two} : {mlir_t}")
    return result

_codegen_sinh = _compound_codegen("sinh", _codegen_sinh_inner)


def _codegen_cosh_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    # cosh(x) = (exp(x) + exp(-x)) / 2
    exp_x = codegen._fresh()
    codegen._emit(f"{exp_x} = stablehlo.exponential {arg_ssa} : {mlir_t}")
    neg_x = codegen._fresh()
    codegen._emit(f"{neg_x} = stablehlo.negate {arg_ssa} : {mlir_t}")
    exp_neg = codegen._fresh()
    codegen._emit(f"{exp_neg} = stablehlo.exponential {neg_x} : {mlir_t}")
    sum_val = codegen._fresh()
    codegen._emit(f"{sum_val} = stablehlo.add {exp_x}, {exp_neg} : {mlir_t}")
    two = codegen._fresh()
    codegen._emit(f"{two} = stablehlo.constant dense<2.000000e+00> : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.divide {sum_val}, {two} : {mlir_t}")
    return result

_codegen_cosh = _compound_codegen("cosh", _codegen_cosh_inner)


def _codegen_asin_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    # asin(x) = atan2(x, sqrt(1 - x^2))
    x_sq = codegen._fresh()
    codegen._emit(f"{x_sq} = stablehlo.multiply {arg_ssa}, {arg_ssa} : {mlir_t}")
    one = codegen._fresh()
    codegen._emit(f"{one} = stablehlo.constant dense<1.000000e+00> : {mlir_t}")
    one_minus = codegen._fresh()
    codegen._emit(f"{one_minus} = stablehlo.subtract {one}, {x_sq} : {mlir_t}")
    sqrt_val = codegen._fresh()
    codegen._emit(f"{sqrt_val} = stablehlo.sqrt {one_minus} : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.atan2 {arg_ssa}, {sqrt_val} : {mlir_t}")
    return result

_codegen_asin = _compound_codegen("asin", _codegen_asin_inner)


def _codegen_acos_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    # acos(x) = atan2(sqrt(1 - x^2), x)
    x_sq = codegen._fresh()
    codegen._emit(f"{x_sq} = stablehlo.multiply {arg_ssa}, {arg_ssa} : {mlir_t}")
    one = codegen._fresh()
    codegen._emit(f"{one} = stablehlo.constant dense<1.000000e+00> : {mlir_t}")
    one_minus = codegen._fresh()
    codegen._emit(f"{one_minus} = stablehlo.subtract {one}, {x_sq} : {mlir_t}")
    sqrt_val = codegen._fresh()
    codegen._emit(f"{sqrt_val} = stablehlo.sqrt {one_minus} : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.atan2 {sqrt_val}, {arg_ssa} : {mlir_t}")
    return result

_codegen_acos = _compound_codegen("acos", _codegen_acos_inner)


def _codegen_atan_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    # atan(x) = atan2(x, 1)
    one = codegen._fresh()
    codegen._emit(f"{one} = stablehlo.constant dense<1.000000e+00> : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.atan2 {arg_ssa}, {one} : {mlir_t}")
    return result

_codegen_atan = _compound_codegen("atan", _codegen_atan_inner)


def _codegen_asinh_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    # asinh(x) = log(x + sqrt(x^2 + 1))
    x_sq = codegen._fresh()
    codegen._emit(f"{x_sq} = stablehlo.multiply {arg_ssa}, {arg_ssa} : {mlir_t}")
    one = codegen._fresh()
    codegen._emit(f"{one} = stablehlo.constant dense<1.000000e+00> : {mlir_t}")
    xsq_p1 = codegen._fresh()
    codegen._emit(f"{xsq_p1} = stablehlo.add {x_sq}, {one} : {mlir_t}")
    sqrt_val = codegen._fresh()
    codegen._emit(f"{sqrt_val} = stablehlo.sqrt {xsq_p1} : {mlir_t}")
    x_plus = codegen._fresh()
    codegen._emit(f"{x_plus} = stablehlo.add {arg_ssa}, {sqrt_val} : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.log {x_plus} : {mlir_t}")
    return result

_codegen_asinh = _compound_codegen("asinh", _codegen_asinh_inner)


def _codegen_acosh_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    # acosh(x) = log(x + sqrt(x^2 - 1))
    x_sq = codegen._fresh()
    codegen._emit(f"{x_sq} = stablehlo.multiply {arg_ssa}, {arg_ssa} : {mlir_t}")
    one = codegen._fresh()
    codegen._emit(f"{one} = stablehlo.constant dense<1.000000e+00> : {mlir_t}")
    xsq_m1 = codegen._fresh()
    codegen._emit(f"{xsq_m1} = stablehlo.subtract {x_sq}, {one} : {mlir_t}")
    sqrt_val = codegen._fresh()
    codegen._emit(f"{sqrt_val} = stablehlo.sqrt {xsq_m1} : {mlir_t}")
    x_plus = codegen._fresh()
    codegen._emit(f"{x_plus} = stablehlo.add {arg_ssa}, {sqrt_val} : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.log {x_plus} : {mlir_t}")
    return result

_codegen_acosh = _compound_codegen("acosh", _codegen_acosh_inner)


def _codegen_atanh_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    # atanh(x) = 0.5 * log((1+x) / (1-x))
    one = codegen._fresh()
    codegen._emit(f"{one} = stablehlo.constant dense<1.000000e+00> : {mlir_t}")
    half = codegen._fresh()
    codegen._emit(f"{half} = stablehlo.constant dense<5.000000e-01> : {mlir_t}")
    one_plus = codegen._fresh()
    codegen._emit(f"{one_plus} = stablehlo.add {one}, {arg_ssa} : {mlir_t}")
    one_minus = codegen._fresh()
    codegen._emit(f"{one_minus} = stablehlo.subtract {one}, {arg_ssa} : {mlir_t}")
    ratio = codegen._fresh()
    codegen._emit(f"{ratio} = stablehlo.divide {one_plus}, {one_minus} : {mlir_t}")
    log_val = codegen._fresh()
    codegen._emit(f"{log_val} = stablehlo.log {ratio} : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.multiply {half}, {log_val} : {mlir_t}")
    return result

_codegen_atanh = _compound_codegen("atanh", _codegen_atanh_inner)


def _codegen_exp2_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    # exp2(x) = exp(x * ln(2))
    ln2 = codegen._fresh()
    codegen._emit(f"{ln2} = stablehlo.constant dense<6.931472e-01> : {mlir_t}")
    x_ln2 = codegen._fresh()
    codegen._emit(f"{x_ln2} = stablehlo.multiply {arg_ssa}, {ln2} : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.exponential {x_ln2} : {mlir_t}")
    return result

_codegen_exp2 = _compound_codegen("exp2", _codegen_exp2_inner)


def _codegen_log10_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    # log10(x) = log(x) / ln(10)
    log_x = codegen._fresh()
    codegen._emit(f"{log_x} = stablehlo.log {arg_ssa} : {mlir_t}")
    ln10 = codegen._fresh()
    codegen._emit(f"{ln10} = stablehlo.constant dense<2.302585e+00> : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.divide {log_x}, {ln10} : {mlir_t}")
    return result

_codegen_log10 = _compound_codegen("log10", _codegen_log10_inner)


# ---------------------------------------------------------------------------
# Elementwise builtin registry
# ---------------------------------------------------------------------------

ELEMENTWISE: dict[str, ElementwiseBuiltin] = {
    # -- Existing builtins (7) --
    "exp": ElementwiseBuiltin(
        "exp", "stablehlo.exponential", _grad_exp,
        "Compute element-wise exponential (e^x).",
    ),
    "log": ElementwiseBuiltin(
        "log", "stablehlo.log", _grad_log,
        "Compute element-wise natural logarithm (ln x).",
    ),
    "tanh": ElementwiseBuiltin(
        "tanh", "stablehlo.tanh", _grad_tanh,
        "Compute element-wise hyperbolic tangent.",
    ),
    "sqrt": ElementwiseBuiltin(
        "sqrt", "stablehlo.sqrt", _grad_sqrt,
        "Compute element-wise square root.",
    ),
    "abs": ElementwiseBuiltin(
        "abs", "stablehlo.abs", _grad_abs,
        "Compute element-wise absolute value.",
    ),
    "cos": ElementwiseBuiltin(
        "cos", "stablehlo.cosine", _grad_cos,
        "Compute element-wise cosine.",
    ),
    "sin": ElementwiseBuiltin(
        "sin", "stablehlo.sine", _grad_sin,
        "Compute element-wise sine.",
    ),

    # -- New builtins --
    "neg": ElementwiseBuiltin(
        "neg", "stablehlo.negate", _grad_neg,
        "Compute element-wise negation (-x).",
    ),
    "sigmoid": ElementwiseBuiltin(
        "sigmoid", None, _grad_sigmoid,
        "Compute element-wise sigmoid (logistic function): 1 / (1 + exp(-x)).",
        codegen_fn=_codegen_sigmoid,
    ),
    "log2": ElementwiseBuiltin(
        "log2", None, _grad_log2,
        "Compute element-wise base-2 logarithm.",
        codegen_fn=_codegen_log2,
    ),
    "rsqrt": ElementwiseBuiltin(
        "rsqrt", "stablehlo.rsqrt", _grad_rsqrt,
        "Compute element-wise reciprocal square root (1 / sqrt(x)).",
    ),
    "floor": ElementwiseBuiltin(
        "floor", "stablehlo.floor", _grad_floor,
        "Compute element-wise floor (round toward negative infinity). Not differentiable (gradient is zero).",
    ),
    "ceil": ElementwiseBuiltin(
        "ceil", "stablehlo.ceil", _grad_ceil,
        "Compute element-wise ceiling (round toward positive infinity). Not differentiable (gradient is zero).",
    ),
    "sign": ElementwiseBuiltin(
        "sign", "stablehlo.sign", _grad_sign,
        "Compute element-wise sign (-1, 0, or 1). Not differentiable (gradient is zero).",
    ),
    "reciprocal": ElementwiseBuiltin(
        "reciprocal", None, _grad_reciprocal,
        "Compute element-wise reciprocal (1/x).",
        codegen_fn=_codegen_reciprocal,
    ),
    "log1p": ElementwiseBuiltin(
        "log1p", "stablehlo.log_plus_one", _grad_log1p,
        "Compute element-wise log(1 + x), accurate for small x.",
    ),
    "expm1": ElementwiseBuiltin(
        "expm1", "stablehlo.exponential_minus_one", _grad_expm1,
        "Compute element-wise exp(x) - 1, accurate for small x.",
    ),

    # -- Compound activation functions --
    "square": ElementwiseBuiltin(
        "square", None, _grad_square,
        "Compute element-wise square (x^2).",
        codegen_fn=_codegen_square,
    ),
    "softplus": ElementwiseBuiltin(
        "softplus", None, _grad_softplus,
        "Compute element-wise softplus: log(1 + exp(x)).",
        codegen_fn=_codegen_softplus,
    ),
    "relu": ElementwiseBuiltin(
        "relu", None, _grad_relu,
        "Compute element-wise ReLU: max(x, 0).",
        codegen_fn=_codegen_relu,
    ),
    "silu": ElementwiseBuiltin(
        "silu", None, _grad_silu,
        "Compute element-wise SiLU/Swish: x * sigmoid(x).",
        codegen_fn=_codegen_silu,
    ),
    "gelu": ElementwiseBuiltin(
        "gelu", None, _grad_gelu,
        "Compute element-wise GELU (Gaussian Error Linear Unit, tanh approximation).",
        codegen_fn=_codegen_gelu,
    ),

    # -- Trig/math builtins --
    "tan": ElementwiseBuiltin(
        "tan", None, _grad_tan,
        "Compute element-wise tangent.",
        codegen_fn=_codegen_tan,
    ),
    "sinh": ElementwiseBuiltin(
        "sinh", None, _grad_sinh,
        "Compute element-wise hyperbolic sine.",
        codegen_fn=_codegen_sinh,
    ),
    "cosh": ElementwiseBuiltin(
        "cosh", None, _grad_cosh,
        "Compute element-wise hyperbolic cosine.",
        codegen_fn=_codegen_cosh,
    ),
    "asin": ElementwiseBuiltin(
        "asin", None, _grad_asin,
        "Compute element-wise arcsine (inverse sine).",
        codegen_fn=_codegen_asin,
    ),
    "acos": ElementwiseBuiltin(
        "acos", None, _grad_acos,
        "Compute element-wise arccosine (inverse cosine).",
        codegen_fn=_codegen_acos,
    ),
    "atan": ElementwiseBuiltin(
        "atan", None, _grad_atan,
        "Compute element-wise arctangent (inverse tangent).",
        codegen_fn=_codegen_atan,
    ),
    "asinh": ElementwiseBuiltin(
        "asinh", None, _grad_asinh,
        "Compute element-wise inverse hyperbolic sine.",
        codegen_fn=_codegen_asinh,
    ),
    "acosh": ElementwiseBuiltin(
        "acosh", None, _grad_acosh,
        "Compute element-wise inverse hyperbolic cosine (domain x >= 1).",
        codegen_fn=_codegen_acosh,
    ),
    "atanh": ElementwiseBuiltin(
        "atanh", None, _grad_atanh,
        "Compute element-wise inverse hyperbolic tangent (domain |x| < 1).",
        codegen_fn=_codegen_atanh,
    ),
    "exp2": ElementwiseBuiltin(
        "exp2", None, _grad_exp2,
        "Compute element-wise base-2 exponential (2^x).",
        codegen_fn=_codegen_exp2,
    ),
    "log10": ElementwiseBuiltin(
        "log10", None, _grad_log10,
        "Compute element-wise base-10 logarithm.",
        codegen_fn=_codegen_log10,
    ),
}


# ---------------------------------------------------------------------------
# Complex builtin registry
# ---------------------------------------------------------------------------

COMPLEX: dict[str, ComplexBuiltin] = {
    # Reductions
    "sum": ComplexBuiltin(
        "sum", "reduction", "has_rule",
        "Compute the sum. `sum(x)` over all elements, `sum(x, axis=1)` along an axis, `sum(x, axis=1, keepdims=true)` to keep reduced dim as size 1.",
        (["x", "axis", "keepdims"], ["f32[...]", "int", "bool"], "f32"),
    ),
    "mean": ComplexBuiltin(
        "mean", "reduction", "has_rule",
        "Compute the mean. `mean(x)` over all elements, `mean(x, axis=1)` along an axis, `mean(x, axis=1, keepdims=true)` to keep reduced dim as size 1.",
        (["x", "axis", "keepdims"], ["f32[...]", "int", "bool"], "f32"),
    ),
    "max": ComplexBuiltin(
        "max", "reduction", "has_rule",
        "Reduce-max. `max(x)` over all elements, `max(x, axis=1)` along an axis, `max(x, axis=1, keepdims=true)` to keep reduced dim as size 1.",
        (["x", "axis", "keepdims"], ["f32[...]", "int", "bool"], "f32"),
    ),
    "min": ComplexBuiltin(
        "min", "reduction", "has_rule",
        "Reduce-min. `min(x)` over all elements, `min(x, axis=1)` along an axis, `min(x, axis=1, keepdims=true)` to keep reduced dim as size 1.",
        (["x", "axis", "keepdims"], ["f32[...]", "int", "bool"], "f32"),
    ),
    "logsumexp": ComplexBuiltin(
        "logsumexp", "reduction", "has_rule",
        "Numerically stable log-sum-exp. `logsumexp(x)` over all elements, `logsumexp(x, axis=1)` along an axis, `logsumexp(x, axis=1, keepdims=true)` to keep reduced dim as size 1.",
        (["x", "axis", "keepdims"], ["f32[...]", "int", "bool"], "f32"),
    ),

    # Argmax/argmin
    "argmax": ComplexBuiltin(
        "argmax", "argmax", "zero_grad",
        "Index of maximum element. `argmax(x)` over all elements, `argmax(x, axis)` along an axis. Returns i32.",
        (["x"], ["f32[...]"], "i32"),
    ),
    "argmin": ComplexBuiltin(
        "argmin", "argmax", "zero_grad",
        "Index of minimum element. `argmin(x)` over all elements, `argmin(x, axis)` along an axis. Returns i32.",
        (["x"], ["f32[...]"], "i32"),
    ),

    # Shape operations
    "reshape": ComplexBuiltin(
        "reshape", "shape", "has_rule",
        "Reshape an array to the given dimensions.\n\nTotal element count must be preserved.",
        (["x", "dims..."], ["f32[...]", "int..."], "f32[...]"),
    ),
    "concat": ComplexBuiltin(
        "concat", "shape", "has_rule",
        "Concatenate arrays along the given axis.",
        (["arrays...", "axis"], ["f32[...]...", "int"], "f32[...]"),
    ),
    "transpose": ComplexBuiltin(
        "transpose", "shape", "has_rule",
        "Permute array axes. transpose(x) swaps 2D; transpose(x, 0, 2, 1, 3) for general permutation.",
        (["x", "perm..."], ["f32[...]", "int..."], "f32[...]"),
    ),
    "stack": ComplexBuiltin(
        "stack", "shape", "has_rule",
        "Stack arrays along a new axis. Last argument is the axis.",
        (["arrays...", "axis"], ["f32[...]...", "int"], "f32[...]"),
    ),
    "pad": ComplexBuiltin(
        "pad", "shape", "has_rule",
        "Pad array with constant value. pad(x, val, pad_lo, pad_hi).",
        (["x", "val", "pad_lo", "pad_hi"], ["f32[...]", "f32", "int", "int"], "f32[...]"),
    ),
    "expand_dims": ComplexBuiltin(
        "expand_dims", "shape", "has_rule",
        "Insert a size-1 dimension at the given axis.",
        (["x", "axis"], ["f32[...]", "int"], "f32[...]"),
    ),
    "squeeze": ComplexBuiltin(
        "squeeze", "shape", "has_rule",
        "Remove a size-1 dimension at the given axis.",
        (["x", "axis"], ["f32[...]", "int"], "f32[...]"),
    ),
    "broadcast_to": ComplexBuiltin(
        "broadcast_to", "shape", "has_rule",
        "Broadcast array to the given shape.",
        (["x", "dims..."], ["f32[...]", "int..."], "f32[...]"),
    ),

    # Array construction
    "iota": ComplexBuiltin(
        "iota", "construction", "zero_grad",
        "Generate an integer sequence `[0, 1, ..., n-1]` as `i32[n]`.",
        (["n"], ["int"], "i32[n]"),
    ),
    "one_hot": ComplexBuiltin(
        "one_hot", "construction", "zero_grad",
        "Convert integer indices to one-hot float vectors. one_hot(index, n) → f32[..., n].",
        (["index", "n"], ["i32", "int"], "f32[...]"),
    ),
    "zeros": ComplexBuiltin(
        "zeros", "construction", "zero_grad",
        "Create an array of zeros with the given shape. `zeros(3, 4)` → `f32[3, 4]`.",
        (["dims..."], ["int..."], "f32[...]"),
    ),
    "ones": ComplexBuiltin(
        "ones", "construction", "zero_grad",
        "Create an array of ones with the given shape. `ones(3, 4)` → `f32[3, 4]`.",
        (["dims..."], ["int..."], "f32[...]"),
    ),
    "full": ComplexBuiltin(
        "full", "construction", "zero_grad",
        "Create an array filled with a scalar value. `full(0.5, 3, 4)` → `f32[3, 4]`.",
        (["value", "dims..."], ["f32", "int..."], "f32[...]"),
    ),

    # Convolution / Pooling
    "conv2d": ComplexBuiltin(
        "conv2d", "conv_pool", "has_rule",
        "2D convolution.\n\nInput: `[N, C, H, W]`, Kernel: `[O, C, kH, kW]`.\nPadding: `\"valid\"` or `\"same\"`.",
        (["input", "kernel", "strides", "padding"], ["f32[N,C,H,W]", "f32[O,C,kH,kW]", "(sH,sW)", "str"], "f32[...]"),
    ),
    "max_pool": ComplexBuiltin(
        "max_pool", "conv_pool", "has_rule",
        "2D max pooling.\n\nInput: `[N, C, H, W]`. Reduces spatial dims by window size.",
        (["input", "window", "strides", "padding"], ["f32[N,C,H,W]", "(wH,wW)", "(sH,sW)", "str"], "f32[...]"),
    ),
    "avg_pool": ComplexBuiltin(
        "avg_pool", "conv_pool", "has_rule",
        "2D average pooling.\n\nInput: `[N, C, H, W]`. Reduces spatial dims by window size.",
        (["input", "window", "strides", "padding"], ["f32[N,C,H,W]", "(wH,wW)", "(sH,sW)", "str"], "f32[...]"),
    ),

    # Control flow / gradient
    "stop_gradient": ComplexBuiltin(
        "stop_gradient", "stop_grad", "zero_grad",
        "Identity function that blocks gradient flow backward.",
        (["x"], ["any"], "any"),
    ),
    "where": ComplexBuiltin(
        "where", "where", "has_rule",
        "Element-wise conditional: `where(cond, x, y)`. Fully differentiable.",
        (["cond", "x", "y"], ["bool", "f32", "f32"], "f32"),
    ),

    # Clamp
    "clip": ComplexBuiltin(
        "clip", "clip", "has_rule",
        "Clamp values to range [lo, hi]. Element-wise: max(lo, min(hi, x)).",
        (["x", "lo", "hi"], ["f32", "f32", "f32"], "f32"),
    ),

    # RNG
    "random.key": ComplexBuiltin(
        "random.key", "rng", "zero_grad",
        "Create a PRNG key from an integer seed.",
        (["seed"], ["i32"], "Key"),
    ),
    "random.split": ComplexBuiltin(
        "random.split", "rng", "zero_grad",
        "Split a PRNG key into `n` independent subkeys.",
        (["key", "n"], ["Key", "int"], "Key[n]"),
    ),
    "random.uniform": ComplexBuiltin(
        "random.uniform", "rng", "zero_grad",
        "Sample uniform random values in `[low, high)`.",
        (["key", "low", "high", "dims..."], ["Key", "f32", "f32", "int..."], "f32[...]"),
    ),
    "random.normal": ComplexBuiltin(
        "random.normal", "rng", "zero_grad",
        "Sample normal random values with given mean and std (Box-Muller).",
        (["key", "mean", "std", "dims..."], ["Key", "f32", "f32", "int..."], "f32[...]"),
    ),
    "random.bernoulli": ComplexBuiltin(
        "random.bernoulli", "rng", "zero_grad",
        "Sample Bernoulli random values (0.0 or 1.0) with given probability.",
        (["key", "prob", "dims..."], ["Key", "f32", "int..."], "f32[...]"),
    ),
    "random.categorical": ComplexBuiltin(
        "random.categorical", "rng", "zero_grad",
        "Sample from categorical distribution using Gumbel-max trick.",
        (["key", "logits"], ["Key", "f32[...]"], "i32[...]"),
    ),
    "random.truncated_normal": ComplexBuiltin(
        "random.truncated_normal", "rng", "zero_grad",
        "Sample truncated normal random values clipped to [lo, hi].",
        (["key", "lo", "hi", "dims..."], ["Key", "f32", "f32", "int..."], "f32[...]"),
    ),

    # Utility / inspection
    "isfinite": ComplexBuiltin(
        "isfinite", "construction", "zero_grad",
        "Test element-wise if value is finite (not NaN or Inf). Returns bool.",
        (["x"], ["f32[...]"], "bool[...]"),
    ),
    "zeros_like": ComplexBuiltin(
        "zeros_like", "construction", "zero_grad",
        "Create an array of zeros with the same shape as the input.",
        (["x"], ["f32[...]"], "f32[...]"),
    ),
    "ones_like": ComplexBuiltin(
        "ones_like", "construction", "zero_grad",
        "Create an array of ones with the same shape as the input.",
        (["x"], ["f32[...]"], "f32[...]"),
    ),

    # Callbacks
    "callback": ComplexBuiltin(
        "callback", "callback", "nondiff",
        "Host callback (no-op in compiled code). Useful for debugging.",
        ([], [], "void"),
    ),

    # Two-arg elementwise
    "maximum": ComplexBuiltin(
        "maximum", "two_arg_elementwise", "has_rule",
        "Element-wise maximum of two values.",
        (["x", "y"], ["f32", "f32"], "f32"),
    ),
    "minimum": ComplexBuiltin(
        "minimum", "two_arg_elementwise", "has_rule",
        "Element-wise minimum of two values.",
        (["x", "y"], ["f32", "f32"], "f32"),
    ),
    "pow": ComplexBuiltin(
        "pow", "two_arg_elementwise", "has_rule",
        "Element-wise power: x raised to the power y.",
        (["x", "y"], ["f32", "f32"], "f32"),
    ),
    # Cumulative reductions
    "cumsum": ComplexBuiltin(
        "cumsum", "cumulative", "has_rule",
        "Cumulative sum along an axis. `cumsum(x, axis=0)` computes running totals.",
        (["x", "axis"], ["f32[...]", "int"], "f32[...]"),
    ),
    "cumprod": ComplexBuiltin(
        "cumprod", "cumulative", "has_rule",
        "Cumulative product along an axis. `cumprod(x, axis=0)` computes running products.",
        (["x", "axis"], ["f32[...]", "int"], "f32[...]"),
    ),

    # Sorting
    "sort": ComplexBuiltin(
        "sort", "sorting", "has_rule",
        "Sort array along an axis (ascending). `sort(x)` along last axis, `sort(x, axis=0)` along axis 0.",
        (["x", "axis"], ["f32[...]", "int"], "f32[...]"),
    ),
    "argsort": ComplexBuiltin(
        "argsort", "sorting", "zero_grad",
        "Return indices that would sort the array. `argsort(x)` along last axis. Returns i32.",
        (["x", "axis"], ["f32[...]", "int"], "i32[...]"),
    ),

    # Array manipulation
    "flip": ComplexBuiltin(
        "flip", "array_manip", "has_rule",
        "Reverse array along an axis. `flip(x, 0)` reverses first axis.",
        (["x", "axis"], ["f32[...]", "int"], "f32[...]"),
    ),
    "tril": ComplexBuiltin(
        "tril", "array_manip", "has_rule",
        "Lower triangular matrix. Zeros out elements above the diagonal. Input must be 2D.",
        (["x"], ["f32[N,M]"], "f32[N,M]"),
    ),
    "triu": ComplexBuiltin(
        "triu", "array_manip", "has_rule",
        "Upper triangular matrix. Zeros out elements below the diagonal. Input must be 2D.",
        (["x"], ["f32[N,M]"], "f32[N,M]"),
    ),

    # Two-arg math
    "atan2": ComplexBuiltin(
        "atan2", "two_arg_elementwise", "has_rule",
        "Element-wise two-argument arctangent: atan2(y, x) = arctan(y/x) with correct quadrant.",
        (["y", "x"], ["f32", "f32"], "f32"),
    ),

    # Einsum
    "einsum": ComplexBuiltin(
        "einsum", "einsum", "has_rule",
        "Einstein summation: `einsum(\"ij,jk->ik\", a, b)` for matrix multiply, `einsum(\"ij->ji\", a)` for transpose, etc.",
        (["spec", "a", "b"], ["str", "f32[...]", "f32[...]"], "f32[...]"),
    ),
}

# ---------------------------------------------------------------------------
# Derived convenience sets
# ---------------------------------------------------------------------------

ALL_NAMES: frozenset[str] = frozenset(ELEMENTWISE) | frozenset(COMPLEX)
