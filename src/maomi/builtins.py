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


# ---------------------------------------------------------------------------
# Compound elementwise codegen functions
# ---------------------------------------------------------------------------
# For builtins without a single StableHLO op, these emit compound StableHLO.
# Each takes (codegen_ctx, call_expr, env) -> SSA variable name.
# Uses the codegen's internal API: _gen_expr, _type_of, _fresh, _emit.
# Uses the module-level _mlir_type function from codegen_stablehlo.

def _codegen_sigmoid(codegen: Any, expr: Any, env: dict[str, str]) -> str:
    """sigmoid(x) = 1 / (1 + exp(-x))"""
    from .codegen.stablehlo.utils import _mlir_type
    from .types import StructType

    arg = codegen._gen_expr(expr.args[0], env)
    result_type = codegen._type_of(expr)

    if isinstance(result_type, StructType):
        return codegen._gen_struct_compound_elementwise(
            "sigmoid", _codegen_sigmoid_inner, arg, result_type,
        )

    mlir_t = _mlir_type(result_type)
    return _codegen_sigmoid_inner(codegen, arg, mlir_t)


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


def _codegen_log2(codegen: Any, expr: Any, env: dict[str, str]) -> str:
    """log2(x) = log(x) / ln(2)"""
    from .codegen.stablehlo.utils import _mlir_type
    from .types import StructType

    arg = codegen._gen_expr(expr.args[0], env)
    result_type = codegen._type_of(expr)

    if isinstance(result_type, StructType):
        return codegen._gen_struct_compound_elementwise(
            "log2", _codegen_log2_inner, arg, result_type,
        )

    mlir_t = _mlir_type(result_type)
    return _codegen_log2_inner(codegen, arg, mlir_t)


def _codegen_log2_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    log_x = codegen._fresh()
    codegen._emit(f"{log_x} = stablehlo.log {arg_ssa} : {mlir_t}")
    ln2 = codegen._fresh()
    codegen._emit(f"{ln2} = stablehlo.constant dense<6.931470e-01> : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.divide {log_x}, {ln2} : {mlir_t}")
    return result


def _codegen_reciprocal(codegen: Any, expr: Any, env: dict[str, str]) -> str:
    """reciprocal(x) = 1 / x"""
    from .codegen.stablehlo.utils import _mlir_type
    from .types import StructType

    arg = codegen._gen_expr(expr.args[0], env)
    result_type = codegen._type_of(expr)

    if isinstance(result_type, StructType):
        return codegen._gen_struct_compound_elementwise(
            "reciprocal", _codegen_reciprocal_inner, arg, result_type,
        )

    mlir_t = _mlir_type(result_type)
    return _codegen_reciprocal_inner(codegen, arg, mlir_t)


def _codegen_reciprocal_inner(codegen: Any, arg_ssa: str, mlir_t: str) -> str:
    one = codegen._fresh()
    codegen._emit(f"{one} = stablehlo.constant dense<1.000000e+00> : {mlir_t}")
    result = codegen._fresh()
    codegen._emit(f"{result} = stablehlo.divide {one}, {arg_ssa} : {mlir_t}")
    return result


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

    # Array construction
    "iota": ComplexBuiltin(
        "iota", "construction", "zero_grad",
        "Generate an integer sequence `[0, 1, ..., n-1]` as `i32[n]`.",
        (["n"], ["int"], "i32[n]"),
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

    # Callbacks
    "callback": ComplexBuiltin(
        "callback", "callback", "nondiff",
        "Host callback (no-op in compiled code). Useful for debugging.",
        ([], [], "void"),
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
