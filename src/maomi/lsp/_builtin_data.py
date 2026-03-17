from __future__ import annotations

from ..builtins import ELEMENTWISE as _EW_REGISTRY, COMPLEX as _CX_REGISTRY

_KEYWORDS = [
    "fn", "let", "if", "else", "scan", "map", "grad", "value_and_grad", "cast", "fold",
    "struct", "with", "import", "from", "in", "true", "false",
    "while", "do", "limit", "type",
]

_TYPE_NAMES = ["f32", "f64", "bf16", "i32", "i64", "bool"]

# Derive from central registry — stays in sync automatically
_BUILTINS = sorted(set(_EW_REGISTRY.keys()) | set(_CX_REGISTRY.keys()))

_BUILTIN_SET = set(_BUILTINS)

_BUILTIN_NAMESPACES: dict[str, list[str]] = {
    "random": ["key", "split", "uniform", "normal", "bernoulli", "categorical", "truncated_normal"],
}

# Signature and doc data derived from registry
_BUILTIN_SIGNATURES: dict[str, tuple[list[str], list[str], str]] = {}
_BUILTIN_DOCS: dict[str, str] = {}

for _name, _b in _EW_REGISTRY.items():
    _BUILTIN_SIGNATURES[_name] = (["x"], ["f32"], "f32")
    _BUILTIN_DOCS[_name] = _b.doc
for _name, _b in _CX_REGISTRY.items():
    _BUILTIN_SIGNATURES[_name] = _b.lsp_params
    _BUILTIN_DOCS[_name] = _b.doc

# config() — compile-time constant from TOML (not in builtins registry)
_BUILTIN_SET.add("config")
_BUILTINS.append("config")
_BUILTIN_SIGNATURES["config"] = (["key"], ["str"], "f32 | i32 | str")
_BUILTIN_DOCS["config"] = "Read a compile-time constant from the config TOML file. Usage: config(\"lr\")"

# ---------------------------------------------------------------------------
# Category labels for completion detail (LSP-only, not a language feature)
# ---------------------------------------------------------------------------
_BUILTIN_CATEGORIES: dict[str, str] = {}

_ACTIVATION_FNS = {"relu", "gelu", "silu", "sigmoid", "softplus"}
_TRIG_FNS = {"sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
             "asinh", "acosh", "atanh"}

for _name in _EW_REGISTRY:
    if _name in _ACTIVATION_FNS:
        _BUILTIN_CATEGORIES[_name] = "activation"
    elif _name in _TRIG_FNS:
        _BUILTIN_CATEGORIES[_name] = "trig"
    else:
        _BUILTIN_CATEGORIES[_name] = "math"

_COMPLEX_CATEGORY_LABELS = {
    "reduction": "reduction",
    "bool_reduction": "reduction",
    "argmax": "reduction",
    "cumulative": "reduction",
    "shape": "shape",
    "array_manip": "shape",
    "construction": "creation",
    "conv_pool": "conv/pool",
    "rng": "random",
    "two_arg_elementwise": "math",
    "sorting": "sorting",
    "linalg": "linalg",
    "einsum": "linalg",
    "stop_grad": "gradient",
    "where": "control",
    "clip": "math",
    "callback": "debug",
}

for _name, _b in _CX_REGISTRY.items():
    _BUILTIN_CATEGORIES[_name] = _COMPLEX_CATEGORY_LABELS.get(_b.category, "builtin")

_BUILTIN_CATEGORIES["config"] = "config"
