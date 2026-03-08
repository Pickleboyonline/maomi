from __future__ import annotations

from ..builtins import ELEMENTWISE as _EW_REGISTRY, COMPLEX as _CX_REGISTRY

_KEYWORDS = [
    "fn", "let", "if", "else", "scan", "map", "grad", "cast", "fold",
    "struct", "with", "import", "from", "in", "true", "false",
    "while", "do", "limit",
]

_TYPE_NAMES = ["f32", "f64", "i32", "i64", "bool"]

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
