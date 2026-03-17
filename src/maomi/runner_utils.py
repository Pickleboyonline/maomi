"""Shared utilities for execution backends (JAX runner, TVM runner)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .type_checker import FnSignature

# Maomi base types → numpy dtypes
_DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "bool": np.bool_,
}


def _gen_array_for_type(param_type, rng) -> np.ndarray:
    """Generate a random array for a single leaf type."""
    from .types import ArrayType
    dtype = _DTYPE_MAP[param_type.base]
    if isinstance(param_type, ArrayType):
        shape = tuple(param_type.dims)
        if np.issubdtype(dtype, np.floating):
            return rng.standard_normal(shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            return rng.integers(-10, 10, size=shape, dtype=dtype)
        else:
            return rng.choice([True, False], size=shape)
    else:
        # Scalar → 0-d array
        if np.issubdtype(dtype, np.floating):
            return dtype(rng.standard_normal())
        elif np.issubdtype(dtype, np.integer):
            return dtype(rng.integers(-10, 10))
        else:
            return dtype(rng.choice([True, False]))


def generate_inputs(fn_sig: FnSignature, seed: int) -> list[np.ndarray]:
    """Generate random input arrays matching the function signature.
    Handles StructType and StructArrayType by flattening to leaf tensors."""
    from .api import flatten_type

    rng = np.random.default_rng(seed)
    inputs = []
    for param_type in fn_sig.param_types:
        flat_types = flatten_type(param_type)
        for leaf_type in flat_types:
            inputs.append(_gen_array_for_type(leaf_type, rng))
    return inputs
