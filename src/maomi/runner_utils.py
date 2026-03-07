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


def generate_inputs(fn_sig: FnSignature, seed: int) -> list[np.ndarray]:
    """Generate random input arrays matching the function signature."""
    from .types import ArrayType

    rng = np.random.default_rng(seed)
    inputs = []
    for param_type in fn_sig.param_types:
        dtype = _DTYPE_MAP[param_type.base]
        if isinstance(param_type, ArrayType):
            shape = tuple(param_type.dims)
            if np.issubdtype(dtype, np.floating):
                arr = rng.standard_normal(shape).astype(dtype)
            elif np.issubdtype(dtype, np.integer):
                arr = rng.integers(-10, 10, size=shape, dtype=dtype)
            else:
                arr = rng.choice([True, False], size=shape)
        else:
            # Scalar → 0-d array (matches tensor<f32>)
            if np.issubdtype(dtype, np.floating):
                arr = dtype(rng.standard_normal())
            elif np.issubdtype(dtype, np.integer):
                arr = dtype(rng.integers(-10, 10))
            else:
                arr = dtype(rng.choice([True, False]))
        inputs.append(arr)
    return inputs
