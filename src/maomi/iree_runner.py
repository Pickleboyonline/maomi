"""IREE-based execution backend for Maomi StableHLO modules.

All IREE-dependent code lives here. This module is imported lazily
only when the `run` subcommand is invoked, so the core compiler
stays zero-dependency.

Requires: iree-base-compiler, iree-base-runtime  (install with `uv sync --extra run`)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from iree.compiler import compile_str
from iree import runtime as ireert

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


def run_stablehlo(
    mlir_text: str,
    fn_name: str,
    fn_sig: FnSignature,
    seed: int = 42,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Compile and execute a StableHLO module, returning (inputs, output).

    Args:
        mlir_text: MLIR text from Maomi codegen (contains `module { ... }`)
        fn_name: Name of the function to execute
        fn_sig: Function signature (param names, types, return type)
        seed: Random seed for reproducible input generation

    Returns:
        (inputs, output) where inputs is a list of numpy arrays and
        output is the numpy result.
    """
    # Add module sym_name required by IREE
    mlir_text = _prepare_module(mlir_text)

    # Compile StableHLO → IREE flatbuffer
    compiled = compile_str(mlir_text, target_backends=["llvm-cpu"], input_type="stablehlo")

    # Load and execute
    config = ireert.Config("local-task")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled)
    ctx.add_vm_module(vm_module)

    fn = ctx.modules.main[fn_name]
    inputs = generate_inputs(fn_sig, seed)
    result = fn(*inputs)
    output = result.to_host()

    return inputs, output


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


def _prepare_module(mlir_text: str) -> str:
    """Add module sym_name attribute required by IREE."""
    return mlir_text.replace("module {", "module @main {", 1)
