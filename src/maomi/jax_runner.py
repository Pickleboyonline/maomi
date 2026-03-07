"""JAX-based execution backend for Maomi StableHLO modules.

All JAX-dependent code lives here. This module is imported lazily
only when the `run` subcommand is invoked, so the core compiler
stays zero-dependency.

Requires: jax[cpu]>=0.4.20  (install with `uv sync --extra run`)

Uses JAX internal APIs (tested with JAX 0.9.1):
  - jax._src.interpreters.mlir.make_ir_context
  - jax._src.lib.mlir.ir.Module.parse
  - jax._src.compiler.backend_compile_and_load
  - jax._src.xla_bridge.get_backend
  - jax._src.lib.xla_client.{CompileOptions, DeviceList}
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import jax
import numpy as np
from jax._src import compiler, xla_bridge as xb
from jax._src.interpreters import mlir
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir

if TYPE_CHECKING:
    from .type_checker import FnSignature
    from .types import MaomiType

from .runner_utils import generate_inputs  # noqa: F401 — re-export for backwards compat


def run_stablehlo(
    mlir_text: str,
    fn_name: str,
    fn_sig: FnSignature,
    seed: int = 42,
    inputs: list[np.ndarray] | None = None,
    host_callbacks: list | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Compile and execute a StableHLO module, returning (inputs, output).

    Args:
        mlir_text: MLIR text from Maomi codegen (contains `module { ... }`)
        fn_name: Name of the function to execute
        fn_sig: Function signature (param names, types, return type)
        seed: Random seed for reproducible input generation
        inputs: Optional pre-built inputs (overrides seed-based generation)
        host_callbacks: Optional list of Python callables for callback() builtins

    Returns:
        (inputs, output) where inputs is a list of numpy arrays and
        output is the numpy result.
    """
    # Ensure JAX is initialized (registers compiler factories)
    jax.devices()

    # Prepare module: rename target function to @main, add module sym_name
    mlir_text = _prepare_module(mlir_text, fn_name)

    # Parse and compile
    ctx = mlir.make_ir_context()
    with ctx:
        module = ir.Module.parse(mlir_text)

    backend = xb.get_backend()
    executable = compiler.backend_compile_and_load(
        backend,
        module,
        xc.DeviceList(tuple(jax.devices())),
        xc.CompileOptions(),
        host_callbacks or [],
    )

    # Generate inputs if not provided
    if inputs is None:
        inputs = generate_inputs(fn_sig, seed)

    # Execute
    buffers = [backend.buffer_from_pyval(arr) for arr in inputs]
    results = executable.execute(buffers)
    output = np.asarray(results[0])

    return inputs, output


def _prepare_module(mlir_text: str, fn_name: str) -> str:
    """Rename target function to @main and set module sym_name.

    XLA expects the entry point to be named @main in a module with
    sym_name @main. Other functions in the module (builtins, helpers)
    are left as-is, but any calls to the renamed function are updated.
    """
    # Add module sym_name
    mlir_text = mlir_text.replace("module {", "module @main {", 1)

    if fn_name == "main":
        return mlir_text

    # Rename the target function definition and any calls to it
    # Use word-boundary matching to avoid renaming substrings
    mlir_text = re.sub(
        rf"@{re.escape(fn_name)}\b", "@main", mlir_text
    )

    return mlir_text
