"""TVM-based execution backend for Maomi Relax IR modules.

Compiles a TVM IRModule to a target (CPU/Metal/CUDA) and executes it
via TVM's VirtualMachine. This module is imported lazily only when
the `run` subcommand is invoked with `--backend relax`.

Requires: mlc-ai-nightly-cpu  (install with `uv sync --extra tvm`)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tvm
from tvm import relax
from tvm.s_tir.dlight import ApplyDefaultSchedule, gpu

from .runner_utils import generate_inputs

if TYPE_CHECKING:
    from .type_checker import FnSignature


def run_relax(
    ir_mod: tvm.IRModule,
    fn_name: str,
    fn_sig: FnSignature,
    target: str = "llvm",
    seed: int = 42,
    inputs: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Compile and execute a Relax IRModule, returning (inputs, output).

    Args:
        ir_mod: TVM IRModule from RelaxCodegen.generate()
        fn_name: Name of the function to execute
        fn_sig: Function signature (param names, types, return type)
        target: Compilation target — "llvm" (CPU), "metal" (Apple GPU), "cuda"
        seed: Random seed for reproducible input generation
        inputs: Optional pre-built inputs (overrides seed-based generation)

    Returns:
        (inputs, output) where inputs is a list of numpy arrays and
        output is the numpy result.
    """
    target_obj, dev = _resolve_target(target)

    # Apply target-specific passes
    with target_obj:
        mod = relax.transform.LegalizeOps()(ir_mod)
        if _is_gpu(target):
            mod = ApplyDefaultSchedule(gpu.Matmul(), gpu.Fallback())(mod)

    ex = relax.build(mod, target=target_obj)
    vm = relax.VirtualMachine(ex, dev)

    if inputs is None:
        inputs = generate_inputs(fn_sig, seed)

    tvm_inputs = [_np_to_tvm(arr, dev) for arr in inputs]
    output = vm[fn_name](*tvm_inputs)
    return inputs, output.numpy()


def _resolve_target(target: str) -> tuple[tvm.target.Target, tvm.runtime.Device]:
    if target == "metal":
        return (
            tvm.target.Target({"kind": "metal", "host": {"kind": "llvm"}}),
            tvm.metal(),
        )
    elif target == "cuda":
        return (
            tvm.target.Target({"kind": "cuda", "host": {"kind": "llvm"}}),
            tvm.cuda(),
        )
    else:
        return tvm.target.Target("llvm"), tvm.cpu()


def _is_gpu(target: str) -> bool:
    return target in ("metal", "cuda")


def _np_to_tvm(arr: np.ndarray, dev: tvm.runtime.Device) -> tvm.runtime.Tensor:
    t = tvm.runtime.empty(arr.shape, dtype=str(arr.dtype), device=dev)
    t.copyfrom(arr)
    return t
