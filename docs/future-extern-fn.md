# `extern fn` — Host Function Interface (Design Notes)

## Summary

`extern fn` lets Maomi call host functions (C, Rust, TypeScript, etc.) and get values back. Unlike `callback()` which is fire-and-forget, `extern fn` returns data into the compiled program.

```maomi
extern fn load_batch(step: i32) -> { x: f32[32, 784], y: f32[32, 10] };
extern fn env_step(action: f32[4]) -> { obs: f32[64], reward: f32, done: bool };

fn main(seed: i32) {
    fold (state, step) in (init, range(0, 1000)) {
        let batch = load_batch(step);
        let g = grad(loss(state, batch), state);
        sgd_update(state, g, 0.01)
    };
}
```

## Why

Maomi currently depends on Python as glue: the Python API calls compiled functions and passes data in. `extern fn` inverts this — Maomi is the host, external code is the plugin. This eliminates the Python dependency for training loops.

## Architecture

```
Maomi program (compiled, runs on accelerator)
  │
  ├── extern fn load_batch(...)   → host function (C/Rust/TS)
  ├── extern fn env_step(...)     → host function
  ├── extern fn save(...)         → host function
  │
  └── Everything else: grad, scan, loss — stays compiled
```

## Implementation Path

### What exists today

`callback()` already works end-to-end:
- Parser recognizes it as a builtin call
- Type checker treats it as `void` return, non-differentiable
- StableHLO codegen emits `stablehlo.custom_call` with `call_target_name = "xla_ffi_python_cpu_callback"`, `result_layouts = []`
- JAX runner passes `host_callbacks` list to `compiler.backend_compile_and_load()`
- Each callback is indexed; `mhlo.backend_config = {index = N}` maps to `host_callbacks[N]`

### What `extern fn` adds

The only fundamental difference from `callback` is **return values**.

| | `callback` | `extern fn` |
|---|---|---|
| Syntax | `callback(x, y)` (builtin) | `extern fn name(x: T) -> U;` (declaration) |
| Return | `void` / `result_layouts = []` | Declared type / `result_layouts = [layout]` |
| Body | N/A (builtin) | N/A (host-implemented) |
| AD | Non-differentiable | Non-differentiable |
| Side effects | `has_side_effect = true` | Configurable (`pure` vs impure) |

### Files to change

1. **`tokens.py`** — Add `EXTERN` keyword
2. **`parser.py`** — Parse `extern fn name(args) -> T;` (no body)
3. **`ast_nodes.py`** — `FnDef.body` becomes `Block | None`, add `extern: bool = False`
4. **`type_checker.py`** — Register extern in `fn_table`, skip body check, mark non-differentiable
5. **`codegen/stablehlo/core.py`** — Emit `stablehlo.custom_call` with non-empty `result_layouts`
6. **`codegen/relax/core.py`** — Emit `call_packed` with return value
7. **`jax_runner.py`** — Accept extern host functions alongside callbacks
8. **`api.py`** — Add `_externs` parameter

### StableHLO codegen detail

Current callback emits:
```mlir
"stablehlo.custom_call"(%arg) {
    call_target_name = "xla_ffi_python_cpu_callback",
    has_side_effect = true,
    api_version = 1 : i32,
    mhlo.backend_config = {index = 0 : ui64},
    operand_layouts = [dense<0> : tensor<1xindex>],
    result_layouts = []
} : (tensor<f32>) -> ()
```

Extern fn would emit:
```mlir
%result = "stablehlo.custom_call"(%arg) {
    call_target_name = "xla_ffi_python_cpu_callback",
    has_side_effect = true,
    api_version = 1 : i32,
    mhlo.backend_config = {index = 0 : ui64},
    operand_layouts = [dense<0> : tensor<1xindex>],
    result_layouts = [dense<[1, 0]> : tensor<2xindex>]
} : (tensor<f32>) -> tensor<32x784xf32>
```

### Gotchas to resolve

**1. `api_version`**: Maomi uses `api_version = 1` (legacy). JAX's modern FFI uses `api_version = 4`. Return values may behave differently across versions. Need to test which version supports Python callbacks with return values.

**2. Callback target**: `xla_ffi_python_cpu_callback` may not support return values. Alternative: use JAX's `pure_callback` / `io_callback` lowering pattern, which emits the correct custom_call for callbacks with returns. Or use `jax.pure_callback` directly in the runner.

**3. CPU-only**: Host callbacks always run on CPU. For GPU backends, data copies device→host→device around the call. Acceptable for I/O-bound calls (data loading, env steps), expensive for compute.

**4. Blocking**: The custom_call blocks XLA execution until the host function returns. For async data loading, the user can split into `start_load` / `wait_load` pattern:
```maomi
extern fn start_load(step: i32) -> i64;        // returns handle, non-blocking
extern fn wait_load(handle: i64) -> Batch;     // blocks until ready

fold (state, step) in (init, range(0, steps)) {
    let batch = wait_load(state.pending);       // data already loaded
    let next = start_load(step + 1);            // kick off next load
    let g = grad(loss(state.params, batch), state.params);
    TrainState { params: sgd_update(state.params, g, lr), pending: next }
}
```

**5. No gradient**: `extern fn` is opaque to AD. Calling `grad(loss(...), p)` where `loss` calls an extern fn that doesn't touch `p` is fine. Trying to differentiate through an extern fn should be a compile error.

**6. Struct returns**: Extern fn returning a struct needs to be lowered to a tuple of arrays in StableHLO (which doesn't have structs). The codegen already handles struct→tuple conversion for regular functions.

### Relax / TVM path

Relax recently added `call_packed` with return values and callback-as-argument support. The Maomi Relax codegen would emit:
```python
# Relax IR (conceptual)
result = R.call_packed("load_batch", step, sinfo_args=[R.Tensor((32, 784), "float32")])
```

This is simpler than the StableHLO path since TVM's VM natively supports calling host functions mid-execution.

### Python API usage

```python
import maomi
import numpy as np

mod = maomi.compile("train.mao")

def load_batch(step):
    # Load from disk, return numpy array
    return {"x": np.random.randn(32, 784).astype(np.float32),
            "y": np.random.randn(32, 10).astype(np.float32)}

result = mod.main(seed=42, _externs={"load_batch": load_batch})
```

### Other host languages

Since extern fn compiles to `stablehlo.custom_call` or `call_packed`, any language that can register a C-ABI function works:
- **C/Rust/Zig**: Register via XLA FFI C API or TVM packed function registry
- **Bun/TypeScript**: Use `bun:ffi` to expose C-compatible functions
- **Swift**: Native C interop

### Related features to consider later

- **`extern fn` with `pure` annotation**: `extern pure fn hash(x: f32[N]) -> f32` — allows XLA to optimize (CSE, reorder)
- **Array save/load**: Could be implemented as stdlib extern fns rather than language primitives
- **CLI args**: `fn main(lr: f32, steps: i32)` with `maomi run --lr 0.001 --steps 10000`
- **Timer**: `extern fn clock() -> f64` backed by `clock_gettime`
