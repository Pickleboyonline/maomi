# Maomi 猫咪

A pure functional ML language that compiles to XLA via StableHLO. **If it compiles, it's fast.**

LLMs write JAX code that works but is slow — Python loops instead of `scan`/`vmap`, unintentional retracing, host-device transfers, numpy mixed with jnp. No errors, just silent performance loss. Maomi eliminates this: there is no Python to fall back on. The language only expresses operations XLA can optimize. The fast path is the only path.

## Example

```maomi
fn linear(x: f32[32, 128], w: f32[128, 64], b: f32[64]) -> f32[32, 64] {
    x @ w + b
}

fn relu(xs: f32[32, 64]) -> f32[32, 64] {
    map x in xs {
        if x > 0.0 { x } else { 0.0 }
    }
}

fn mse_loss(pred: f32[32], target: f32[32]) -> f32 {
    let diff = pred - target;
    mean(diff * diff)
}

fn train_step(x: f32[4], w: f32[4]) -> f32[4] {
    let loss = mean(x * w);
    grad(loss, w)
}

fn cumsum(xs: f32[10], init: f32) -> f32[10] {
    scan (acc, x) in (init, xs) {
        acc + x
    }
}
```

## Language

| Construct | What it does |
|---|---|
| `fn f(x: f32[B, 128]) -> f32` | Function with shape-typed params |
| `let x = expr;` | Immutable binding |
| `if c { a } else { b }` | Conditional expression (returns a value) |
| `map x in xs { ... }` | Elementwise transform (compiles to vectorized op) |
| `scan (acc, x) in (init, xs) { ... }` | Sequential fold with carried state |
| `grad(expr, var)` | Reverse-mode automatic differentiation |

**Types:** `f32` `f64` `i32` `i64` `bool` — arrays as `f32[B, 128]` with symbolic or concrete dims.

**Builtins:** `mean` `sum` `exp` `log` `tanh` `sqrt` `abs` — elementwise builtins lift to arrays automatically.

**Operators:** `+` `-` `*` `/` `@` (matmul) `**` (power) `==` `!=` `<` `>` `<=` `>=`

## How It Works

```
.mao → lexer → parser → type checker → AD transform → StableHLO → IREE
```

The compiler is written in Python. It emits StableHLO (an MLIR dialect), which can be lowered via IREE for execution on CPU/GPU/TPU.

## Install

Requires Python >= 3.11.

```bash
uv sync                  # compiler only
uv sync --extra run      # with IREE execution backend
```

## Usage

```bash
# Compile to StableHLO (default)
uv run maomi compile examples/mlp.mao --emit stablehlo

# Other emit formats: tokens, ast, types
uv run maomi compile examples/mlp.mao --emit types

# Compile and run (requires IREE)
uv run maomi run examples/grad.mao --fn grad_loss
```

## Status

**v0.3** — 140+ tests across lexer, parser, type checker, codegen, and AD. Full pipeline from source to StableHLO.

**Works:** shape-typed arrays, `scan`/`map`/`grad`, StableHLO codegen, IREE execution for concrete-dimension programs.

**Limitations:**
- Codegen requires concrete dimensions (symbolic dims type-check but don't compile)
- `grad`: no if/else, no non-builtin calls, no grad-of-grad
- `map`: elementwise bodies only
- Effect system parsed but not enforced
- No rank polymorphism
