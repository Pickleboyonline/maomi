# Maomi 猫咪 🐱

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

fn get_row(matrix: f32[4, 8], i: i32) -> f32[8] {
    matrix[i]
}

fn slice_window(x: f32[10]) -> f32[3] {
    x[2:5]
}
```

## Language

| Construct | What it does |
|---|---|
| `fn f(x: f32[B, 128]) -> f32` | Function with shape-typed params |
| `struct S { x: f32, y: i32 }` | Named struct definition |
| `S { x: 1.0, y: 2 }` | Struct construction |
| `s.x` | Field access (nestable: `s.inner.x`) |
| `s with { x = 1.0 }` | Functional update (nestable: `s with { inner.x = 1.0 }`) |
| `let x = expr;` | Immutable binding |
| `if c { a } else { b }` | Conditional expression (returns a value) |
| `map x in xs { ... }` | Elementwise transform (compiles to vectorized op) |
| `scan (acc, x) in (init, xs) { ... }` | Sequential fold with carried state |
| `grad(expr, var)` | Reverse-mode AD (supports structs, scan, indexing) |
| `x[i]` `x[1:3]` `x[:, 0]` | Array indexing and slicing |
| `import math;` | Qualified module import (`math.relu(x)`) |
| `from math import { relu };` | Selective import (`relu(x)`) |
| `import "../lib/nn" as nn;` | Path-based import with alias |
| `callback(args...);` | Host callback (no-op in codegen, ignored by `grad`) |

**Types:** `f32` `f64` `i32` `i64` `bool` — arrays as `f32[B, 128]` with symbolic or concrete dims. Named structs for grouping data.

**Builtins:** `mean` `sum` `exp` `log` `tanh` `sqrt` `abs` `callback` — elementwise builtins lift to arrays automatically.

**Operators:** `+` `-` `*` `/` `@` (matmul) `**` (power) `==` `!=` `<` `>` `<=` `>=`

**Indexing:** `x[0]` (single), `x[i]` (dynamic), `x[1:3]` (slice, static bounds), `x[:, 0]` (multi-axis), `x[0][1]` (chaining). Fully differentiable — `grad` propagates through indexing via `dynamic_update_slice`.

## How It Works

```
.mao → lexer → parser → resolver → type checker → AD transform → StableHLO → JAX/XLA
```

The compiler is written in Python. It emits StableHLO (an MLIR dialect), which is executed via JAX's XLA backend. The resolver handles module imports — loading, prefixing, and merging functions from other `.mao` files.

## Install

Requires Python >= 3.11.

```bash
uv sync                  # compiler only
uv sync --extra run      # with JAX execution backend
```

## Usage

```bash
# Compile to StableHLO (default)
uv run maomi compile examples/mlp.mao --emit stablehlo

# Other emit formats: tokens, ast, types
uv run maomi compile examples/mlp.mao --emit types

# Compile and run (requires JAX)
uv run maomi run examples/grad.mao --fn grad_loss
```

## Status

**v0.5** — 242 tests across lexer, parser, type checker, codegen, AD, modules, and indexing. Full pipeline from source to StableHLO.

**Works:** shape-typed arrays, array indexing/slicing, named structs (nested, with functional updates), `scan`/`map`/`grad`, scan gradients, struct-shaped gradients, import/module system, StableHLO codegen, JAX/XLA execution for concrete-dimension programs.

**Limitations:**
- Codegen requires concrete dimensions (symbolic dims type-check but don't compile)
- `grad`: no grad-of-grad
- `map`: elementwise bodies only
- Slice bounds must be integer literals (no dynamic ranges)
- No negative indices or open-ended ranges (`x[1:]`, `x[-1]`)
- `callback`: compiles but doesn't execute host callbacks yet (IREE outfeed integration pending)
- No rank polymorphism
