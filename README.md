# Maomi 猫咪 🐱

A pure functional ML language that compiles to XLA via StableHLO. **If it compiles, it's fast.**

LLMs can write working JAX, but getting fully transform-friendly, performance-correct code still often takes review. Maomi narrows the language so the fast, compilable path is also the natural path.

**[Getting Started](docs/getting-started.md)** · **[Language Reference](docs/reference.md)**

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

fn init_weights(seed: i32) -> f32[4, 4] {
    let key: Key = rng_key(seed);
    let keys = rng_split(key, 2);
    rng_normal(keys[0], 0.0, 1.0, 4, 4)
}

fn conv_block(x: f32[1, 3, 8, 8], w: f32[16, 3, 3, 3]) -> f32[1, 16, 2, 2] {
    let h = conv2d(x, w, 2, 1);
    max_pool(h, 2, 2, 2, 2)
}

fn hessian_diag(x: f32[4]) -> f32[4] {
    let loss = sum(x * x * x);
    grad(sum(grad(loss, x)), x)
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
| `grad(expr, var)` | Reverse-mode AD (supports grad-of-grad, structs, scan, indexing, conv2d) |
| `reshape(x, 4, 8)` | Reshape array (element count must match) |
| `concat(a, b)` `concat(a, b, 1)` | Concatenate arrays (optional axis, default 0) |
| `iota(N)` | Integer sequence `[0, 1, ..., N-1]` |
| `conv2d(x, w)` `conv2d(x, w, stride, pad)` | 2D convolution (NCHW layout) |
| `max_pool(x, wh, ww, sh, sw)` | Max pooling |
| `avg_pool(x, wh, ww, sh, sw)` | Average pooling |
| `rng_key(seed)` | Create RNG key from integer seed |
| `rng_split(key, n)` | Split key into n subkeys |
| `rng_uniform(key, lo, hi, d1, d2, ...)` | Uniform random in [lo, hi) |
| `rng_normal(key, mu, std, d1, d2, ...)` | Normal random (Box-Muller) |
| `x[i]` `x[1:3]` `x[:, 0]` `x[-1]` `x[1:]` `x[:-1]` | Array indexing and slicing |
| `table[ids]` | Gather indexing (ids is an integer array) |
| `import math;` | Qualified module import (`math.relu(x)`) |
| `from math import { relu };` | Selective import (`relu(x)`) |
| `import "../lib/nn" as nn;` | Path-based import with alias |
| `callback(args...);` | Host callback (no-op in codegen, ignored by `grad`) |

**Types:** `f32` `f64` `i32` `i64` `bool` `Key` — arrays as `f32[B, 128]` with symbolic or concrete dims. Named structs for grouping data. `Key` is `i32[4]` (RNG key alias).

**Builtins:** `exp` `log` `tanh` `sqrt` `abs` `mean` `sum` `reshape` `concat` `iota` `conv2d` `max_pool` `avg_pool` `rng_key` `rng_split` `rng_uniform` `rng_normal` `callback`

**Operators:** `+` `-` `*` `/` `@` (matmul) `**` (power) `==` `!=` `<` `>` `<=` `>=`

**Indexing:** `x[0]` (single), `x[i]` (dynamic), `x[-1]` (negative), `x[1:3]` (slice), `x[1:]` `x[:3]` `x[:-1]` (open-ended), `x[:, 0]` (multi-axis), `x[0][1]` (chaining), `table[ids]` (gather). Fully differentiable — `grad` propagates through all indexing forms.

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
uv run maomi run examples/cnn.mao --fn conv_forward --seed 7
```

## Status

**v0.7** — 416 tests across lexer, parser, type checker, codegen, AD, modules, indexing, array manipulation, conv/pooling, and RNG. Full pipeline from source to StableHLO.

**Works:** shape-typed arrays, array indexing/slicing (including negative indices, open-ended ranges, gather), `reshape`/`concat`/`transpose`/`iota` builtins, named structs (nested, with functional updates), `scan`/`map`/`grad`, grad-of-grad (higher-order differentiation), scan gradients, struct-shaped gradients, `conv2d`/`max_pool`/`avg_pool` with AD support, deterministic RNG (`rng_key`/`rng_split`/`rng_uniform`/`rng_normal`), import/module system, StableHLO codegen, JAX/XLA execution.

**Limitations:**
- Codegen requires concrete dimensions (symbolic dims type-check but don't compile)
- `map`: elementwise bodies only
- Slice bounds must be integer literals (no dynamic ranges)
- `callback`: compiles but doesn't execute host callbacks yet
- No rank polymorphism
- Literal types fixed: `int` → `i32`, `float` → `f32`
