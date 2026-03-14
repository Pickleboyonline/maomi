# Maomi 猫咪 🐱

A pure functional ML language that compiles to XLA via StableHLO. **If it compiles, it's fast.**

I made this so I can vibe code fast jittable jax code in 1 shot. In my experience, LLMs can typically write working JAX, but getting fully transform-friendly, performance-correct code still often takes review. Maomi narrows the language so the fast, compilable path is also the natural path and only algorithmic performance is left for review.

Also, if you like Jax but want a Rust like language for it instead of dealing with Python/Jax boundaries -- this project is for you.

Maomi also supports bounded AD while loops:

```maomi
fn converge_diff(x: f32) -> f32 {
    let result = while s in x limit 100 { s > 0.01 } do {
        s * 0.5
    };
    grad(result, x)
}
```

Memory is still preallocated for gradients like a jax's `scan`, but you can early stop computation with compiled AD.


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
    let { value, gradient } = value_and_grad(mean(x * w), w);
    callback(value);
    gradient
}

fn cumsum(xs: f32[10], init: f32) -> f32[10] {
    scan (acc, x) in (init, xs) {
        acc + x
    }
}

fn total(xs: f32[10]) -> f32 {
    fold (acc, x) in (0.0, xs) { acc + x }
}

fn init_weights(seed: i32) -> f32[4, 4] {
    let key: Key = random.key(seed);
    let keys = random.split(key, 2);
    random.normal(keys[0], 0.0, 1.0, 4, 4)
}

fn conv_block(x: f32[1, 3, 8, 8], w: f32[16, 3, 3, 3]) -> f32[1, 16, 2, 2] {
    let h = conv2d(x, w, 2, 1);
    max_pool(h, 2, 2, 2, 2)
}

fn softmax(x: f32[32, 128]) -> f32[32, 128] {
    let e = exp(x - max(x, 1, keepdims=true));
    e / sum(e, 1, keepdims=true)
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
| `fn f(x: f32[..], comptime axis: i32) -> f32[..]` | Rank-polymorphic with compile-time param |
| `struct S { x: f32, y: i32 }` | Named struct definition |
| `S { x: 1.0, y: 2 }` | Struct construction |
| `s.x` | Field access (nestable: `s.inner.x`) |
| `s with { x = 1.0 }` | Functional update (nestable: `s with { inner.x = 1.0 }`) |
| `let { x, y } = s;` | Struct destructuring |
| `let { x: alias } = s;` | Destructuring with rebinding |
| `let x = expr;` | Immutable binding |
| `cast(x, f32)` | Type conversion (preserves shape) |
| `if c { a } else { b }` | Conditional expression (returns a value) |
| `map x in xs { ... }` | General map/vmap (compiles to vectorized op) |
| `scan (acc, x) in (init, xs) { ... }` | Sequential fold with carried state (stacks results) |
| `fold (acc, x) in (init, xs) { ... }` | Sequential fold returning final value only (supports struct carries) |
| `while s in x limit N { cond } do { body }` | Bounded while loop (differentiable with limit) |
| `grad(expr, var)` | Reverse-mode AD (supports grad-of-grad, structs, scan, indexing, conv2d) |
| `value_and_grad(expr, var)` | Forward value + gradient in one pass |
| `f(x, axis=1)` | Named arguments at call sites |

**Builtins — Math:** `exp` `log` `tanh` `sqrt` `abs` `cos` `sin` `tan` `acos` `asin` `atan` `sinh` `cosh` `asinh` `acosh` `atanh` `sigmoid` `relu` `gelu` `silu` `softplus` `log1p` `expm1` `log2` `log10` `exp2` `square` `rsqrt` `reciprocal` `neg` `sign` `floor` `ceil`

**Builtins — Reductions:** `sum` `mean` `max` `min` `logsumexp` `argmax` `argmin` `cumsum` `cumprod`

**Builtins — Shape:** `reshape` `concat` `transpose` `stack` `pad` `expand_dims` `squeeze` `broadcast_to` `iota` `zeros` `ones` `full` `one_hot` `zeros_like` `ones_like`

**Builtins — Other:** `where` `stop_gradient` `clip` `maximum` `minimum` `pow` `atan2` `sort` `argsort` `einsum` `conv2d` `max_pool` `avg_pool` `callback` `isfinite` `cast`

**RNG:** `random.key` `random.split` `random.uniform` `random.normal` `random.bernoulli` `random.categorical` `random.truncated_normal`

**Types:** `f32` `f64` `bf16` `i32` `i64` `bool` `Key` — arrays as `f32[B, 128]` with symbolic or concrete dims. `f32[..]` for rank-polymorphic. Named structs for grouping data.

**Operators:** `+` `-` `*` `/` `@` (matmul) `**` (power) `==` `!=` `<` `>` `<=` `>=` `and` `or` `not`

**Broadcasting:** Numpy-style broadcasting including size-1 dimensions: `f32[3, 1] * f32[3, 4]` → `f32[3, 4]`

**Indexing:** `x[0]` (single), `x[i]` (dynamic), `x[-1]` (negative), `x[1:3]` (slice), `x[1:]` `x[:3]` `x[:-1]` (open-ended), `x[:, 0]` (multi-axis), `x[0][1]` (chaining), `table[ids]` (gather). Fully differentiable — `grad` propagates through all indexing forms.

**Modules:** `import math;` `from math import { relu };` `import "../lib/nn" as nn;`

**Stdlib:** `nn` (relu, sigmoid, softmax, log_softmax, leaky_relu, elu, selu, mish, layer_norm) · `optim` (sgd_update, adam_update, linear_decay, cosine_decay)

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

**Python API:**

```python
import maomi
module = maomi.compile("model.mao")
result = module.forward(input_array)
```

## Status

**v0.9** — 1373 tests across lexer, parser, type checker, codegen, AD, modules, indexing, array manipulation, conv/pooling, RNG, broadcasting, einsum, sorting, shape ops, and numerical AD verification. Full pipeline from source to StableHLO.

**Works:** shape-typed arrays, rank polymorphism (`f32[..]`), compile-time params (`comptime`), named arguments, struct destructuring, array indexing/slicing (negative, open-ended, gather), `reshape`/`concat`/`transpose`/`stack`/`pad`/`expand_dims`/`squeeze`/`broadcast_to`/`iota`, 32+ elementwise builtins (trig, hyperbolic, activations), axis-specific reductions with `keepdims`, `logsumexp`, `cumsum`/`cumprod`, `sort`/`argsort`, `einsum`, `clip`, `maximum`/`minimum`, size-1 broadcasting, `where`/`stop_gradient`, named structs (nested, functional updates, destructuring, arithmetic, generics), `scan`/`fold`/`map`/`grad`/`value_and_grad`, grad-of-grad, struct-shaped gradients, `conv2d`/`max_pool`/`avg_pool` with AD, deterministic RNG (key/split/uniform/normal/bernoulli/categorical/truncated_normal), `callback` with JAX FFI execution, import/module system with stdlib (nn, optim), Python API, StableHLO codegen, JAX/XLA execution, LSP with full IDE support.

**Limitations:**
- Codegen requires concrete dimensions (symbolic dims type-check but don't compile)
- Slice bounds must be integer literals (no dynamic ranges)
- Fixed literal types: `int` → `i32`, `float` → `f32`
- No visibility modifiers (all functions importable)
- Grad-of-grad not yet supported through conv2d/pool/while
