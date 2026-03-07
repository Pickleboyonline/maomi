# Maomi Language Reference

Complete reference for the Maomi programming language.

## Table of Contents

- [Types](#types)
- [Syntax](#syntax)
- [Operators](#operators)
- [Builtins — Elementwise](#builtins--elementwise)
- [Builtins — Reductions](#builtins--reductions)
- [Builtins — Shape & Array](#builtins--shape--array)
- [Builtins — Neural Network](#builtins--neural-network)
- [Builtins — RNG](#builtins--rng)
- [Array Indexing](#array-indexing)
- [Structs](#structs)
- [Control Flow](#control-flow)
- [Automatic Differentiation](#automatic-differentiation)
- [Module System](#module-system)
- [CLI](#cli)
- [Compilation Pipeline](#compilation-pipeline)
- [Known Limitations](#known-limitations)

---

## Types

### Scalar Types

| Type | Description |
|------|-------------|
| `f32` | 32-bit float |
| `f64` | 64-bit float |
| `i32` | 32-bit integer |
| `i64` | 64-bit integer |
| `bool` | Boolean |

Literal inference: integer literals (`42`) are `i32`, float literals (`3.14`, `1e-3`) are `f32`, boolean literals (`true`, `false`) are `bool`.

### Array Types

Arrays have a base scalar type and a shape. Dimensions can be concrete integers or symbolic names.

```maomi
f32[4]          // 1D float array of length 4
i32[3, 4]       // 2D integer array, 3 rows, 4 columns
f32[B, 128]     // symbolic batch dimension B, concrete feature dimension 128
bool[H, W, C]   // 3D boolean array with symbolic dims
```

Symbolic dimensions are valid in type annotations for type checking, but codegen requires all dimensions to be concrete. This is by design — XLA (the compilation backend) requires concrete shapes to generate optimized machine code. Symbolic dims let you express shape relationships (e.g. that two parameters share a batch dimension) that the type checker verifies, but they must resolve to concrete integers by the time code is generated. This matches how JAX works: `jax.jit` recompiles for each new input shape.

### Key Type

`Key` is a compiler-level type alias for `i32[4]`. It exists for readability in RNG-related code.

```maomi
fn init(key: Key) -> f32[4, 4] {
    random.normal(key, 0.0, 1.0, 4, 4)
}
```

`Key` and `i32[4]` are interchangeable — using `Key` is optional.

### Struct Types

Named product types defined with the `struct` keyword. See [Structs](#structs) for details.

---

## Syntax

### Comments

Line comments start with `//` and are ignored by the compiler.

```maomi
// This is a comment
fn f(x: f32) -> f32 { x } // inline comment
```

### Doc Comments

Doc comments start with `///` and attach to the next function or struct definition. They are preserved in the AST and surfaced by the LSP (hover, completion, signature help).

```maomi
/// Compute the ReLU activation.
/// Returns max(0, x) element-wise.
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// A 2D point in space.
struct Point { x: f32, y: f32 }
```

### Functions

Functions are the top-level unit. Parameters and return types are annotated. The last expression in the body is the return value.

```maomi
fn add(a: f32, b: f32) -> f32 {
    a + b
}

fn linear(x: f32[32, 128], w: f32[128, 64], b: f32[64]) -> f32[32, 64] {
    x @ w + b
}
```

### Let Bindings

Immutable bindings with optional type annotation.

```maomi
let x = 1.0;
let y: f32[4] = x * w;
```

### Blocks

Braces delimit blocks. Statements are separated by semicolons. The last expression (without a trailing semicolon) is the block's value.

```maomi
fn example(x: f32) -> f32 {
    let a = x * x;
    let b = a + 1.0;
    b * 2.0
}
```

### Literals

```maomi
42        // i32
3.14      // f32
1e-3      // f32
true      // bool
false     // bool
```

---

## Operators

### Arithmetic

| Operator | Operation | Works on |
|----------|-----------|----------|
| `+` | Addition | scalars, arrays (elementwise) |
| `-` | Subtraction | scalars, arrays (elementwise) |
| `*` | Multiplication | scalars, arrays (elementwise) |
| `/` | Division | scalars, arrays (elementwise) |
| `**` | Power | scalars, arrays (elementwise) |
| `@` | Matrix multiply | 2D arrays |

### Unary

| Operator | Operation |
|----------|-----------|
| `-x` | Negation |
| `+x` | Unary plus (identity) |

### Comparison

| Operator | Operation |
|----------|-----------|
| `==` | Equal |
| `!=` | Not equal |
| `<` | Less than |
| `>` | Greater than |
| `<=` | Less or equal |
| `>=` | Greater or equal |

Comparisons return `bool` (scalar) or `bool[...]` (elementwise on arrays).

### Broadcasting

Binary operations support numpy-style broadcasting:

- **Scalar + Array**: `f32 * f32[3, 4]` → `f32[3, 4]` (scalar broadcasts to all elements)
- **Rank extension**: `f32[4] + f32[3, 4]` → `f32[3, 4]` (lower-rank array right-aligned)
- **Size-1 stretching**: `f32[3, 1] * f32[3, 4]` → `f32[3, 4]` (dimension of size 1 stretches)

This enables common patterns like normalizing along an axis:

```maomi
fn normalize_rows(x: f32[32, 128]) -> f32[32, 128] {
    let row_sums = reshape(sum(x, 1), 32, 1);
    x / row_sums
}
```

---

## Type Conversion

`cast(expr, type)` converts between scalar types. Shape is preserved.

```maomi
fn to_float(x: i32[4]) -> f32[4] {
    cast(x, f32)
}

fn to_int(x: f32[4]) -> i32[4] {
    cast(x, i32)
}

fn indicator(mask: bool[4]) -> f32[4] {
    cast(mask, f32)
}
```

Supported target types: `f32`, `f64`, `i32`, `i64`, `bool`. Struct types cannot be cast.

**AD behavior:** Gradient flows through float-to-float casts (e.g. `f32 → f64`), casting the adjoint back to the source type. Casts to non-float types (`i32`, `i64`, `bool`) are non-differentiable — gradient is zero.

---

## Builtins — Elementwise

These operate on scalars and lift to arrays automatically (applied elementwise).

| Builtin | Signature | Description |
|---------|-----------|-------------|
| `exp(x)` | `f32 -> f32` | Exponential |
| `log(x)` | `f32 -> f32` | Natural logarithm |
| `tanh(x)` | `f32 -> f32` | Hyperbolic tangent |
| `sqrt(x)` | `f32 -> f32` | Square root |
| `abs(x)` | `f32 -> f32` | Absolute value |

```maomi
fn softplus(x: f32[4]) -> f32[4] {
    log(exp(x) + 1.0)
}
```

All elementwise builtins are differentiable.

---

## Builtins — Reductions

| Builtin | Signature | Description |
|---------|-----------|-------------|
| `sum(x)` | `f32[...] -> f32` | Sum all elements |
| `sum(x, axis)` | `f32[M, N] -> f32[M]` (axis=1) | Sum along specific axis |
| `mean(x)` | `f32[...] -> f32` | Average of all elements |
| `mean(x, axis)` | `f32[M, N] -> f32[N]` (axis=0) | Average along specific axis |
| `max(x)` | `f32[...] -> f32` | Maximum of all elements |
| `max(x, axis)` | `f32[M, N] -> f32[M]` (axis=1) | Maximum along specific axis |
| `min(x)` | `f32[...] -> f32` | Minimum of all elements |
| `min(x, axis)` | `f32[M, N] -> f32[N]` (axis=0) | Minimum along specific axis |
| `argmax(x)` | `f32[...] -> i32` | Index of maximum element |
| `argmax(x, axis)` | `f32[M, N] -> i32[M]` (axis=1) | Indices of maxima along axis |
| `argmin(x)` | `f32[...] -> i32` | Index of minimum element |
| `argmin(x, axis)` | `f32[M, N] -> i32[N]` (axis=0) | Indices of minima along axis |

```maomi
fn mse_loss(pred: f32[32], target: f32[32]) -> f32 {
    let diff = pred - target;
    mean(diff * diff)
}

fn softmax(x: f32[32, 10]) -> f32[32, 10] {
    let m = reshape(max(x, 1), 32, 1);
    let e = exp(x - m);
    let s = reshape(sum(e, 1), 32, 1);
    e / s
}

fn predictions(logits: f32[32, 10]) -> i32[32] {
    argmax(logits, 1)
}
```

Axis-specific reductions remove the specified dimension from the result shape. The axis must be an integer literal in range `[0, ndim)`.

- `sum` / `mean`: fully differentiable — `grad` through `sum` distributes ones, `grad` through `mean` distributes `1/N`.
- `max` / `min`: differentiable via indicator rule — gradient flows only to the element(s) that achieved the max/min. If there are ties, gradient is split equally among winners.
- `argmax` / `argmin`: **non-differentiable** (returns i32 indices). No gradient flows through them.

---

## Builtins — Conditional & Control

| Builtin | Signature | Description |
|---------|-----------|-------------|
| `where(cond, x, y)` | `bool[...], T, T -> T` | Element-wise conditional select |
| `stop_gradient(x)` | `T -> T` | Identity in forward pass, blocks gradient flow |

```maomi
fn masked_scores(scores: f32[128], mask: bool[128]) -> f32[128] {
    where(mask, scores, 0.0)
}

fn td_target(reward: f32, gamma: f32, next_v: f32) -> f32 {
    reward + gamma * stop_gradient(next_v)
}
```

`where` is the array-conditional counterpart to `if/else` — it selects element-by-element based on a bool array. Both branches are evaluated. Differentiable: gradient flows to `x` where condition is true, to `y` where false.

`stop_gradient` prevents gradient flow during reverse-mode AD. The value passes through unchanged, but `grad` treats it as a constant (zero adjoint). Used for RL targets, detached baselines, and any computation that should not be differentiated.

---

## Builtins — Debugging

### callback

Fire-and-forget host callback — sends values from the XLA device to Python during execution. Used for logging, printing intermediate values, and debugging.

```maomi
callback(expr1, expr2, ...)   // any number of args (including zero)
```

```maomi
fn train_step(x: f32[4], w: f32[4]) -> f32[4] {
    let loss = sum(x * w);
    callback(loss);
    grad(loss, w)
}
```

When using `maomi run`, callbacks print their arguments to stdout prefixed with `[callback]`. When using the Python API (`compile_source` + `run_stablehlo`), you provide custom Python callables via the `host_callbacks` parameter.

`callback` returns no value (type `None`), has side effects (the compiler cannot eliminate it), and is ignored by AD (zero gradient). It is the equivalent of JAX's `jax.debug.callback`.

---

## Builtins — Shape & Array

### reshape

Reshape an array to new dimensions. The total element count must match.

```maomi
reshape(x, d1, d2, ...)
```

```maomi
fn flatten(x: f32[4, 8]) -> f32[32] {
    reshape(x, 32)
}

fn unflatten(x: f32[32]) -> f32[4, 8] {
    reshape(x, 4, 8)
}
```

Differentiable — gradient reshapes back to the original shape.

### concat

Concatenate arrays along an axis (default: axis 0).

```maomi
concat(a, b)          // concatenate along axis 0
concat(a, b, 1)       // concatenate along axis 1
concat(a, b, c, 0)    // three arrays along axis 0
```

```maomi
fn join(a: f32[4], b: f32[6]) -> f32[10] {
    concat(a, b)
}
```

Differentiable — gradient slices back to original shapes.

### iota

Generate an integer sequence `[0, 1, 2, ..., N-1]`.

```maomi
fn indices(n: i32) -> i32[8] {
    iota(8)
}
```

The argument must be an integer literal. Returns `i32[N]`. Not differentiable (integer output).

---

## Builtins — Neural Network

### conv2d

2D convolution with NCHW layout.

```
conv2d(input, kernel)                              // stride=1, pad=0
conv2d(input, kernel, stride, pad)                 // same stride/pad for both spatial dims
conv2d(input, kernel, stride_h, stride_w, pad_h, pad_w)  // independent stride/pad
```

- `input`: `f32[N, C_in, H, W]`
- `kernel`: `f32[C_out, C_in, K_h, K_w]`
- Returns: `f32[N, C_out, H_out, W_out]`

Output dimensions: `H_out = (H + 2*pad_h - K_h) / stride_h + 1`

```maomi
fn conv_forward(x: f32[1, 1, 4, 4], w: f32[1, 1, 3, 3]) -> f32[1, 1, 2, 2] {
    conv2d(x, w)
}

fn conv_strided(x: f32[1, 3, 8, 8], w: f32[16, 3, 3, 3]) -> f32[1, 16, 4, 4] {
    conv2d(x, w, 2, 1)
}
```

Differentiable w.r.t. both input and kernel.

### max_pool

Max pooling with specified window and stride.

```
max_pool(input, window_h, window_w, stride_h, stride_w)
```

- `input`: `f32[N, C, H, W]`
- Returns: `f32[N, C, H_out, W_out]`

```maomi
fn pool(x: f32[1, 1, 4, 4]) -> f32[1, 1, 2, 2] {
    max_pool(x, 2, 2, 2, 2)
}
```

Differentiable — gradient flows only through the max elements (select-and-scatter).

### avg_pool

Average pooling with specified window and stride. Same signature as `max_pool`.

```maomi
fn pool(x: f32[1, 1, 4, 4]) -> f32[1, 1, 2, 2] {
    avg_pool(x, 2, 2, 2, 2)
}
```

Differentiable — gradient distributes evenly across the pooling window.

---

## Builtins — RNG

Maomi uses deterministic, key-threaded random number generation (same model as JAX). All randomness is explicit: you create a key from a seed, split it to get independent subkeys, and pass keys to sampling functions. Same seed always produces the same output.

### random.key

Create an RNG key from an integer seed.

```maomi
random.key(seed: i32) -> Key    // Key = i32[4]
```

```maomi
let key = random.key(42);
```

### random.split

Split one key into `n` independent subkeys.

```maomi
random.split(key: Key, n: int_literal) -> i32[n, 4]
```

```maomi
let keys = random.split(key, 3);
let k1 = keys[0];   // use for one operation
let k2 = keys[1];   // use for another
```

The count `n` must be an integer literal (known at compile time).

### random.uniform

Sample from a uniform distribution in `[low, high)`.

```maomi
random.uniform(key: Key, low: f32, high: f32, d1, d2, ...) -> f32[d1, d2, ...]
```

```maomi
let x = random.uniform(key, 0.0, 1.0, 4, 4);    // f32[4, 4] in [0, 1)
let y = random.uniform(key, -0.1, 0.1, 128);     // f32[128] in [-0.1, 0.1)
```

### random.normal

Sample from a normal (Gaussian) distribution.

```maomi
random.normal(key: Key, mean: f32, std: f32, d1, d2, ...) -> f32[d1, d2, ...]
```

```maomi
let w = random.normal(key, 0.0, 1.0, 128, 64);   // standard normal, f32[128, 64]
```

Uses the Box-Muller transform internally.

### RNG Usage Pattern

```maomi
fn init_weights(seed: i32) -> f32[128, 64] {
    let key: Key = random.key(seed);
    let keys = random.split(key, 2);
    let w = random.normal(keys[0], 0.0, 0.01, 128, 64);
    w
}
```

All dimension arguments must be integer literals. RNG builtins have zero gradient — they participate in forward computation but are not differentiable (same as JAX).

---

## Array Indexing

### Single Element

```maomi
x[0]      // first element
x[-1]     // last element
x[i]      // dynamic index (i is an i32 variable)
```

### Slicing

```maomi
x[1:3]    // elements at indices 1, 2
x[1:]     // from index 1 to end
x[:3]     // from start to index 3 (exclusive)
x[:-1]    // all but last element
```

Slice bounds must be integer literals.

### Multi-Axis

```maomi
x[:, 0]     // first column of a 2D array
x[1:3, 2]   // rows 1-2, column 2
```

### Chained Indexing

```maomi
x[0][1]     // row 0, then element 1
```

### Gather Indexing

Index with an integer array to gather multiple elements.

```maomi
fn lookup(table: f32[10, 4], ids: i32[3]) -> f32[3, 4] {
    table[ids]
}
```

`ids` must be an `i32` array. The result shape is `ids.shape + element_shape`.

### Differentiability

All indexing forms are differentiable. `grad` through single indexing uses `dynamic_update_slice` (places the adjoint at the indexed position in a zero array). `grad` through gather indexing uses scatter (accumulates adjoints at gathered positions).

---

## Structs

### Definition

```maomi
struct Point { x: f32, y: f32 }
struct Layer { w: f32[4, 4], b: f32[4] }
struct Net { hidden: Layer, output: Layer }
```

Structs compile to StableHLO tuples.

### Construction

```maomi
let p = Point { x: 1.0, y: 2.0 };
let layer = Layer { w: weights, b: bias };
```

### Field Access

```maomi
p.x          // access field
net.hidden.w // nested access
```

### Functional Update

Create a new struct with one or more fields changed (F#-style `with` expression).

```maomi
let p2 = p with { x = 3.0 };                // change x, keep y
let net2 = net with { hidden.b = new_bias }; // nested update
```

### Struct Gradients

`grad` works with struct-typed variables. The result is a struct with per-field gradients.

```maomi
struct Params { w: f32[4, 4], b: f32[4] }

fn train(p: Params, x: f32[4]) -> Params {
    let loss = sum(x @ p.w + p.b);
    grad(loss, p)
}
```

---

## Control Flow

### if/else

Conditional expression — both branches must return the same type. Returns a value.

```maomi
if x > 0.0 { x } else { 0.0 }
```

Differentiable: both branches are differentiated, and the condition selects which gradient to use (condition itself is not differentiated).

### map

Elementwise transform over a sequence. The body must be trivially batchable (no cross-element dependencies).

```maomi
map elem in sequence {
    body_expr
}
```

```maomi
fn relu(xs: f32[32, 64]) -> f32[32, 64] {
    map x in xs {
        if x > 0.0 { x } else { 0.0 }
    }
}
```

Differentiable: the gradient of `map` distributes over the body.

### scan

Sequential fold with carried state. Processes a sequence element by element, threading an accumulator.

```maomi
scan (carry, elem) in (init, sequence) {
    new_carry_expr
}
```

```maomi
fn cumsum(xs: f32[10], init: f32) -> f32[10] {
    scan (acc, x) in (init, xs) {
        acc + x
    }
}
```

Multi-sequence scan:

```maomi
scan (c, x, y) in (init, xs, ys) {
    c + x * y
}
```

Differentiable: backward pass runs a reverse scan to propagate gradients.

### while

Loop with a runtime condition. Unlike `scan`, the iteration count is determined by the condition, not a sequence length — the program doesn't need to recompile when the number of iterations changes.

```maomi
while state in init { condition } do {
    body
}
```

```maomi
fn converge(x: f32) -> f32 {
    while s in x { s > 0.01 } do {
        s * 0.5
    }
}
```

The condition block must return `bool`. The body must return the same type as the initial state. The result is the final state value.

**Bounded while** — adding `limit N` pre-allocates a trajectory buffer of N entries, enabling reverse-mode AD. The loop still stops when the condition is false, but runs at most N iterations.

```maomi
fn converge_diff(x: f32) -> f32 {
    let result = while s in x limit 100 { s > 0.01 } do {
        s * 0.5
    };
    grad(result, x)
}
```

| Variant | Differentiable? | Use case |
|---------|----------------|----------|
| `while s in init { cond } do { body }` | No | Inference, dynamic stopping |
| `while s in init limit N { cond } do { body }` | Yes | Training through dynamic loops |

Attempting `grad` through an unbounded `while` (no `limit`) produces a compile error. This matches JAX's semantics — `jax.grad` through `jax.lax.while_loop` is also unsupported.

### fold

Sequential reduction that returns only the final accumulator — like `scan` but without stacking intermediate results.

```maomi
fold (carry, elem) in (init, sequence) {
    new_carry_expr
}
```

```maomi
fn manual_sum(xs: f32[10]) -> f32 {
    fold (acc, x) in (0.0, xs) { acc + x }
}
```

Unlike `scan` (which stacks all intermediate carries into an array), `fold` returns just the final carry value. This makes it ideal for training loops where you only want the trained parameters, not all intermediate states. Critically, `fold` supports struct carries — `scan` cannot (since it would need to stack structs).

```maomi
struct Model { w: f32[784, 10], b: f32[10] }

fn train(model: Model, data: f32[60000, 784], labels: f32[60000, 10]) -> Model {
    fold (m, (x, y)) in (model, (data, labels)) {
        let pred = x @ m.w + m.b;
        let g = grad(sum((pred - y) * (pred - y)), m);
        m with { w = m.w - 0.01 * g.w, b = m.b - 0.01 * g.b }
    }
}
```

**AD:** `grad` through `fold` is fully supported. The compiler internally records a forward trajectory and runs a reverse accumulation loop, identical to scan's backward pass.

---

## Automatic Differentiation

### Basic Usage

`grad(expr, var)` computes the gradient of a scalar expression with respect to a variable using reverse-mode AD.

```maomi
fn train_step(x: f32[4], w: f32[4]) -> f32[4] {
    let loss = mean(x * w);
    grad(loss, w)
}
```

The expression must produce a scalar (`f32`). The result has the same type as the variable.

### What's Differentiable

| Operation | Differentiable? | Notes |
|-----------|----------------|-------|
| `+` `-` `*` `/` `**` | Yes | Standard rules |
| `@` (matmul) | Yes | Transposes for backward pass |
| `exp` `log` `tanh` `sqrt` `abs` | Yes | Chain rule applied |
| `sum` `mean` | Yes | Distributes ones / `1/N` (supports axis-specific) |
| `where` | Yes | Gradient flows per-element based on condition |
| `stop_gradient` | Blocks | Identity forward, zero adjoint backward |
| `reshape` `concat` | Yes | Reshapes/slices gradient back |
| `if/else` | Yes | Both branches differentiated, condition selects |
| User functions | Yes | Inlined then differentiated |
| `map` | Yes | Gradient distributes over body |
| `scan` | Yes | Reverse scan for backward pass |
| `while ... limit N` | Yes | Pre-allocated trajectory for backward pass |
| `while` (no limit) | No | Unknown iteration count; use bounded variant for grad |
| Structs / field access | Yes | Per-field gradients |
| Array indexing / gather | Yes | `dynamic_update_slice` / scatter |
| `conv2d` | Yes | Gradient w.r.t. input and kernel |
| `max_pool` | Yes | Select-and-scatter backward |
| `avg_pool` | Yes | Scale and reduce-window backward |
| `rng_*` | No | Zero gradient (non-differentiable, but values flow forward) |
| `iota` | No | Integer output, zero gradient |
| `callback` | No | Ignored by AD |

### Grad-of-Grad

Higher-order differentiation is supported by nesting `grad` calls.

```maomi
fn hessian_diag(x: f32[4]) -> f32[4] {
    let loss = sum(x * x * x);
    grad(sum(grad(loss, x)), x)
}
```

The inner `grad` produces a gradient, which is then differentiated again by the outer `grad`. Useful for computing Hessian diagonals, second-order optimization methods, etc.

---

## Module System

### Qualified Import

Import a module and access its functions with a prefix.

```maomi
import math;

fn example(x: f32[4]) -> f32[4] {
    math.relu(x)
}
```

### Selective Import

Import specific functions without a prefix.

```maomi
from math import { relu, sigmoid };

fn example(x: f32[4]) -> f32[4] {
    relu(x)
}
```

### Path-Based Import

Import from a file path with an alias.

```maomi
import "../lib/nn" as nn;

fn example(x: f32[4]) -> f32[4] {
    nn.linear(x)
}
```

### Path + Selective

```maomi
from "../lib/nn" as nn import { linear };
```

Module resolution handles cycle detection, diamond dependencies, and caching. All functions in a module are importable (no visibility modifiers yet).

---

## CLI

### compile

Compile a `.mao` file and emit a representation.

```bash
uv run maomi compile <file> --emit <format>
```

Formats:
- `tokens` — tokenized output (one token per line)
- `ast` — parsed AST as JSON
- `types` — type-checked function signatures
- `stablehlo` — StableHLO MLIR text (default)

### run

Compile and execute via JAX/XLA. Requires `uv sync --extra run`.

```bash
uv run maomi run <file> --fn <function_name> [--seed <int>]
```

- `--fn` (required): function to execute
- `--seed` (default: 42): random seed for generating input values

All function parameters must have concrete (non-symbolic) dimensions.

---

## Compilation Pipeline

```
Source (.mao)
    ↓
  Lexer        →  tokens
    ↓
  Parser       →  AST
    ↓
  Resolver     →  merged AST (imports resolved)
    ↓
  Type Checker →  typed AST + type_map
    ↓
  AD Transform →  grad expressions replaced with gradient code
    ↓
  Codegen      →  StableHLO (MLIR text)
    ↓
  JAX/XLA      →  compiled + executed on CPU/GPU/TPU
```

The compiler is written in Python. Source lives in `src/maomi/`. Key files:

| File | Role |
|------|------|
| `tokens.py` | Token type definitions |
| `lexer.py` | Tokenizer |
| `parser.py` | Recursive descent parser |
| `ast_nodes.py` | AST node definitions |
| `types.py` | Type system (`ArrayType`, `StructType`, etc.) |
| `resolver.py` | Module import resolution |
| `type_checker.py` | Type checking and builtin validation |
| `ad.py` | Reverse-mode AD (AST-to-AST transform) |
| `codegen_stablehlo.py` | StableHLO code generation |
| `jax_runner.py` | JAX/XLA execution backend |
| `cli.py` | CLI entry point |
| `errors.py` | Error types with source location |

---

## Known Limitations

- **Concrete dimensions required for codegen.** Symbolic dimensions (`f32[B, 128]`) are valid in type annotations for expressing shape relationships, but produce an error during code generation. All dimensions must be concrete integers to compile. This is inherent to XLA — it needs static shapes to make tiling, fusion, layout, and memory allocation decisions. JAX works the same way (`jax.jit` recompiles per input shape). Symbolic dims are a type-checking convenience for verifying shape consistency, not a runtime feature.
- **`map` bodies must be elementwise.** No cross-element dependencies allowed in `map` bodies.
- **Slice bounds must be literals.** `x[1:3]` works, but `x[a:b]` where `a` and `b` are variables does not.
- **No rank polymorphism.** Functions have fixed-rank parameters — you cannot write a single function that works on any rank.
- **Fixed literal types.** Integer literals are always `i32`, float literals are always `f32`. No `f64` or `i64` literals.
- **No visibility modifiers.** All functions in a module are importable. No `pub`/`private` distinction.
