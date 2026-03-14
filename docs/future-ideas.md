# Future Ideas

## Implemented StableHLO Primitives

The following primitives from the original roadmap have been implemented:

**Tier 1 (all complete):**
- `stablehlo.convolution` → `conv2d(input, kernel, stride, padding)` with full AD
- `stablehlo.gather` → `x[ids]` array-based indexing with AD (scatter for backward)
- `stablehlo.scatter` → Used internally for gather gradients
- `stablehlo.reduce_window` → `max_pool` and `avg_pool` with AD
- `stablehlo.iota` → `iota(N)` integer sequences

**Tier 2 (mostly complete):**
- `stablehlo.concatenate` → `concat(a, b, axis)`
- `stablehlo.reshape` → `reshape(x, d1, d2, ...)`
- `stablehlo.pad` → `pad(x, pad_width)`
- `stablehlo.sort` → `sort(x, axis)` and `argsort(x, axis)` with AD
- `stablehlo.rng` → `random.key`, `random.split`, `random.uniform`, `random.normal`, `random.bernoulli`, `random.categorical`, `random.truncated_normal`

**Still not implemented:**
- `stablehlo.reverse` — reverse along axes
- `stablehlo.cholesky` — Cholesky decomposition
- `stablehlo.triangular_solve` — solve triangular systems
- `stablehlo.fft` — FFT / spectral methods
- `stablehlo.all_reduce` / `all_gather` — distributed training
- `stablehlo.custom_call` — escape hatch to custom kernels

---

## Language Features (compile to same StableHLO, zero runtime cost)

### Named dimensions

JAX tracks axis numbers mentally, gets runtime errors. Maomi could make dimension names part of the type system:

```maomi
fn attention(q: f32[batch: B, seq: S, head: D]) -> f32[batch: B, seq: S, head: D] {
    let scores = sum(q, axis=head);   // axis by name, not number
    ...
}
```

Compiles to the same `stablehlo.reduce`. But the compiler rejects `sum(q, axis=batch)` if that's not what you meant. Named dims also make broadcasting rules explicit: `f32[batch: B, 1]` broadcasts with `f32[batch: B, features: 128]` because the names align.

### Dimension arithmetic in types

JAX figures out output shapes at trace time. No static guarantees.

```maomi
fn split_heads(x: f32[B, S, D]) -> f32[B, S, D/8, 8] {
    reshape(x)   // compiler infers output shape from return type
}

fn concat_pair(a: f32[N], b: f32[M]) -> f32[N + M] {
    concat(a, b)
}
```

Type checker does symbolic arithmetic — `D/8` requires `D` divisible by 8, checked at compile time. StableHLO just sees concrete shapes, but the language proves they're correct.

### Differentiability in the type system

JAX: everything is a float array, `grad` just tries and fails at runtime.

```maomi
fn train(params: diff f32[128, 64], data: f32[32, 128], labels: i32[32]) -> diff f32[128, 64] {
    let loss = cross_entropy(data @ params, labels);
    grad(loss, params)   // compiler knows params is diff, labels is not
}
```

A `diff` qualifier means: this participates in AD. `grad(loss, labels)` is a compile-time error (i32 isn't differentiable, not marked `diff`). JAX discovers differentiability by tracing; Maomi would prove it statically.

### Explicit batching / auto-vectorization

JAX's `vmap` is a function transform with confusing semantics.

```maomi
fn predict(w: f32[128, 10], x: f32[128]) -> f32[10] {
    x @ w
}

// Option A: explicit map (already works)
fn batch_predict(w: f32[128, 10], xs: f32[B, 128]) -> f32[B, 10] {
    map x in xs { predict(w, x) }
}

// Option B: compiler auto-lifts — call with f32[B, 128] where f32[128] expected
fn batch_predict(w: f32[128, 10], xs: f32[B, 128]) -> f32[B, 10] {
    predict(w, xs)   // auto-batched because types demand it
}
```

Same StableHLO output, zero ceremony.

### Shape pattern matching

JAX: `if x.ndim == 2` at trace time, hope for the best.

```maomi
fn norm(x: f32[...]) -> f32[...] {
    match rank(x) {
        1 => x / sqrt(sum(x * x)),
        2 => map row in x { norm(row) },
    }
}
```

Dispatch at compile time based on rank or dimension values. Compiles away entirely.

### Checkpointing / rematerialization hints

JAX: `jax.checkpoint` is a decorator you manually apply.

```maomi
fn transformer_block(x: f32[B, S, D]) -> f32[B, S, D] {
    let attn = checkpoint { self_attention(x) };  // recompute in backward, don't save
    let out = ffn(attn);
    out
}
```

Tells AD: don't store this on the tape, recompute in backward. Controls memory for training as a compile-time guarantee.

### Priority order

1. **Named dimensions** — biggest source of bugs in JAX/numpy is axis mixups. Compile-time checked with zero runtime cost.
2. **Dimension arithmetic** — `f32[N + M]`, `f32[N/8, 8]` in types. Makes reshape/concat/split safe by construction.
3. **Differentiability in types** — `diff` qualifier. Makes grad errors impossible instead of runtime surprises.

All three compile to the exact same StableHLO. They're purely compile-time features that disappear after type checking. The opportunity: Maomi's compiler proves things that JAX discovers by running.
