# Future Ideas

## StableHLO Primitives to Add

### Tier 1 — Big unlocks for ML

| Op | What it enables | Syntax idea |
|---|---|---|
| `stablehlo.convolution` | CNNs — image models impossible without this | `conv2d(input, kernel, stride=1, padding="same")` |
| `stablehlo.gather` | Fancy indexing — `x[ids]` where `ids: i32[B]`. Embedding lookups, batched gathers | `x[ids]` where `ids` is an array |
| `stablehlo.scatter` | Indexed scatter/update — `x[indices] += values`. Inverse of gather, essential for embedding gradients | Paired with gather for AD |
| `stablehlo.reduce_window` | Pooling (max pool, avg pool). Sliding window reductions | `pool(x, size=2, stride=2, mode="max")` |
| `stablehlo.iota` | Generate `[0, 1, 2, ..., N-1]`. Positional encodings, index arrays, arange | `range(10)` or `iota(10)` |

### Tier 2 — Important utilities

| Op | What it enables | Syntax idea |
|---|---|---|
| `stablehlo.concatenate` | Join arrays along an axis. Can't build sequences or combine results without this | `concat(a, b, axis=0)` |
| `stablehlo.reshape` | User-facing reshape (already used internally). Flattening, unflattening, view changes | `reshape(x, [4, 8])` |
| `stablehlo.pad` | Zero/constant padding. Needed for conv padding, sequence padding | `pad(x, [(0,1), (2,2)])` |
| `stablehlo.reverse` | Reverse along axes. Sequence processing, flipping | `reverse(x, axis=0)` |
| `stablehlo.sort` | Sorting + argsort. Top-k, ranking, beam search | `sort(x)`, `argsort(x)` |
| `stablehlo.rng` | Random number generation on-device. Dropout, initialization, sampling | `random(shape, seed)` |

### Tier 3 — Linear algebra / specialized

| Op | What it enables |
|---|---|
| `stablehlo.cholesky` | Cholesky decomposition — Gaussian processes, covariance |
| `stablehlo.triangular_solve` | Solve triangular systems — paired with Cholesky |
| `stablehlo.fft` | FFT — signal processing, spectral methods, efficient convolution |
| `stablehlo.all_reduce` / `all_gather` | Distributed training across multiple devices |
| `stablehlo.custom_call` | Escape hatch to call custom kernels (FlashAttention, cuBLAS, etc.) |

### Priority order

1. **`iota` + `gather` + `scatter`** — Completes the indexing story. `x[indices]` where indices is an array (embedding lookup) is probably the most used pattern in transformers.
2. **`concatenate` + `reshape`** — Fundamental array manipulation currently impossible. Can't even stack two arrays.
3. **`convolution` + `reduce_window`** — Opens up CNNs (conv + pooling).
4. **`rng`** — No randomness means no dropout, no stochastic anything.

These four groups take Maomi from "can express MLPs and RNNs" to "can express most standard architectures."

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

### First-class einsum syntax

JAX uses a string-based DSL with no editor support or type checking.

```maomi
fn attend(q: f32[B, S, H, D], k: f32[B, T, H, D]) -> f32[B, H, S, T] {
    contract(q, k, over=D)   // or: q *. k  (dot over shared named dims)
}
```

Compiler knows contraction dimensions from the types. Emits the same `stablehlo.dot_general` with the right contraction/batch dims inferred.

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
