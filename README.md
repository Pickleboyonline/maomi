# Maomi

A pure functional ML language that compiles to XLA via StableHLO.

**If it compiles, it's fast.** No `@jit` decorators, no tracing surprises, no sharp bits.

## Example

```
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

## Install

```bash
uv sync
```

## Usage

```bash
# Tokenize
uv run maomi compile examples/linear.mao --emit tokens

# Parse to AST
uv run maomi compile examples/linear.mao --emit ast

# Type check
uv run maomi compile examples/mlp.mao --emit types

# Compile to StableHLO
uv run maomi compile examples/grad.mao --emit stablehlo
```

## Status

v0.2 — Frontend (lexer, parser, type checker) + `scan`, `map`, `grad` primitives + StableHLO codegen.
