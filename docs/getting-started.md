# Getting Started with Maomi

This tutorial walks you through writing, compiling, and running your first Maomi programs.

## Installation

Maomi requires Python >= 3.11 and uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
git clone https://github.com/your-username/maomi.git
cd maomi

# Install the compiler
uv sync

# Install with JAX execution backend (to actually run programs)
uv sync --extra run
```

## 1. Your First Function

Create a file called `hello.mao`:

```maomi
fn add(a: f32, b: f32) -> f32 {
    a + b
}
```

Every Maomi program is a collection of functions. Parameters and return types are always annotated. The last expression in a function body is the return value — no `return` keyword needed.

Compile it to see the StableHLO output:

```bash
uv run maomi compile hello.mao --emit stablehlo
```

Or check just the types:

```bash
uv run maomi compile hello.mao --emit types
```

## 2. Working with Arrays

Maomi's type system is shape-aware. Array types specify their element type and dimensions.

```maomi
fn dot(a: f32[4], b: f32[4]) -> f32 {
    sum(a * b)
}

fn scale(x: f32[8], factor: f32) -> f32[8] {
    x * factor
}
```

`a * b` multiplies elementwise. `sum` reduces all elements to a scalar. The compiler verifies that shapes are compatible at compile time.

## 3. Defining a Model

Let's build a simple neural network layer. Maomi uses `@` for matrix multiplication, `map` for elementwise transforms, and `if/else` for conditionals.

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
```

`map x in xs { ... }` applies the body to each element. The `if/else` is an expression — it returns a value.

## 4. Computing Gradients

`grad(expr, var)` computes the gradient of a scalar expression with respect to a variable using reverse-mode automatic differentiation.

```maomi
fn train_step(x: f32[4], w: f32[4]) -> f32[4] {
    let loss = mean(x * w);
    grad(loss, w)
}
```

`grad` returns an array with the same shape as the variable you're differentiating with respect to. The expression must be a scalar (`f32`).

Run it:

```bash
uv run maomi run hello.mao --fn train_step
```

This compiles the function, generates random inputs, and executes via JAX/XLA.

## 5. Sequential Computation with Scan

`scan` is a sequential fold — it processes a sequence element by element, carrying state forward.

```maomi
fn cumsum(xs: f32[10], init: f32) -> f32[10] {
    scan (acc, x) in (init, xs) {
        acc + x
    }
}
```

`acc` is the carry (initialized to `init`), `x` is the current element from `xs`. The body returns the new carry. The result is the sequence of all carry values.

Scan is fully differentiable — the gradient runs a reverse scan automatically.

## 6. Grouping Parameters with Structs

As models grow, you'll want to group related parameters. Structs let you do this.

```maomi
struct Params { w: f32[4, 4], b: f32[4] }

fn train(p: Params, x: f32[4]) -> Params {
    let loss = sum(x @ p.w + p.b);
    grad(loss, p)
}
```

`grad` with a struct variable returns a struct of the same type, with per-field gradients. You can also do functional updates:

```maomi
fn update(p: Params, grads: Params, lr: f32) -> Params {
    let new_w = p.w - lr * grads.w;
    let new_b = p.b - lr * grads.b;
    Params { w: new_w, b: new_b }
}
```

Structs can be nested:

```maomi
struct Net { hidden: Params, output: Params }
```

And updated at nested paths:

```maomi
let net2 = net with { hidden.b = new_bias };
```

## 7. Random Initialization

Real models need random weight initialization. Maomi provides deterministic, key-threaded RNG — same model as JAX. You create a key from a seed, split it for independent randomness, and pass keys to sampling functions.

```maomi
fn init(seed: i32) -> f32[128, 64] {
    let key: Key = random.key(seed);
    let keys = random.split(key, 2);
    let w = random.normal(keys[0], 0.0, 0.01, 128, 64);
    w
}
```

- `random.key(seed)` — create a key from an integer seed
- `random.split(key, n)` — split into `n` independent subkeys
- `random.uniform(key, low, high, d1, d2, ...)` — uniform random in [low, high)
- `random.normal(key, mean, std, d1, d2, ...)` — normal (Gaussian) random

Same seed always produces the same output. Different seeds produce different output.

## 8. Convolutions and Pooling

For CNNs, Maomi provides conv2d and pooling builtins with NCHW layout.

```maomi
fn conv_block(x: f32[1, 3, 8, 8], w: f32[16, 3, 3, 3]) -> f32[1, 16, 2, 2] {
    let h = conv2d(x, w, 2, 1);
    max_pool(h, 2, 2, 2, 2)
}
```

`conv2d(input, kernel, stride, padding)` performs 2D convolution. `max_pool(input, window_h, window_w, stride_h, stride_w)` performs max pooling. Both are differentiable.

See `examples/cnn.mao` for more examples including gradient computation.

## 9. Running Your Program

```bash
# Compile and run, executing a specific function
uv run maomi run your_file.mao --fn function_name

# With a specific random seed for reproducible inputs
uv run maomi run your_file.mao --fn function_name --seed 7

# Just compile (no execution)
uv run maomi compile your_file.mao --emit stablehlo
```

When you use `run`, Maomi compiles your function, generates random inputs matching the parameter types, and executes via JAX's XLA backend. It prints the inputs and output.

## 10. Using Modules

Split your code across files and import functions.

Create `math.mao`:
```maomi
fn relu(x: f32[4]) -> f32[4] {
    if x > 0.0 { x } else { 0.0 }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(0.0 - x))
}
```

Use it from another file:
```maomi
import "math" as math;

fn forward(x: f32[4]) -> f32[4] {
    math.relu(x)
}
```

Or import specific functions:
```maomi
from "math" import { relu };

fn forward(x: f32[4]) -> f32[4] {
    relu(x)
}
```

## Next Steps

- Browse `examples/` for complete working programs (mlp, cnn, structs, scan, grad)
- Read the [Language Reference](reference.md) for complete syntax and builtin documentation
- Check `docs/future-ideas.md` for the roadmap
