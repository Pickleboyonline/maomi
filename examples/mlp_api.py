"""MLP training example using the maomi.compile API."""

import numpy as np
import maomi

mod = maomi.compile("""
    struct Params {
        w1: f32[4, 8],
        b1: f32[8],
        w2: f32[8, 1],
        b2: f32[1]
    }

    fn relu(x: f32[32, 8]) -> f32[32, 8] {
        if x > 0.0 { x } else { 0.0 }
    }

    fn forward(p: Params, x: f32[32, 4]) -> f32[32, 1] {
        let h = relu(x @ p.w1 + p.b1);
        h @ p.w2 + p.b2
    }

    fn loss(p: Params, x: f32[32, 4], y: f32[32, 1]) -> f32 {
        let pred = forward(p, x);
        let diff = pred - y;
        mean(diff * diff)
    }

    fn train_step(p: Params, x: f32[32, 4], y: f32[32, 1]) -> Params {
        let g = grad(loss(p, x, y), p);
        let lr = 0.01;
        p with {
            w1 = p.w1 - lr * g.w1,
            b1 = p.b1 - lr * g.b1,
            w2 = p.w2 - lr * g.w2,
            b2 = p.b2 - lr * g.b2
        }
    }
""")

print(mod)

# Initialize params with small random weights
rng = np.random.default_rng(42)
params = mod.Params(
    w1=rng.standard_normal((4, 8)).astype(np.float32) * 0.1,
    b1=np.zeros(8, dtype=np.float32),
    w2=rng.standard_normal((8, 1)).astype(np.float32) * 0.1,
    b2=np.zeros(1, dtype=np.float32),
)
print("Initial params:", params)

# Generate synthetic data: y = sum(x, axis=1, keepdims=True)
x = rng.standard_normal((32, 4)).astype(np.float32)
y = x.sum(axis=1, keepdims=True).astype(np.float32)

# Training loop
for step in range(200):
    l = mod.loss(params, x, y)
    if step % 20 == 0:
        print(f"step {step:3d}  loss = {float(l):.6f}")
    params = mod.train_step(params, x, y)

l = mod.loss(params, x, y)
print(f"step 200  loss = {float(l):.6f}")
print("\nFinal params:", params)

# Check predictions
pred = mod.forward(params, x)
print(f"\nPrediction sample (first 5):")
for i in range(5):
    print(f"  target={y[i,0]:.3f}  pred={pred[i,0]:.3f}")
