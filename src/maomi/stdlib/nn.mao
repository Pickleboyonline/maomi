/// Rectified Linear Unit — returns max(x, 0).
fn relu(x: f32[..]) -> f32[..] {
    where(x > 0.0, x, 0.0)
}

/// Sigmoid activation — 1 / (1 + exp(-x)).
fn sigmoid(x: f32[..]) -> f32[..] {
    1.0 / (1.0 + exp(0.0 - x))
}

/// Softmax for 1D vectors. Use with map for batched input.
fn softmax(x: f32[N]) -> f32[N] {
    let e = exp(x - max(x));
    e / sum(e)
}

/// Log-softmax for 1D vectors (numerically stable).
fn log_softmax(x: f32[N]) -> f32[N] {
    let m = max(x);
    x - m - log(sum(exp(x - m)))
}
