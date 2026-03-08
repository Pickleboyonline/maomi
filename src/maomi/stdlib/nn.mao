/// Rectified Linear Unit — returns max(x, 0).
fn relu(x: f32[..]) -> f32[..] {
    where(x > 0.0, x, 0.0)
}

/// Sigmoid activation — 1 / (1 + exp(-x)).
fn sigmoid(x: f32[..]) -> f32[..] {
    1.0 / (1.0 + exp(0.0 - x))
}

/// Softmax along a specific axis (numerically stable).
fn softmax(x: f32[..], comptime axis: i32) -> f32[..] {
    let m = max(x, axis=axis, keepdims=true);
    let e = exp(x - m);
    e / sum(e, axis=axis, keepdims=true)
}

/// Log-softmax along a specific axis (numerically stable).
fn log_softmax(x: f32[..], comptime axis: i32) -> f32[..] {
    let m = max(x, axis=axis, keepdims=true);
    x - m - log(sum(exp(x - m), axis=axis, keepdims=true))
}

/// Leaky ReLU — returns x if x > 0, else alpha * x.
fn leaky_relu(x: f32[..], alpha: f32) -> f32[..] {
    where(x > 0.0, x, alpha * x)
}

/// ELU — Exponential Linear Unit: x if x > 0, else alpha * (exp(x) - 1).
fn elu(x: f32[..], alpha: f32) -> f32[..] {
    where(x > 0.0, x, alpha * (exp(x) - 1.0))
}

/// SELU — Scaled Exponential Linear Unit with self-normalizing constants.
fn selu(x: f32[..]) -> f32[..] {
    let alpha = 1.6732632;
    let scale = 1.0507010;
    scale * where(x > 0.0, x, alpha * (exp(x) - 1.0))
}

/// Mish — x * tanh(softplus(x)), a smooth self-regularized activation.
fn mish(x: f32[..]) -> f32[..] {
    x * tanh(softplus(x))
}

/// Layer normalization along a specific axis.
fn layer_norm(x: f32[..], comptime axis: i32) -> f32[..] {
    let m = mean(x, axis=axis, keepdims=true);
    let v = mean(square(x - m), axis=axis, keepdims=true);
    (x - m) / sqrt(v + 1e-5)
}
