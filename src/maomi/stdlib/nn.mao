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

/// CELU — Continuously Differentiable ELU: max(0,x) + min(0, alpha*(exp(x/alpha)-1)).
fn celu(x: f32[..], alpha: f32) -> f32[..] {
    where(x > 0.0, x, alpha * (exp(x / alpha) - 1.0))
}

/// Hard sigmoid — piecewise linear approximation of sigmoid.
fn hard_sigmoid(x: f32[..]) -> f32[..] {
    clip(x / 6.0 + 0.5, 0.0, 1.0)
}

/// Hard swish — x * hard_sigmoid(x), efficient approximation of SiLU.
fn hard_swish(x: f32[..]) -> f32[..] {
    x * clip(x / 6.0 + 0.5, 0.0, 1.0)
}

/// Hard tanh — clips input to [-1, 1].
fn hard_tanh(x: f32[..]) -> f32[..] {
    clip(x, 0.0 - 1.0, 1.0)
}

/// ReLU6 — ReLU clamped at 6: min(max(x, 0), 6).
fn relu6(x: f32[..]) -> f32[..] {
    clip(where(x > 0.0, x, 0.0), 0.0, 6.0)
}

/// Log-sigmoid — numerically stable log(sigmoid(x)) = -softplus(-x).
fn log_sigmoid(x: f32[..]) -> f32[..] {
    0.0 - softplus(0.0 - x)
}

/// Squareplus — smooth approximation of ReLU: (x + sqrt(x^2 + b)) / 2.
fn squareplus(x: f32[..], b: f32) -> f32[..] {
    (x + sqrt(square(x) + b)) / 2.0
}

/// Batch normalization along a specific axis with learnable scale and shift.
fn batch_norm(x: f32[..], gamma: f32[..], beta: f32[..], comptime axis: i32) -> f32[..] {
    let m = mean(x, axis=axis, keepdims=true);
    let v = mean(square(x - m), axis=axis, keepdims=true);
    gamma * (x - m) / sqrt(v + 1e-5) + beta
}
