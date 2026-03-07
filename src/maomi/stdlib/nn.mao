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
