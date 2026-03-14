/// Variance along a specific axis.
fn var(x: f32[..], comptime axis: i32) -> f32[..] {
    let m = mean(x, axis=axis, keepdims=true);
    mean(square(x - m), axis=axis)
}

/// Standard deviation along a specific axis.
fn std(x: f32[..], comptime axis: i32) -> f32[..] {
    let m = mean(x, axis=axis, keepdims=true);
    let v = mean(square(x - m), axis=axis);
    sqrt(v)
}

/// Normalize along a specific axis (zero mean, unit variance).
fn normalize(x: f32[..], comptime axis: i32) -> f32[..] {
    let m = mean(x, axis=axis, keepdims=true);
    let v = mean(square(x - m), axis=axis, keepdims=true);
    (x - m) / sqrt(v + 1e-5)
}
