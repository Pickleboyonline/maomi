/// SGD parameter update: params = params - lr * grads.
fn sgd_update(params: T, grads: T, lr: f32) -> T {
    params - lr * grads
}

/// Adam moment update + parameter step.
/// m and v are first/second moment estimates (same struct type as params).
/// step should start at 1 and increment each call.
/// Typical defaults: b1=0.9, b2=0.999, eps=1e-8.
fn adam_update(params: T, grads: T, m: T, v: T,
              step: i32, lr: f32, b1: f32, b2: f32, eps: f32) -> T {
    let new_m = b1 * m + (1.0 - b1) * grads;
    let new_v = b2 * v + (1.0 - b2) * grads * grads;
    let step_f = cast(step, f32);
    let m_hat = new_m / (1.0 - b1 ** step_f);
    let v_hat = new_v / (1.0 - b2 ** step_f);
    params - lr * m_hat / (sqrt(v_hat) + eps)
}

/// Update first moment estimate (use when you need m separately).
fn adam_m_update(m: T, grads: T, b1: f32) -> T {
    b1 * m + (1.0 - b1) * grads
}

/// Update second moment estimate (use when you need v separately).
fn adam_v_update(v: T, grads: T, b2: f32) -> T {
    b2 * v + (1.0 - b2) * grads * grads
}

/// Linear learning rate decay from init_lr to end_lr over total_steps.
fn linear_decay(step: i32, init_lr: f32, end_lr: f32, total_steps: i32) -> f32 {
    let s = cast(step, f32);
    let n = cast(total_steps, f32);
    let t = s / n;
    init_lr + t * (end_lr - init_lr)
}

/// Cosine annealing: decays from init_lr toward 0 over total_steps.
fn cosine_decay(step: i32, init_lr: f32, total_steps: i32) -> f32 {
    let s = cast(step, f32);
    let n = cast(total_steps, f32);
    let t = s / n;
    init_lr * 0.5 * (1.0 + cos(t * 3.14159265))
}
