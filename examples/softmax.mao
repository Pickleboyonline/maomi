// Softmax, normalization, and RL patterns using axis reduction,
// size-1 broadcasting, where, and stop_gradient

// Softmax: exp(x) / sum(exp(x), axis=1)
fn softmax(x: f32[4, 8]) -> f32[4, 8] {
    let e = exp(x);
    let s = reshape(sum(e, 1), 4, 1);
    e / s
}

// Layer normalization along last axis
fn layer_norm(x: f32[4, 8], gamma: f32[8], beta: f32[8]) -> f32[4, 8] {
    let mu = reshape(mean(x, 1), 4, 1);
    let centered = x - mu;
    let var = reshape(mean(centered * centered, 1), 4, 1);
    let normed = centered / sqrt(var + 0.00001);
    normed * gamma + beta
}

// Masked selection: zero out where mask is false
fn apply_mask(scores: f32[4, 8], mask: bool[4, 8]) -> f32[4, 8] {
    where(mask, scores, 0.0)
}

// TD target with stop_gradient (RL)
fn td_error(reward: f32, gamma: f32, next_v: f32, v: f32) -> f32 {
    let target = reward + gamma * stop_gradient(next_v);
    target - v
}

// Gradient of softmax loss
fn softmax_grad(x: f32[4, 8]) -> f32[4, 8] {
    grad(sum(softmax(x)), x)
}
