// A simple linear layer: y = xW + b
fn linear(x: f32[B, 128], w: f32[128, 64], b: f32[64]) -> f32[B, 64] {
    x @ w + b
}
