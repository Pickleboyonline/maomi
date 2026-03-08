// Multi-layer perceptron
from nn import { sigmoid };

/// lin mapping
fn linear(x: f32[B, N], w: f32[N, M], b: f32[M]) -> f32[B, M] {
    x @ w + b
}

fn relu(x: f32[B, N]) -> f32[B, N] {
    let a = linear();
    if x > 0.0 { x } else { 0.0 }
}

fn apple(x: f32[..]) -> f32[..] {
    exp(x)
} 

fn mse_loss(pred: f32[B], target: f32[B]) -> f32 {
    let diff = pred - target;
    mean(diff * diff)
}

fn mlp(x: f32[B, 784], w1: f32[784, 256], b1: f32[256], w2: f32[256, 10], b2: f32[10]) -> f32[B, 10] {
    let h = relu(linear(x, w1, b1));
    linear(h, w2, b2)
}
