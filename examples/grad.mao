// Gradient of a simple loss function
fn grad_loss(x: f32[4], w: f32[4]) -> f32[4] {
    let loss = mean(x * w);
    grad(loss, w)
}
