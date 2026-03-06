fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn grad_relu(x: f32) -> f32 {
    grad(relu(x), x)
}
