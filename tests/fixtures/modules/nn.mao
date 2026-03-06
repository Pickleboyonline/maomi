fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn linear(x: f32[4], w: f32[4]) -> f32 {
    sum(x * w)
}
