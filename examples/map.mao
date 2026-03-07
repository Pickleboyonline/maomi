// Batch ReLU using map
fn batch_relu(xs: f32[32, 64]) -> f32[32, 64] {
    map x in xs {
        if x > 0.0 { x } else { 0.0 }
    }
}
