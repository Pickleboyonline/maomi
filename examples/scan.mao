// Cumulative sum using scan
fn cumsum(xs: f32[10], init: f32) -> f32[10] {
    scan (acc, x) in (init, xs) {
        acc + x
    }
}
