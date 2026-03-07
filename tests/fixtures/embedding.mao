fn lookup(table: f32[10, 4], ids: i32[3]) -> f32[3, 4] {
    table[ids]
}

fn grad_1d(x: f32[100], ids: i32[8]) -> f32[100] {
    grad(sum(x[ids]), x)
}
