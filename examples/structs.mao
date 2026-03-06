struct Point { x: f32, y: f32 }

struct Layer { w: f32[4, 4], b: f32[4] }

struct Net { hidden: Layer, output: Layer }

fn make_point(a: f32, b: f32) -> Point {
    Point { x: a, y: b }
}

fn shift_point(p: Point, dx: f32) -> Point {
    p with { x = p.x + dx }
}

fn forward_layer(layer: Layer, x: f32[4]) -> f32[4] {
    x @ layer.w + layer.b
}

fn grad_point(p: Point) -> Point {
    let loss = p.x * p.x + p.y * p.y;
    grad(loss, p)
}
