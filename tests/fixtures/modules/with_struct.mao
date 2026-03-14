struct Point { x: f32, y: f32 }

fn make_point(a: f32, b: f32) -> Point {
    Point { x: a, y: b }
}

fn get_x(p: Point) -> f32 {
    p.x
}
