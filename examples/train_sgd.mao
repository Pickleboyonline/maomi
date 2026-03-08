// Training a simple linear model with SGD from stdlib
from optim import { sgd_update };

struct Model { w: f32[4, 4], b: f32[4] }

fn loss(m: Model, x: f32[32, 4], y: f32[32, 4]) -> f32 {
    let pred = x @ m.w + m.b;
    mean((pred - y) * (pred - y))
}

fn train_step(m: Model, x: f32[32, 4], y: f32[32, 4]) -> Model {
    let g = grad(loss(m, x, y), m);
    sgd_update(m, g, 0.01)
}
