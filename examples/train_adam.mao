// Training a simple linear model with Adam from stdlib
from optim import { adam_update, adam_m_update, adam_v_update };

struct Model { w: f32[4, 4], b: f32[4] }

// Adam state: params + moment estimates + step counter
struct TrainState { params: Model, m: Model, v: Model, step: i32 }

fn loss(m: Model, x: f32[32, 4], y: f32[32, 4]) -> f32 {
    let pred = x @ m.w + m.b;
    mean((pred - y) * (pred - y))
}

fn train_step(p: Model, m: Model, v: Model, step: i32,
              x: f32[32, 4], y: f32[32, 4]) -> TrainState {
    let g = grad(loss(p, x, y), p);
    let new_m = adam_m_update(m, g, 0.9);
    let new_v = adam_v_update(v, g, 0.999);
    let new_p = adam_update(p, g, m, v, step, 0.001, 0.9, 0.999, 0.00000001);
    TrainState { params: new_p, m: new_m, v: new_v, step: step + 1 }
}
