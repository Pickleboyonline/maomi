// CNN building blocks: conv2d, pooling, and their gradients

// Forward pass: conv2d with default stride=1, pad=0
fn conv_forward(x: f32[1, 1, 4, 4], w: f32[1, 1, 3, 3]) -> f32[1, 1, 2, 2] {
    conv2d(x, w)
}

// Conv2d with stride and padding
fn conv_strided(x: f32[1, 3, 8, 8], w: f32[16, 3, 3, 3]) -> f32[1, 16, 4, 4] {
    conv2d(x, w, 2, 1)
}

// Max pooling: 2x2 window, stride 2
fn pool_max(x: f32[1, 1, 4, 4]) -> f32[1, 1, 2, 2] {
    max_pool(x, 2, 2, 2, 2)
}

// Average pooling: 2x2 window, stride 2
fn pool_avg(x: f32[1, 1, 4, 4]) -> f32[1, 1, 2, 2] {
    avg_pool(x, 2, 2, 2, 2)
}

// Gradient of conv2d w.r.t. input
fn conv_grad_input(x: f32[1, 1, 4, 4], w: f32[1, 1, 3, 3]) -> f32[1, 1, 4, 4] {
    let y = conv2d(x, w);
    let flat = reshape(y, 4);
    grad(sum(flat), x)
}

// Gradient of conv2d w.r.t. kernel
fn conv_grad_kernel(x: f32[1, 1, 4, 4], w: f32[1, 1, 3, 3]) -> f32[1, 1, 3, 3] {
    let y = conv2d(x, w);
    let flat = reshape(y, 4);
    grad(sum(flat), w)
}

// Gradient of max_pool
fn max_pool_grad(x: f32[1, 1, 4, 4]) -> f32[1, 1, 4, 4] {
    let y = max_pool(x, 2, 2, 2, 2);
    let flat = reshape(y, 4);
    grad(sum(flat), x)
}

// Gradient of avg_pool
fn avg_pool_grad(x: f32[1, 1, 4, 4]) -> f32[1, 1, 4, 4] {
    let y = avg_pool(x, 2, 2, 2, 2);
    let flat = reshape(y, 4);
    grad(sum(flat), x)
}
