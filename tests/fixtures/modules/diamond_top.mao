import diamond_left;
import diamond_right;

fn combined(x: f32) -> f32 {
    diamond_left.left_fn(x) + diamond_right.right_fn(x)
}
