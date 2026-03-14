struct Inner { val: f32 }
struct Outer { inner: Inner, scale: f32 }

fn make_outer(v: f32, s: f32) -> Outer {
    Outer { inner: Inner { val: v }, scale: s }
}
