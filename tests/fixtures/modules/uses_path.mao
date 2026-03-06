import "lib/helpers" as helpers;

fn add_two(x: f32) -> f32 {
    helpers.add_one(helpers.add_one(x))
}
