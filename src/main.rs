#![allow(dead_code)]
#![allow(unused_imports)]
// #![warn(missing_docs)]

mod helpers;
mod data_management;
mod layers;
mod loss_functions;
mod optimizers;

mod graphs;
mod examples;
mod profiling;

use examples::mnist;

fn main() {
    // profiling::benchmarks::conv2d::run();

    mnist::run();
}
