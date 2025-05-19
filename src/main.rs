#![allow(dead_code)]
#![allow(unused_imports)]

use std::thread;

use colored::Colorize;
use layers::chain;
use layers::CompositeLayer;
use layers::Flatten;
use layers::Tracked;
use ndarray::ArrayD;
use ndarray::{Array2, Array3};

use layers::Convolutional1D;
use layers::RawLayer;

pub mod helpers;
pub use helpers::conv_helpers;

//
mod layers;
use layers::Chain;
use layers::Linear;
use layers::Dropout;

use layers::ReLU;
use layers::Sigmoid;

//
mod loss_functions;
use loss_functions::LossFunction;
use loss_functions::MSELoss;

//
mod optimizers;
use optimizers::Optimizer;
use optimizers::SGD;

mod graphs;

#[tokio::main]
async fn main() {
    // Example usage of the library solving the XOR problem
    let mut network = chain!(
        Linear::new_from_rand(2, 16), 
        ReLU, 
        // Dropout::new(0.5, 5), 
        Linear::new_from_rand(16, 1),
        Sigmoid,
        Flatten::new(0),
    );

    let data = [
        ([0., 0.], 0.),
        ([1., 0.], 1.),
        ([0., 1.], 1.),
        ([1., 1.], 0.),
    ];

    let mut optimizer = SGD::new(&network.get_learnable_parameters(), 0.01, 0.9);
    let epochs = 10000;

    let mut avg_costs = Vec::new();
    let mut max_costs = Vec::new();

    let time = std::time::Instant::now();

    for epc in 0..epochs {
        let mut avg_cost = 0.;
        let mut max_cost: f32 = 0.;

        /* thread::spawn(|| { */
        // Iterate over our entire dataset to collect gradients before applying them
        for (x, label) in data.iter() {
            let x = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
            let label = Array2::from_shape_fn((1, 1), |_| *label).into_dyn();

            let pred = network.forward(&x, true);

            let cost = MSELoss::original(&pred, &label);
            avg_cost += cost;
            max_cost = cost.max(max_cost);

            // Back propagation
            network.backward(&MSELoss::derivative(&pred, &label));
        }
        /* }); */

        avg_costs.push(avg_cost);
        max_costs.push(max_cost);

        // Gradient application
        optimizer.step(&mut network.get_learnable_parameters(), data.len());

        // Zero gradients before next epoch
        optimizer.zero_gradients(&mut network.get_learnable_parameters());

        println!("Epoch {} avg cost: {}", epc + 1, avg_cost / data.len() as f32);
    }

    println!("{}", format!("Completed training in {} seconds", time.elapsed().as_secs()).green());

    println!("Generating costs chart");
    let _ = graphs::costs_candle(&avg_costs, &max_costs);
}
