#![allow(dead_code)]
#![allow(unused_imports)]

use layers::chain;
use layers::CompositeLayer;
use layers::Tracked;
use ndarray::ArrayD;
use ndarray::{Array2, Array3};

use layers::Convolutional1D;
use layers::RawLayer;

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

fn main() {
    // Example usage of the library solving the XOR problem
    let mut network = chain!(
        Linear::new_from_rand(2, 16), 
        ReLU, 
        // Dropout::new(0.5, 5), 
        Linear::new_from_rand(16, 1),
        Sigmoid
    );

    let data = [
        ([0., 0.], 0.),
        ([1., 0.], 1.),
        ([0., 1.], 1.),
        ([1., 1.], 0.),
    ];

    let mut optimizer = SGD::new(&network.get_learnable_parameters(), 0.01, 0.9);
    let epochs = 100000;

    for epc in 0..epochs {
        let mut avg_cost = 0.;

        // Iterate over our entire dataset to collect gradients before applying them
        for (x, label) in data.iter() {
            let x = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
            let label = Array2::from_shape_fn((1, 1), |_| *label).into_dyn();

            let pred = network.forward(&x, true);

            let cost = MSELoss::original(&pred, &label);
            avg_cost += cost;

            // Back propagation
            network.backward(&MSELoss::derivative(&pred, &label));
        }

        // Gradient application
        optimizer.step(&mut network.get_learnable_parameters(), data.len());

        // Zero gradients before next epoch
        optimizer.zero_gradients(&mut network.get_learnable_parameters());

        println!("Epoch {} avg cost: {}", epc + 1, avg_cost / data.len() as f32)
    }
}
