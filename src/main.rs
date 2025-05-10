#![allow(dead_code)]

use ndarray::{Array2, Array3};

use layers::Convolutional1D;
use layers::Layer;

//
mod model;
use model::Model;

//
mod layers;
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

    let mut conv_1d = Convolutional1D::new_from_rand(2, 3, 2, false, 1, 0);
    
    let input = Array3::from_shape_vec((1, 2, 7), 
    [0_f32, 1., 2., 3., 4., 5., 6.,
    0_f32, 2., 4., 6., 8., 10., 12.].to_vec()).unwrap();
    let output = conv_1d.forward(&input, true);
    
    println!("input:  {:?}", input);
    println!("output: {:?}", output);

    let error = Array3::from_shape_vec((1, 3, 6), 
    [3_f32, 3., 3., 3., 3., 3.,
    0_f32, 0., 0., 0., 0., 0.,
    0_f32, 0., 0., 0., 0., 0.].to_vec()).unwrap();
    let backward = conv_1d.backward(&error, &input);

    println!("errors: {:?}", error);
    println!("final:  {:?}", backward);

    panic!("Done");

    /* 
    let mut network = Model::new(vec![
        Box::new(Linear::new_from_rand(2, 16)),
        Box::new(ReLU),
        Box::new(Dropout::new(0.5, 5)),
        Box::new(Linear::new_from_rand(16, 1)),
        Box::new(Sigmoid),
    ]);

    let data = [
        ([0., 0.], 0.),
        ([1., 0.], 1.),
        ([0., 1.], 1.),
        ([1., 1.], 0.),
    ];

    let mut optimizer = SGD::new(&network.collect_parameters(), 0.01, 0.9);
    let epochs = 100000;

    for epc in 0..epochs {
        let mut avg_cost = 0.;

        // Iterate over our entire dataset to collect gradients before applying them
        for (x, label) in data.iter() {
            let x = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
            let label = Array2::from_shape_fn((1, 1), |_| *label);

            let pred = network.forward(x);

            let cost = MSELoss::original(&pred, &label);
            avg_cost += cost;

            // Back propagation
            network.backward(MSELoss::derivative(&pred, &label));
        }

        // Gradient application
        optimizer.step(&mut network.collect_parameters(), data.len());

        // Zero gradients before next epoch
        optimizer.zero_gradients(&mut network.collect_parameters());

        println!("Epoch {} avg cost: {}", epc + 1, avg_cost / data.len() as f32)
    }
    */
}
