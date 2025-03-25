use ndarray::Array2;

mod model;
use model::Model;

mod layers;
use layers::Linear;

use layers::ReLU;
use layers::Sigmoid;

fn main() {
    let mut network = Model::new(vec![
        Box::new(Linear::new_from_rand(2, 16)),
        Box::new(ReLU::new()),
        Box::new(Linear::new_from_rand(16, 1)),
        Box::new(Sigmoid::new()),
    ]);

    let data = [
        ([0., 0.], 0.),
        ([1., 0.], 1.),
        ([0., 1.], 1.),
        ([1., 1.], 0.),
    ];

    let lr = 0.01;
    let epochs = 100000;

    for epc in 0..epochs {
        let mut avg_cost = 0.;

        // Iterate over our entire dataset to collect gradients before applying them
        for (x, label) in data.iter() {
            let x = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
            let label = Array2::from_shape_fn((1, 1), |(_i, _j)| *label);

            let pred = network.forward(x);

            println!("output: {}, actual {}", pred, label);

            let cost = binary_cross_entroy_loss(&pred, &label);
            avg_cost += cost;

            // Cost derivative
            let mut error = binary_cross_entroy_loss_derivative(&pred, &label);

            // Back propagation
            for layer in network.iter_mut().rev() {
                error = layer.backward(&error);
            }
        }

        // Gradient application
        for layer in network.iter_mut() {
            layer.apply_gradients(lr, data.len());
            layer.zero_gradients();
        }

        println!("Epoch {} avg cost: {}", epc + 1, avg_cost / data.len() as f32)
    }
}

fn binary_cross_entroy_loss(pred: &Array2<f32>, label: &Array2<f32>) -> f32 {
    // To prevent log(0)
    let epsilon = 1e-12;
    -(label * (pred + epsilon).ln() + (1. - label) * (1. - pred + epsilon).ln()).sum()
}

fn binary_cross_entroy_loss_derivative(pred: &Array2<f32>, label: &Array2<f32>) -> Array2<f32> {
    let epsilon = 1e-12;
    (pred - label) / ((pred * (1. - pred)) + epsilon)
}

