use ndarray::Array2;

mod layers;
use layers::Layer;

fn main() {
    let mut network = [
        layers::Linear::new_from_rand(2, 16, relu, relu_derivative),
        layers::Linear::new_from_rand(16, 1, sigmoid, sigmoid_derivative),
    ];

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

        // Sample a random number of items from our training data to avoid converging to a local minimum
        for (x, label) in data.iter() {
            let x = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
            let label = Array2::from_shape_fn((1, 1), |(_i, _j)| *label);

            let mut forward_signal = x;
            for layer in network.iter_mut() {
                forward_signal = layer.forward(&forward_signal);
            }

            println!("output: {}, actual {}", forward_signal, label);

            let cost = binary_cross_entroy_loss(&forward_signal, &label);
            avg_cost += cost;

            // Cost derivative
            let mut error = binary_cross_entroy_loss_derivative(&forward_signal, &label);

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

fn relu(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|x| x.max(0.))
}

fn relu_derivative(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|x| if x <= 0. { 0. } else { 1. })
}

fn sigmoid(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|x| 1. / (1. + (-x).exp()))
}

fn sigmoid_derivative(input: Array2<f32>) -> Array2<f32> {
    let sig = sigmoid(input);
    &sig * (1. - &sig)
}

