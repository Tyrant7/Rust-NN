use ndarray::{Array2, Axis};
use rand::Rng;

fn main() {
    let mut network = [
        Linear::new_from_rand(2, 16, relu, relu_derivative),
        Linear::new_from_rand(16, 1, sigmoid, sigmoid_derivative),
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

        // Accumulate gradients over all training examples
        let mut wgrads: Vec<Array2<f32>> = Vec::new();
        let mut bgrads: Vec<Array2<f32>> = Vec::new();
        for layer in network.iter().rev() {
            wgrads.push(Array2::zeros(layer.weights.raw_dim()));
            bgrads.push(Array2::zeros(layer.bias.raw_dim()));
        }

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
            for (i, layer) in network.iter_mut().rev().enumerate() {
                let wgrad: Array2<f32>;
                let bgrad: Array2<f32>;
                (error, wgrad, bgrad) = layer.backward(&error);
                
                if let Some(epoch_wgrad) = wgrads.get_mut(i) {
                    *epoch_wgrad += &wgrad.t();
                }
                if let Some(epoch_bgrad) = bgrads.get_mut(i) {
                    *epoch_bgrad += &bgrad.t();
                }
            }
        }

        // Gradient application
        for layer in network.iter_mut() {
            let w = wgrads.pop().unwrap() / data.len() as f32;
            let b = bgrads.pop().unwrap() / data.len() as f32;
            layer.weights -= &(&w * lr);
            layer.bias -= &(&b * lr);
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

pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, error: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>);
}

struct Linear {
    activation: fn(Array2<f32>) -> Array2<f32>,
    activation_derivative: fn(Array2<f32>) -> Array2<f32>,
    weights: Array2<f32>,
    bias: Array2<f32>,
    forward_input: Option<Array2<f32>>,
    forward_z: Option<Array2<f32>>,
}

impl Linear {
    fn new_from_rand(
        inputs: usize, 
        outputs: usize, 
        activation: fn(Array2<f32>) -> Array2<f32>, 
        activation_derivative: fn(Array2<f32>) -> Array2<f32>
    ) -> Linear {
        let mut rng = rand::rng();
        let weights = Array2::from_shape_fn((outputs, inputs), |(_i, _j)| rng.random_range(-1.0..1.));
        let bias = Array2::from_shape_fn((1, outputs), |(_i, _j)| rng.random_range(-1.0..1.));
        Linear {
            activation,
            activation_derivative,
            weights,
            bias,
            forward_input: None,
            forward_z: None,
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward_input = Some(input.clone());
        let z = input.dot(&self.weights.t()) + &self.bias;
        self.forward_z = Some(z.clone());
        (self.activation)(z)
    }

    fn backward(&mut self, error: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let forward_z = self.forward_z.as_ref().expect("Backward called before forward");
        let forward_input = self.forward_input.as_ref().expect("Backward called before forward");

        let delta = error * (self.activation_derivative)(forward_z.clone());

        let wgrad = forward_input.t().dot(&delta);
        let bgrad = delta.sum_axis(Axis(0)).insert_axis(Axis(1));

        let new_error = delta.dot(&self.weights);

        (new_error, wgrad, bgrad)
    }
}

