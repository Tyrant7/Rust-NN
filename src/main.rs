use core::slice;

use ndarray::{Array2, Axis};
use rand::{seq::SliceRandom, Rng};

fn main() {
    let mut network = [
        Layer::new_from_rand(2, 3, relu, relu_derivative),
        Layer::new_from_rand(3, 1, relu, relu_derivative),
    ];

    let data = [
        ([0., 0.], 0.),
        ([1., 0.], 1.),
        ([0., 1.], 1.),
        ([1., 1.], 0.),
    ];

    let lr = 0.001;
    let epochs = 1000;
    let batch_size = 4;

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
        // data.shuffle(&mut rand::rng());
        let batch_iter = data.iter().skip(data.len() - batch_size);
        for (x, label) in batch_iter {
            let x = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
            let label = Array2::from_shape_fn((1, 1), |(_i, _j)| label);

            let mut forward_signal = x;
            for layer in network.iter_mut() {
                forward_signal = layer.forward(&forward_signal);
            }

            println!("output: {}, actual {}", forward_signal, label);

            let cost = &label - forward_signal;
            avg_cost += &cost.pow2().sum();

            // Cost derivative
            let mut error = cost * 2.;

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
            let w = wgrads.pop().unwrap();
            let b = bgrads.pop().unwrap();
            layer.weights += &(&w * lr);
            layer.bias += &(&b * lr);

            println!("Layer weights: \n{:?}", layer.weights);
            println!("Layer biases : \n{:?}", layer.bias);
        }

        println!("Epoch {} avg cost: {}", epc + 1, avg_cost / batch_size as f32)
    }
}

fn relu(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|x| x.max(0.))
}

fn relu_derivative(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|x| if x < 0. { 0. } else { 1. })
}

struct Layer {
    activation: fn(Array2<f32>) -> Array2<f32>,
    activation_derivative: fn(Array2<f32>) -> Array2<f32>,
    weights: Array2<f32>,
    bias: Array2<f32>,
    forward_input: Option<Array2<f32>>,
    forward_activations: Option<Array2<f32>>,
}

impl Layer {
    fn new_from_rand(
        inputs: usize, 
        outputs: usize, 
        activation: fn(Array2<f32>) -> Array2<f32>, 
        activation_derivative: fn(Array2<f32>) -> Array2<f32>
    ) -> Layer {
        let mut rng = rand::rng();
        let weights = Array2::from_shape_fn((outputs, inputs), |(_i, _j)| rng.random_range(-1.0..1.));
        let bias = Array2::from_shape_fn((1, outputs), |(_i, _j)| rng.random_range(-1.0..1.));
        Layer {
            activation,
            activation_derivative,
            weights,
            bias,
            forward_input: None,
            forward_activations: None,
        }
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward_input = Some(input.clone());
        let activation = (self.activation)(input.dot(&self.weights.t()) + &self.bias);
        self.forward_activations = Some(activation.clone());
        activation
    }

    fn backward(&mut self, error: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let forward_act = self.forward_activations.as_ref().expect("Backward called before forward");
        let forward_input = self.forward_input.as_ref().expect("Backward called before forward");

        let delta = error * (self.activation_derivative)(forward_act.clone());

        let wgrad = forward_input.t().dot(&delta);
        let bgrad = delta.sum_axis(Axis(0)).insert_axis(Axis(1));

        let new_error = delta.dot(&self.weights);

        (new_error, wgrad, bgrad)
    }
}

