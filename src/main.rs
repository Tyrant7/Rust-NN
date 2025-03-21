use ndarray::{Array, Array2, Axis};
use rand::Rng;

fn main() {
    let mut network = [
        Layer::new_from_rand(2, 5, relu, relu_derivative),
        Layer::new_from_rand(5, 3, relu, relu_derivative),
        Layer::new_from_rand(3, 1, relu, relu_derivative),
    ];

    let inputs = [
        [0., 0.],
        [1., 0.],
        [0., 1.],
        [1., 1.],
    ];
    let labels = [
        0., 
        1., 
        1., 
        0.,
    ];

    let lr = 0.005;
    let epochs = 1000;

    for epc in 0..epochs {
        let mut avg_cost = 0.;

        // Accumulate gradients over all training examples
        let mut wgrads: Vec<Array2<f32>> = Vec::new();
        let mut bgrads: Vec<Array2<f32>> = Vec::new();

        for (x, label) in inputs.iter().zip(labels.iter()) {
            let x = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
            let label = Array2::from_shape_fn((1, 1), |(_i, _j)| label);

            let mut forward_signal = x;
            for layer in network.iter_mut() {
                forward_signal = layer.forward(&forward_signal);
            }

            let mut error = &label - forward_signal;
            avg_cost += &error.pow2().sum();

            // Back propagation
            for (i, layer) in network.iter_mut().rev().enumerate() {
                let wgrad: Array2<f32>;
                let bgrad: Array2<f32>;
                (error, wgrad, bgrad) = layer.backward(&error);
                
                let new_epoch_wgrad = match wgrads.get(i) {
                    Some(epoch_grad) => epoch_grad + wgrad,
                    None => wgrad
                };
                wgrads.push(new_epoch_wgrad);

                let new_epoch_bgrad = match bgrads.get(i) {
                    Some(epoch_grad) => epoch_grad + bgrad,
                    None => bgrad
                };
                bgrads.push(new_epoch_bgrad);
            }

            // Gradient application
            for layer in network.iter_mut() {
                let w = wgrads.pop().unwrap();
                let b = bgrads.pop().unwrap();
                layer.weights += &(&w.t() * lr);
                layer.bias += &(&b.t() * lr);
            }
        }
        println!("Epoch avg cost: {}", avg_cost / labels.len() as f32)
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

