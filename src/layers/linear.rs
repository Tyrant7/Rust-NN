use rand::Rng;
use ndarray::{Array2, Axis};

use super::Layer;

pub struct Linear {
    activation: fn(Array2<f32>) -> Array2<f32>,
    activation_derivative: fn(Array2<f32>) -> Array2<f32>,
    weights: Array2<f32>,
    bias: Array2<f32>,
    forward_input: Option<Array2<f32>>,
    forward_z: Option<Array2<f32>>,
    wgrads: Array2<f32>,
    bgrads: Array2<f32>,
}

impl Linear {
    pub fn new_from_rand(
        inputs: usize, 
        outputs: usize, 
        activation: fn(Array2<f32>) -> Array2<f32>, 
        activation_derivative: fn(Array2<f32>) -> Array2<f32>
    ) -> Linear {
        let mut rng = rand::rng();
        let weights = Array2::from_shape_fn((outputs, inputs), |(_i, _j)| rng.random_range(-1.0..1.));
        let bias = Array2::from_shape_fn((1, outputs), |(_i, _j)| rng.random_range(-1.0..1.));
        let wgrads = Array2::zeros(weights.raw_dim());
        let bgrads = Array2::zeros(bias.raw_dim());
        Linear {
            activation,
            activation_derivative,
            weights,
            bias,
            forward_input: None,
            forward_z: None,
            wgrads,
            bgrads,
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

    fn backward(&mut self, error: &Array2<f32>) -> Array2<f32> {
        let forward_z = self.forward_z.as_ref().expect("Backward called before forward");
        let forward_input = self.forward_input.as_ref().expect("Backward called before forward");

        let delta = error * (self.activation_derivative)(forward_z.clone());

        // Accumulate gradients
        self.wgrads += &forward_input.t().dot(&delta).t();
        self.bgrads += &delta.sum_axis(Axis(0)).insert_axis(Axis(1)).t();

        delta.dot(&self.weights)
    }

    fn apply_gradients(&mut self, lr: f32, batch_size: usize) {
        let wgrads = &self.wgrads / batch_size as f32;
        let bgrads = &self.bgrads / batch_size as f32;
        self.weights -= &(wgrads * lr);
        self.bias -= &(bgrads * lr);
    }

    fn zero_gradients(&mut self) {
        self.wgrads = Array2::zeros(self.weights.raw_dim());
        self.bgrads = Array2::zeros(self.bias.raw_dim());
    }
}

