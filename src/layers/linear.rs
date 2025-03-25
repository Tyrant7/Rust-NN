use rand::Rng;
use ndarray::{Array2, Axis};

use super::{Layer, Parameter};

pub struct Linear {
    weights: Array2<f32>,
    bias: Array2<f32>,
    forward_input: Option<Array2<f32>>,
    wgrads: Array2<f32>,
    bgrads: Array2<f32>,
}

impl Linear {
    pub fn new_from_rand(
        inputs: usize, 
        outputs: usize, 
    ) -> Linear {
        let mut rng = rand::rng();
        let weights = Array2::from_shape_fn((outputs, inputs), |(_i, _j)| rng.random_range(-1.0..1.));
        let bias = Array2::from_shape_fn((1, outputs), |(_i, _j)| rng.random_range(-1.0..1.));
        let wgrads = Array2::zeros(weights.raw_dim());
        let bgrads = Array2::zeros(bias.raw_dim());
        Linear {
            weights,
            bias,
            forward_input: None,
            wgrads,
            bgrads,
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward_input = Some(input.clone());
        input.dot(&self.weights.t()) + &self.bias
    }

    // Here, we'll be fed the delta after the activation derivative has been applied,
    // since the activation functions will handle that portion themselves
    fn backward(&mut self, delta: &Array2<f32>) -> Array2<f32> {
        let forward_input = self.forward_input.as_ref().expect("Backward called before forward");

        // Accumulate gradients
        self.wgrads += &forward_input.t().dot(delta).t();
        self.bgrads += &delta.sum_axis(Axis(0)).insert_axis(Axis(1)).t();

        // Propagate the signal backward to the previous layer
        delta.dot(&self.weights)
    }

    fn get_learnable_parameters(&mut self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for (weight, grad) in self.weights.iter_mut().zip(self.wgrads.iter_mut()) {
            params.push(Parameter {
                value: weight,
                gradient: grad
            });
        }
        for (bias, grad) in self.bias.iter_mut().zip(self.bgrads.iter_mut()) {
            params.push(Parameter { 
                value: bias, 
                gradient: grad 
            });
        }
        params
    }

    fn zero_gradients(&mut self) {
        self.wgrads = Array2::zeros(self.weights.raw_dim());
        self.bgrads = Array2::zeros(self.bias.raw_dim());
    }
}
