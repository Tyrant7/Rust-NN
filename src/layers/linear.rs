use rand::Rng;
use ndarray::{Array2, Axis};

use super::Layer;

pub struct Linear {
    weights: Array2<f32>,
    bias: Array2<f32>,
    forward_input: Option<Array2<f32>>,
    wgrads: Array2<f32>,
    bgrads: Array2<f32>,
    samples: usize,
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
            samples: 0,
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
        self.samples += 1;

        // Propagate the signal backward to the previous layer
        delta.dot(&self.weights)
    }

    fn apply_gradients(&mut self, step_size: f32) {
        if self.samples == 0 {
            panic!("apply_gradients() called with no gradients");
        }
        let wgrads = &self.wgrads / self.samples as f32;
        let bgrads = &self.bgrads / self.samples as f32;
        self.weights -= &(wgrads * step_size);
        self.bias -= &(bgrads * step_size);
    }

    fn zero_gradients(&mut self) {
        self.wgrads = Array2::zeros(self.weights.raw_dim());
        self.bgrads = Array2::zeros(self.bias.raw_dim());
        self.samples = 0;
    }
}
