use rand::Rng;
use ndarray::{Array2, Axis, Ix2};

use super::{Layer, ParameterGroup};

#[derive(Debug)]
pub struct Linear {
    weights: ParameterGroup<Ix2>,
    bias: ParameterGroup<Ix2>,
}

impl Linear {
    pub fn new_from_rand(
        inputs: usize, 
        outputs: usize, 
    ) -> Linear {
        let mut rng = rand::rng();

        let weights = ParameterGroup::new(
            Array2::from_shape_fn((outputs, inputs), |_| rng.random_range(-1.0..1.))
        );
        let bias = ParameterGroup::new(
            Array2::from_shape_fn((1, outputs), |_| rng.random_range(-1.0..1.))
        );
        Linear {
            weights,
            bias,
        }
    }
}

impl Layer for Linear {
    type Input = Array2<f32>;
    type Output = Array2<f32>;

    fn forward(&mut self, input: &Self::Input, _train: bool) -> Self::Output {
        input.dot(&self.weights.values.t()) + &self.bias.values
    }

    // Here, we'll be fed the delta after the activation derivative has been applied,
    // since the activation functions will handle that portion themselves
    fn backward(&mut self, delta: &Self::Output, forward_input: &Self::Input) -> Self::Input {

        // Accumulate gradients in training
        self.weights.gradients += &forward_input.t().dot(delta).t();
        self.bias.gradients    += &delta.sum_axis(Axis(0)).insert_axis(Axis(1)).t();

        // Propagate the signal backward to the previous layer
        delta.dot(&self.weights.values)
    }

    fn get_learnable_parameters(&mut self) -> Vec<&mut ParameterGroup> {
        vec![
            &mut self.weights, 
            &mut self.bias,
        ]
    }
}
