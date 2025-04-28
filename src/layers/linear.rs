use rand::Rng;
use ndarray::{Array2, Axis};

use super::{Layer, ParameterGroup, Tensor};

#[derive(Debug)]
pub struct Linear {
    weights: ParameterGroup,
    bias: ParameterGroup,
}

impl Linear {
    pub fn new_from_rand(
        inputs: usize, 
        outputs: usize, 
    ) -> Linear {
        let mut rng = rand::rng();

        let weights = ParameterGroup::new(
            Tensor::T2D(
                Array2::from_shape_fn((outputs, inputs), |_| rng.random_range(-1.0..1.))
            )
        );
        let bias = ParameterGroup::new(
            Tensor::T2D(
                Array2::from_shape_fn((1, outputs), |_| rng.random_range(-1.0..1.))
            )
        );
        Linear {
            weights,
            bias,
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Tensor, _train: bool) -> Tensor {
        Tensor::T2D(
            input.as_array2d().dot(&self.weights.values.as_array2d().t()) + self.bias.values.as_array2d()
        )
    }

    // Here, we'll be fed the delta after the activation derivative has been applied,
    // since the activation functions will handle that portion themselves
    fn backward(&mut self, delta: &Tensor, forward_input: &Tensor) -> Tensor {

        let delta = delta.as_array2d();

        // Accumulate gradients in training
        *self.weights.gradients.as_array2d_mut() += &forward_input.as_array2d().t().dot(delta).t();
        *self.bias.gradients.as_array2d_mut()    += &delta.sum_axis(Axis(0)).insert_axis(Axis(1)).t();

        // Propagate the signal backward to the previous layer
        Tensor::T2D(
            delta.dot(self.weights.values.as_array2d())
        )
    }

    fn get_learnable_parameters(&mut self) -> Vec<&mut ParameterGroup> {
        vec![
            &mut self.weights, 
            &mut self.bias,
        ]
    }
}
