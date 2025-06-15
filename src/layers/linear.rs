use rand::Rng;
use ndarray::{Array2, ArrayView2, Axis, Ix2};
use serde::{Deserialize, Serialize};

use crate::helpers::initialize_weights::kaiming_normal;

use super::{RawLayer, LearnableParameter, ParameterGroup};

#[derive(Debug, Serialize, Deserialize)]
pub struct Linear {
    weights: ParameterGroup<Ix2>,
    bias: ParameterGroup<Ix2>,
}

impl Linear {
    pub fn new_from_params(
        weights: Array2<f32>,
        bias: Vec<f32>,
    ) -> Linear {
        let outputs = weights.dim().0;
        let weights = ParameterGroup::new(weights);
        let bias = ParameterGroup::new(
            Array2::from_shape_vec((1, outputs), bias)
            .expect("Mismatch between bias and weight shape in layer initialization!")
        );
        Linear {
            weights,
            bias,
        }
    }

    pub fn new_from_rand(
        inputs: usize, 
        outputs: usize, 
    ) -> Linear {
        let weights = ParameterGroup::new(
            kaiming_normal((outputs, inputs), 1)
        );
        let bias = ParameterGroup::new(
            Array2::zeros((1, outputs))
        );
        Linear {
            weights,
            bias,
        }
    }
}

impl RawLayer for Linear {
    type Input = Ix2;
    type Output = Ix2;

    fn forward(&mut self, input: &Array2<f32>, _train: bool) -> Array2<f32> {
        input.dot(&self.weights.values.t()) + &self.bias.values
    }

    // Here, we'll be fed the delta after the activation derivative has been applied,
    // since the activation functions will handle that portion themselves
    fn backward(&mut self, delta: &Array2<f32>, forward_input: &Array2<f32>) -> Array2<f32> {

        // Accumulate gradients in training
        self.weights.gradients += &forward_input.t().dot(delta).t();
        self.bias.gradients    += &delta.sum_axis(Axis(0)).insert_axis(Axis(1)).t();

        // Propagate the signal backward to the previous layer
        delta.dot(&self.weights.values)
    }

    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> {
        vec![
            self.weights.as_learnable_parameter(), 
            self.bias.as_learnable_parameter(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;

    #[test]
    fn forward() {
        let weights = Array2::from_shape_vec((2, 3), vec![
            1., 0., -1.,
            0., 1., 2.,
        ]).unwrap();
        let bias = vec![1., 0.,];
        let mut linear = Linear::new_from_params(weights, bias);

        let input = Array2::<f32>::from_shape_vec((1, 3), vec![
            1., 2., 3.,
        ]).unwrap();
        let output = linear.forward(&input, false);

        let target = Array2::<f32>::from_shape_vec((1, 2), vec![
            -1., 8.,
        ]).unwrap();

        assert_eq!(output, target);
    }

    #[test]
    fn backward() {
        let weights = Array2::from_shape_vec((1, 3), vec![
            1., 0., -1.,
        ]).unwrap();
        let bias = vec![1.,];
        let mut linear = Linear::new_from_params(weights, bias);

        let input = Array2::<f32>::from_shape_vec((1, 3), vec![
            1., 2., 3.,
        ]).unwrap();
        linear.forward(&input, false);

        let error = Array2::<f32>::from_shape_vec((1, 1), vec![
            -1.,
        ]).unwrap();
        let error_signal = linear.backward(&error, &input);

        let target_signal = Array2::<f32>::from_shape_vec((1, 3), vec![
            -1., 0., 1.,
        ]).unwrap();

        assert_eq!(error_signal, target_signal);

        let target_weight_grads = Array2::<f32>::from_shape_vec((1, 3), vec![
            -1., -2., -3.,
        ]).unwrap();
        let target_bias_grads = Array2::<f32>::from_shape_vec((1, 1), vec![
            -1.,
        ]).unwrap();

        assert_eq!(linear.weights.gradients, target_weight_grads);
        assert_eq!(linear.bias.gradients, target_bias_grads);
    }
}