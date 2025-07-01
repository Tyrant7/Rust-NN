use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

use crate::layers::RawLayer;

/// The sigmoid activation function. 
/// 
/// Sigmoid maps real-value inputs to the range `[0, 1]`, and is commonly used for:
/// - Binary classification output layers (to represent probabilities),
/// - Gating mechanism in recurrent networks (e.g., LSTM gates). 
/// 
/// It smoothly squashes input values such that large positive inputs approach `1` and large negative
/// inputs approach `0`. 
/// 
/// The formula for the sigmoid function is:
/// ```text
/// σ(x) = 1 / (1 + exp(-x))
/// ```
/// 
/// # Notes
/// - Derivative: `σ'(x) = σ(x) * (1 - σ(x))`
/// - For multi-class classification, prefer `softmax` instead. 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sigmoid;

impl Sigmoid {
    fn sigmoid(input: &ArrayD<f32>) -> ArrayD<f32> {
        input.map(|x| 1. / (1. + (-x).exp()))
    }
}

impl RawLayer for Sigmoid {
    type Input = IxDyn;
    type Output = IxDyn;

    fn forward(&mut self, input: &ArrayD<f32>, _train: bool) -> ArrayD<f32> {
        Sigmoid::sigmoid(input)
    }

    fn backward(&mut self, error: &ArrayD<f32>, forward_z: &ArrayD<f32>) -> ArrayD<f32> {
        let sig = Sigmoid::sigmoid(forward_z);
        let one_minus = sig.map(|x| 1. - x);
        let activation_derivative = sig * &one_minus;
        activation_derivative * error
    }
}

#[cfg(test)]
mod tests {
    use crate::layers::tests::test_activation_fn;

    use super::*;

    #[test]
    fn sigmoid() {
        test_activation_fn(Sigmoid, 
            vec![-1., 0., 1.,],
            vec![0.268941, 0.5, 0.731059,],
            vec![-1., 0., 1.,],
            vec![-0.196612, 0., 0.196612,]
        );
    }
}