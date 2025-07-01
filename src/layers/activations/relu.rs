use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

use crate::layers::RawLayer;

/// The Rectified Linear Unit (ReLU) activation function.
///
/// ReLU is one of the most widely used activation functions in deep learning, particularly in hidden layers.
/// It outputs the input value if positive, and zero otherwise, introducing non-linearity while being computationally efficient.
///
/// ReLU maps inputs to the range `[0, ∞)` and helps mitigate the vanishing gradietn problem that affects `sigmoid` and `tanh` activations.
///
/// The formula for the ReLU function is:
/// ```text
/// σ(x) = max(0, x)
/// ```
///
/// # Notes
/// - Derivative:
/// ```text
/// σ'(x) = {
///     1 if x > 0
///     0 if x ≤ 0
/// }
/// ```
/// - ReLU is not differentiable at `x = 0`, but in practice, the subgradient `0` or `1` is used during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReLU;

impl RawLayer for ReLU {
    type Input = IxDyn;
    type Output = IxDyn;

    fn forward(&mut self, input: &ArrayD<f32>, _train: bool) -> ArrayD<f32> {
        input.map(|x| x.max(0.))
    }

    fn backward(&mut self, error: &ArrayD<f32>, forward_z: &ArrayD<f32>) -> ArrayD<f32> {
        let activation_derivative = forward_z.map(|x| if *x <= 0. { 0. } else { 1. });
        activation_derivative * error
    }
}

#[cfg(test)]
mod tests {
    use crate::layers::tests::test_activation_fn;

    use super::*;

    #[test]
    fn relu() {
        test_activation_fn(
            ReLU,
            vec![-1., 0., 1.],
            vec![0., 0., 1.],
            vec![-1., 0., 1.],
            vec![0., 0., 1.],
        );
    }
}
