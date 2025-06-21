use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

use super::RawLayer;

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
        test_activation_fn(ReLU, 
            vec![-1., 0., 1.,],
            vec![0., 0., 1.,],
            vec![-1., 0., 1.,],
            vec![0., 0., 1.,]
        );
    }
}