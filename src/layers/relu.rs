use ndarray::{ArrayD, IxDyn};

use super::RawLayer;

#[derive(Debug)]
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
