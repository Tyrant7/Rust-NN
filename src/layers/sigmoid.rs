use ndarray::{ArrayD, IxDyn};

use super::RawLayer;

#[derive(Debug)]
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
