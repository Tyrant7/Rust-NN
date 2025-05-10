use ndarray::ArrayD;

use super::Layer;

#[derive(Debug)]
pub struct Sigmoid;

impl Sigmoid {
    fn sigmoid(input: &ArrayD<f32>) -> ArrayD<f32> {
        input.map(|x| 1. / (1. + (-x).exp()))
    }
}

impl Layer for Sigmoid {
    type Input = ArrayD<f32>;
    type Output = ArrayD<f32>;

    fn forward(&mut self, input: &Self::Input, _train: bool) -> Self::Output {
        Sigmoid::sigmoid(input)
    }

    fn backward(&mut self, error: &Self::Output, forward_z: &Self::Input) -> Self::Input {
        let sig = Sigmoid::sigmoid(forward_z);
        let one_minus = sig.map(|x| 1. - x);
        let activation_derivative = sig * &one_minus;
        activation_derivative * error
    }
}
