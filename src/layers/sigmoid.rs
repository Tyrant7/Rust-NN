use super::{Layer, Tensor};

#[derive(Debug)]
pub struct Sigmoid;

impl Sigmoid {
    fn sigmoid(input: &Tensor) -> Tensor {
        input.map(|x| 1. / (1. + (-x).exp()))
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Tensor, _train: bool) -> Tensor {
        Sigmoid::sigmoid(input)
    }

    fn backward(&mut self, error: &Tensor, forward_z: &Tensor) -> Tensor {
        let sig = Sigmoid::sigmoid(forward_z);
        let one_minus = sig.map(|x| 1. - x);
        let activation_derivative = sig * &one_minus;
        activation_derivative * error
    }
}
