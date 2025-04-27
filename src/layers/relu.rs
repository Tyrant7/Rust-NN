use super::{Layer, Tensor};

#[derive(Debug)]
pub struct ReLU;

impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor, _train: bool) -> Tensor {
        input.apply(|x| x.max(0.))
    }

    fn backward(&mut self, error: &Tensor, forward_z: &Tensor) -> Tensor {
        let activation_derivative = forward_z.apply(|x| if x <= 0. { 0. } else { 1. });
        activation_derivative * error
    }
}
