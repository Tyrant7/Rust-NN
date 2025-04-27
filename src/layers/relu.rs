use super::{Layer, Tensor};

#[derive(Debug)]
pub struct ReLU;

impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor, _train: bool) -> Tensor {
        match input {
            Tensor::T2D(data) => Tensor::T2D(data.clone().mapv_into(|x| x.max(0.))),
            Tensor::T3D(data) => Tensor::T3D(data.clone().mapv_into(|x| x.max(0.))),
        }
    }

    fn backward(&mut self, error: &Tensor, forward_z: &Tensor) -> Tensor {
        let activation_derivative = forward_z.clone().mapv_into(|x| if x <= 0. { 0. } else { 1. });
        error * activation_derivative
    }
}
