use ndarray::ArrayD;

use super::Layer;

#[derive(Debug)]
pub struct ReLU;

impl Layer for ReLU {
    type Input = ArrayD<f32>;
    type Output = ArrayD<f32>;

    fn forward(&mut self, input: &Self::Input, _train: bool) -> Self::Output {
        input.map(|x| x.max(0.))
    }

    fn backward(&mut self, error: &Self::Output, forward_z: &Self::Input) -> Self::Input {
        let activation_derivative = forward_z.map(|x| if *x <= 0. { 0. } else { 1. });
        activation_derivative * error
    }
}
