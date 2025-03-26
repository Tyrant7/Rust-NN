use ndarray::Array2;
use super::Layer;

pub struct ReLU;

impl Layer for ReLU {
    type State = Array2<f32>;

    fn forward(&mut self, input: &Array2<f32>, train: bool) -> (Array2<f32>, Self::State) {
        (input.clone().mapv_into(|x| x.max(0.)), input.clone())
    }

    fn backward(&mut self, error: &Array2<f32>, forward_z: Self::State) -> Array2<f32> {
        let activation_derivative = forward_z.clone().mapv_into(|x| if x <= 0. { 0. } else { 1. });
        error * activation_derivative
    }
}
