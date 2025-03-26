use ndarray::Array2;
use super::Layer;

pub struct Sigmoid;

impl Sigmoid {
    fn sigmoid(input: &Array2<f32>) -> Array2<f32> {
        input.clone().mapv_into(|x| 1. / (1. + (-x).exp()))
    }
}

impl Layer for Sigmoid {
    type State = Array2<f32>;

    fn forward(&mut self, input: &Array2<f32>, train: bool) -> (Array2<f32>, Self::State) {
        (Sigmoid::sigmoid(input), input.clone())
    }

    fn backward(&mut self, error: &Array2<f32>, forward_z: Self::State) -> Array2<f32> {
        let sig = Sigmoid::sigmoid(&forward_z);
        let activation_derivative = &sig * (1. - &sig);
        error * activation_derivative
    }
}
