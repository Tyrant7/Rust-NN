use ndarray::Array2;
use super::Layer;

pub struct Sigmoid;

impl Sigmoid {
    fn sigmoid(input: &Array2<f32>) -> Array2<f32> {
        input.clone().mapv_into(|x| 1. / (1. + (-x).exp()))
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Array2<f32>, _train: bool) -> Array2<f32> {
        Sigmoid::sigmoid(input)
    }

    fn backward(&mut self, error: &Array2<f32>, forward_z: &Array2<f32>) -> Array2<f32> {
        let sig = Sigmoid::sigmoid(forward_z);
        let activation_derivative = &sig * (1. - &sig);
        error * activation_derivative
    }
}
