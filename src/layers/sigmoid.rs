use ndarray::Array2;
use super::Layer;

pub struct Sigmoid;

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        input.clone().mapv_into(|x| 1. / (1. + (-x).exp()))
    }

    fn backward(&mut self, error: &Array2<f32>) -> Array2<f32> {
        let sig = self.forward(error);
        &sig * (1. - &sig)
    }
}
