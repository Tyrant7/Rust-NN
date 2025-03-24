use ndarray::Array2;
use super::Layer;

pub struct ReLU;

impl Layer for ReLU {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        input.clone().mapv_into(|x| x.max(0.))
    }

    fn backward(&mut self, error: &Array2<f32>) -> Array2<f32> {
        error.clone().mapv_into(|x| if x <= 0. { 0. } else { 1. })
    }
}
