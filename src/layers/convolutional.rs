use rand::Rng;
use ndarray::{Array1, Array2, Array3};

use super::{Layer, Parameter};

pub struct Convolutional1D
{
    kernels: Array3<f32>,
    bias: Array1<f32>,

    kgrads: Array3<f32>,
    bgrads: Array1<f32>,

    stride: usize,
    padding: usize,
}

impl Convolutional1D {
    pub fn new_from_rand(in_features: usize, out_features: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let mut rng = rand::rng();

        let kernels = Array3::from_shape_fn((kernel_size, in_features, out_features), |_| rng.random_range(-1.0..1.));
        let bias = Array1::from_shape_fn(out_features, |_| rng.random_range(-1.0..1.));
        let kgrads = Array3::zeros(kernels.raw_dim());
        let bgrads = Array1::zeros(bias.raw_dim());
        Convolutional1D { 
            kernels, 
            bias, 
            kgrads, 
            bgrads, 
            stride,
            padding,
        }
    }
}

impl Layer for Convolutional1D {
    fn forward(&mut self, input: &Array2<f32>, _train: bool) -> Array2<f32> {
        unimplemented!();
    }

    // Here, we'll be fed the delta after the activation derivative has been applied,
    // since the activation functions will handle that portion themselves
    fn backward(&mut self, delta: &Array2<f32>, forward_input: &Array2<f32>) -> Array2<f32> {
        unimplemented!();
    }

    fn get_learnable_parameters(&mut self) -> Vec<Parameter> {
        unimplemented!();
    }
}