use rand::Rng;
use ndarray::{s, stack, Array1, Array2, Array3, Axis, Dim, Ix2, Ix3};

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

        // let kernels = Array3::from_shape_fn((out_features, in_features, kernel_size), |_| rng.random_range(-1.0..1.));
        let kernels = Array3::from_shape_fn((in_features, out_features, kernel_size), |_| 1.);
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

impl /* Layer for */ Convolutional1D {
    // Input shape: (batch_size, in_features)
    fn forward(&mut self, input: &Array3<f32>, _train: bool) -> Array3<f32> {
        let batch_size = input.dim().0;

        // Pad the input
        let padded_input = {
            if self.padding > 0 {
                let mut padded = Array3::zeros((batch_size, input.dim().1 + self.padding * 2, input.dim().2));
                padded.slice_mut(s![0..batch_size, self.padding..padded.dim().1 - self.padding, 0..input.dim().2]).assign(input);
                padded
            } else {
                input.clone()
            }
        };

        // 1D convolution
        let mut output = Array2::zeros((batch_size, self.kernels.dim().1));
        for (o, kernel) in self.kernels.outer_iter().enumerate() {

            println!("O: {}", o);
            println!("K: {:?}", kernel.dim());

            for b in 0..batch_size {
                let windows: Vec<ndarray::ArrayBase<ndarray::ViewRepr<&f32>, Dim<[usize; 2]>>> = padded_input
                    .windows_with_stride(kernel.raw_dim(), Ix3(1, self.stride, 1)).into_iter().collect();
                for window in windows {
                    output[[b, o]] += (&kernel * &window).sum();
                }
            }
        }
        output
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