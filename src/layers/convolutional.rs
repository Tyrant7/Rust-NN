use rand::Rng;
use ndarray::{s, Array1, Array2, Array3, Axis};

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
    pub fn new_from_rand(
        in_features: usize, 
        out_features: usize, 
        kernel_size: usize, 
        stride: usize, 
        padding: usize
    ) -> Self {
        // let mut rng = rand::rng();
        // let kernels = Array3::from_shape_fn((out_features, in_features, kernel_size), |_| rng.random_range(-1.0..1.));
        let kernels = Array3::from_shape_fn((out_features, in_features, kernel_size), |_| 1.);
        Convolutional1D::new_from_kernel(out_features, kernels, stride, padding)
    }

    pub fn new_from_kernel(
        out_features: usize, 
        kernels: Array3<f32>, 
        stride: usize, 
        padding: usize
    ) -> Self {
        let mut rng = rand::rng();

        let bias = Array1::from_shape_fn(out_features, |_| rng.random_range(-1.0..1.));
        let kgrads = Array3::zeros(kernels.raw_dim());
        let bgrads = Array1::zeros(bias.raw_dim());
        Convolutional1D { 
            kernels, 
            bias, 
            kgrads, 
            bgrads, 
            stride, 
            padding 
        }
    }
}

impl /* Layer for */ Convolutional1D {
    pub fn forward(&mut self, input: &Array3<f32>, _train: bool) -> Array3<f32> {
        let (batch_size, in_features, sample_size) = input.dim();

        // Pad the input
        let padded_input = {
            if self.padding > 0 {
                let mut padded = Array3::zeros((batch_size, in_features + self.padding * 2, sample_size));
                padded.slice_mut(s![0..batch_size, 0..in_features, self.padding..sample_size - self.padding]).assign(input);
                padded
            } else {
                input.clone()
            }
        };

        // 1D convolution
        let (out_features, _, kernel_size) = self.kernels.dim();
        let output_size = ((sample_size - kernel_size + (2 * self.padding)) / self.stride) + 1;
        let mut output = Array3::zeros((batch_size, out_features, output_size));

        // Iterate over input samples across batch_size dimension
        for (b, sample) in padded_input.axis_iter(Axis(0)).enumerate() {

            // Iterate over kernels across kernel_size dimension
            for (f, kernel) in self.kernels.axis_iter(Axis(0)).enumerate() {

                // Iterate over input sample features across in_features dimension
                for (in_f, in_feature) in sample.axis_iter(Axis(0)).enumerate() {

                    // Iterate over windows of input feature
                    let windows = in_feature.windows_with_stride(kernel_size, self.stride);
                    for (i, window) in windows.into_iter().enumerate() {
                        output[[b, f, i]] += (&window * &kernel.slice(s![in_f, ..])).sum();
                    }
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