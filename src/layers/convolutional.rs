use std::char::from_digit;

use rand::Rng;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView3, Axis};

use super::{Layer, Parameter};

pub struct Convolutional1D
{
    kernels: Array3<f32>,
    bias: Option<Array1<f32>>,

    kgrads: Array3<f32>,
    bgrads: Option<Array1<f32>>,

    stride: usize,
    padding: usize,
}

impl Convolutional1D {
    pub fn new_from_rand(
        in_features: usize, 
        out_features: usize, 
        kernel_size: usize, 
        use_bias: bool,
        stride: usize, 
        padding: usize,
    ) -> Self {
        let mut rng = rand::rng();

        // let kernels = Array3::from_shape_fn((out_features, in_features, kernel_size), |_| rng.random_range(-1.0..1.));
        let kernels = Array3::from_shape_fn((out_features, in_features, kernel_size), |_| 1.);
        let bias = match use_bias {
            true => Some(Array1::from_shape_fn(out_features, |_| rng.random_range(-1.0..1.))),
            false => None,
        };
        
        Convolutional1D::new_from_kernel(kernels, bias, stride, padding)
    }

    pub fn new_from_kernel(
        kernels: Array3<f32>, 
        bias: Option<Array1<f32>>,
        stride: usize, 
        padding: usize,
    ) -> Self {
        let kgrads = Array3::zeros(kernels.raw_dim());
        let bgrads = bias.as_ref().map(|b| Array1::zeros(b.raw_dim()));
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
        // (batch_size, in_features, width)
        let padded_input = {
            if self.padding > 0 {
                let mut padded = Array3::zeros((batch_size, in_features, sample_size + self.padding * 2));
                padded.slice_mut(s![0..batch_size, 0..in_features, self.padding..sample_size + self.padding]).assign(input);
                padded
            } else {
                input.clone()
            }
        };

        // 1D convolution
        let (out_features, _, kernel_size) = self.kernels.dim();
        let output_size = ((sample_size - kernel_size + (2 * self.padding)) / self.stride) + 1;
        let mut output = conv1d_with_batch_and_features(
            &padded_input, 
            &self.kernels, 
            (batch_size, out_features, output_size), 
            self.stride
        );

        // Apply bias to the second dimension (features)
        if let Some(b) = &self.bias {
            output += &b.view()
                .insert_axis(Axis(0))
                .insert_axis(Axis(2))
                .broadcast(output.dim())
                .unwrap();
        }

        output
    }

    pub fn backward(&mut self, delta: &Array3<f32>, forward_input: &Array3<f32>) -> Array3<f32> {
        let (batch_size, in_features, width) = forward_input.dim();
        let (out_features, _, kernel_size) = self.kernels.dim();

        // Compute kernel gradients
        self.kgrads = Array3::zeros(self.kernels.dim());
        for b in 0..batch_size {
            for out_f in 0..out_features {
                for in_f in 0..in_features {
                    // Align error slice with input slice
                    let input_slice = forward_input.slice(s![b, in_f, ..]);
                    let error_slice = delta.slice(s![b, out_f, ..]);

                    // 1D convolution
                    let grad = convolve1d(input_slice, error_slice, self.stride);
                    self.kgrads
                        .slice_mut(s![out_f, in_f, ..])
                        .scaled_add(1., &grad);
                }
            }
        }

        println!("kernels:{:?}", self.kernels);
        println!("kgrads: {}", self.kgrads);

        // Compute loss signal for backpropagation
        let mut error_signal = Array3::zeros(forward_input.dim());
        for b in 0..batch_size {
            for in_f in 0..in_features {
                for out_f in 0..out_features {
                    let delta_slice = delta.slice(s![b, out_f, ..]);

                    // Flip over width dimension (180 rotation)
                    let kernel_slice = self.kernels.slice(s![out_f, in_f, ..;-1]);

                    let padded = pad_1d(&delta_slice, kernel_slice.dim() - 1);
                    let conv = convolve1d(padded.view(), kernel_slice, self.stride);
                    error_signal
                        .slice_mut(s![b, in_f, ..])
                        .scaled_add(1., &conv);
                }
            }
        }
        error_signal
    }

    fn get_learnable_parameters(&mut self) -> Vec<Parameter> {
        // TODO
        unimplemented!();
    }
}

/// Dimensions:
/// input:   (batch_size, in_features, width)
/// kernels: (out_features, in_features, width)
fn conv1d_with_batch_and_features(input: &Array3<f32>, kernels: &Array3<f32>, output_dim: (usize, usize, usize), stride: usize) -> Array3<f32> {
    let mut output = Array3::zeros(output_dim);

    // (batch_size, in_features, width) -> (in_features, width)[] for the input
    for (b, batch) in input.axis_iter(Axis(0)).enumerate() {

        // (out_features, in_features, width) -> (in_features, width)[] for the kernels
        for kernel in kernels.axis_iter(Axis(0)) {

            // (in_features, width) -> (width)[] for input samples within each batch
            // We'll use the same index to index our kernels as we do to assign our input features
            for (in_f, in_feature) in batch.axis_iter(Axis(0)).enumerate() {

                // Convolve each input feature with each kernel
                let conv = convolve1d(in_feature, kernel.slice(s![in_f, ..]), stride);
                output.slice_mut(s![b, in_f, ..]).assign(&conv);
            }
        }
    }
    output
}

fn pad_1d(input: &ArrayView1<f32>, padding: usize) -> Array1<f32> {
    assert!(padding > 0);

    let mut padded = Array1::zeros(input.dim() + padding * 2);
    padded
        .slice_mut(s![padding..input.dim() + padding])
        .assign(input);
    padded
}

fn convolve1d(input: ArrayView1<f32>, kernel: ArrayView1<f32>, stride: usize) -> Array1<f32> {
    let output_size = ((input.dim() - kernel.dim()) / stride) + 1;
    let mut output = Array1::zeros(output_size);
    let windows = input.windows_with_stride(kernel.dim(), stride);
    for (i, window) in windows.into_iter().enumerate() {
        output[i] += (&window * &kernel).sum();
    }
    output
}