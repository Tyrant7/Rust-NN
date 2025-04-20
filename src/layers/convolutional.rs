use std::char::from_digit;

use rand::Rng;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, Axis};

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
        let mut output = Array3::zeros((batch_size, out_features, output_size));

        // Iterate over input samples across batch_size dimension
        for (b, sample) in padded_input.axis_iter(Axis(0)).enumerate() {

            // Iterate over input kernels across out_features dimension
            for kernels in self.kernels.axis_iter(Axis(0)) {

                // Iterate over input sample features across in_features dimension
                for (in_f, in_feature) in sample.axis_iter(Axis(0)).enumerate() {
                    let conv = convolve1d(in_feature, kernels.slice(s![in_f, ..]), self.stride);
                    output.slice_mut(s![b, in_f, ..]).assign(&conv);
                }
            }
        }
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
        unimplemented!();
        /*
        let (batch_size, in_features, sample_size) = forward_input.dim();
        let (out_features, _, kernel_size) = self.kernels.dim();

        // To calculate gradients
        self.kgrads = Array3::zeros(self.kernels.dim());

        // Iterate over error samples across batch_size dimension
        for (b, sample) in delta.axis_iter(Axis(0)).enumerate() {

            // Iterate over kernels across kernel_size dimension
            for (f, kernel) in self.kernels.axis_iter(Axis(0)).enumerate() {

                // Iterate over error sample features across in_features dimension
                for (in_f, in_feature) in sample.axis_iter(Axis(0)).enumerate() {

                    // Iterate over windows of error feature
                    let windows = forward_input.slice(s![b, in_f, ..]).windows_with_stride(kernel_size, self.stride);
                    for (i, window) in windows.into_iter().enumerate() {
                        self.kgrads[[f, in_f, i]] += window.dot(&sample[[f, i]]);
                    }
                }
            }
        }

        // Create the padded loss gradient for the full convolution to calculate propagated loss signal
        let padded_loss = {
            let padding = (kernel_size - 1) * 2;
            let mut padded = Array3::zeros((batch_size, out_features, sample_size + padding));
            padded.slice_mut(s![0..batch_size, 0..in_features, padding..sample_size + padding]).assign(delta);
            padded
        };

        let mut signal = Array3::zeros(forward_input.dim());

        // Iterate over input samples across batch_size dimension
        for (b, sample) in padded_loss.axis_iter(Axis(0)).enumerate() {

            // Iterate over kernels across kernel_size dimension
            for (f, kernel) in self.kernels.axis_iter(Axis(0)).enumerate() {
                let kernel = kernel.slice_move(s![..;-1, ..;-1]);

                // Iterate over input sample features across in_features dimension
                for (in_f, in_feature) in sample.axis_iter(Axis(0)).enumerate() {

                    // Iterate over windows of input feature
                    let windows = in_feature.windows_with_stride(kernel_size, self.stride);
                    for (i, window) in windows.into_iter().enumerate() {
                        signal[[b, f, i]] += (&window * &kernel.slice(s![in_f, ..])).sum();
                    }
                }
            }
        }

        println!("grads:  {}", self.kgrads);
        println!("signal: {}", signal);

        signal

        */
    }

    fn get_learnable_parameters(&mut self) -> Vec<Parameter> {
        // TODO
        unimplemented!();
    }
}

fn convolve1d(input: ArrayView1<f32>, kernel: ArrayView1<f32>, stride: usize) -> Array1<f32> {
    let output_size = ((input.dim() - kernel.dim()) / stride) + 1;
    let mut output = Array1::from_elem(output_size, 0.);
    let windows = input.windows_with_stride(kernel.dim(), stride);
    for (i, window) in windows.into_iter().enumerate() {
        output[i] += (&window * &kernel).sum();
    }
    output
}