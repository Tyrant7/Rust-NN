use std::sync::Mutex;

use rand::Rng;
use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, Axis, Ix1, Ix2, Ix3, Ix4};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{conv_helpers::{convolve2d, crop_4d, pad_2d, pad_4d}, helpers::initialize_weights::kaiming_normal};

use super::{RawLayer, LearnableParameter, ParameterGroup};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Convolutional2D {
    kernels: ParameterGroup<Ix4>,
    bias: Option<ParameterGroup<Ix1>>,

    stride: (usize, usize),
    padding: (usize, usize),
}

impl Convolutional2D {
    /// Pairs will follow order of (height, width)
    pub fn new_from_rand(
        in_features: usize, 
        out_features: usize, 
        kernel_size: (usize, usize), 
        use_bias: bool,
        stride: (usize, usize), 
        padding: (usize, usize),
    ) -> Self {
        let kernels = kaiming_normal((out_features, in_features, kernel_size.0, kernel_size.1), 1);
        let bias = match use_bias {
            true => Some(Array1::zeros(out_features)),
            false => None,
        };
        Convolutional2D::new_from_kernels(kernels, bias, stride, padding)
    }

    /// Stride and padding will follow order of (height, width)
    pub fn new_from_kernels(
        kernels: Array4<f32>, 
        bias: Option<Array1<f32>>,
        stride: (usize, usize), 
        padding: (usize, usize),
    ) -> Self {
        let kernels = ParameterGroup::new(kernels);
        let bias = bias.map(ParameterGroup::new);
        Convolutional2D { 
            kernels, 
            bias, 
            stride, 
            padding 
        }
    }
}

impl RawLayer for Convolutional2D {
    type Input = Ix4;
    type Output = Ix4;

    /// Expected input shape: (batch_size, features, height, width)
    fn forward(&mut self, input: &Array4<f32>, _train: bool) -> Array4<f32> {
        let (batch_size, in_features, height, width) = input.dim();

        // (batch_size, in_features, height, width)
        // We only care about padding the height and width dimensions
        let input = pad_4d(&input.view(), (0, 0, self.padding.0, self.padding.1));

        // 2D convolution
        let (out_features, _, kernel_height, kernel_width) = self.kernels.values.dim();
        let output_width = ((width - kernel_width + (2 * self.padding.1)) / self.stride.1) + 1;
        let output_height = ((height - kernel_height + (2 * self.padding.0)) / self.stride.0) + 1;
        let mut batch_outputs = vec![Array3::<f32>::zeros((out_features, output_height, output_width)); batch_size];
        
        batch_outputs
            .iter_mut()
            .enumerate()
            .for_each(|(b, batch_output)| {
            for out_f in 0..out_features {
                for in_f in 0..in_features {
                    let input_slice = input.slice(s![b, in_f, .., ..]);
                    let kernel_slice = self.kernels.values.slice(s![out_f, in_f, .., ..]);
                    convolve2d(
                        &input_slice, 
                        &kernel_slice, 
                        &mut batch_output
                            .slice_mut(s![out_f, .., ..]),
                        self.stride
                    );
                }
            }
        });

        let mut output = Array4::<f32>::zeros((batch_size, out_features, output_height, output_width));
        for (b, batch) in batch_outputs.into_iter().enumerate() {
            output
                .slice_mut(s![b, .., .., ..])
                .assign(&batch);
        }

        // Apply bias to the second dimension (features)
        if let Some(b) = &self.bias {
            output += &b.values.view()
                .insert_axis(Axis(0))
                .insert_axis(Axis(2))
                .insert_axis(Axis(2))
                .broadcast(output.dim())
                .unwrap();
        }
        output
    }

    fn backward(&mut self, delta: &Array4<f32>, forward_input: &Array4<f32>) -> Array4<f32> {        
        let (batch_size, in_features, input_height, input_width) = forward_input.dim();
        let (out_features, _, kernel_height, kernel_width) = self.kernels.values.dim();

        let output_width = ((input_width - kernel_width + (2 * self.padding.1)) / self.stride.1) + 1;
        let output_height = ((input_height - kernel_height + (2 * self.padding.0)) / self.stride.0) + 1;
        let signal_width = output_width + kernel_width - 1;
        let signal_height = output_height + kernel_height - 1;

        // Compute kernel gradients and error signal for backpropagation in a single step to save performance
        let mut batch_signals = vec![Array3::<f32>::zeros((in_features, signal_height, signal_width)); batch_size];
        let mut kernel_grads = vec![Array4::<f32>::zeros(self.kernels.gradients.dim()); batch_size];
        let mut bias_grads = if let Some(b) = &self.bias {
            vec![Some(Array1::<f32>::zeros(b.gradients.dim())); batch_size]
        } else {
            vec![None; batch_size]
        };
        batch_signals
            .iter_mut()
            .zip(kernel_grads.iter_mut())
            .zip(bias_grads.iter_mut())
            .enumerate()
            .for_each(|(b, ((batch_signal, kernel_grad), bias_grad))| {
            for out_f in 0..out_features {
                for in_f in 0..in_features {
                    // Kernel gradients
                    // Align error slice with input slice
                    let input_slice = forward_input.slice(s![b, in_f, .., ..]);
                    let error_slice = delta.slice(s![b, out_f, .., ..]);

                    // In some cases, the loss may actually be larger than the input due to padding
                    // In these cases, we can swap the kernel and input to achieve the desired result
                    // without causing a shape error
                    if error_slice.dim() < input_slice.dim() {
                        convolve2d(
                            &input_slice, 
                            &error_slice, 
                            &mut kernel_grad
                                .slice_mut(s![out_f, in_f, .., ..]),
                            (1, 1)
                        );
                    } else {
                        convolve2d(
                            &error_slice, 
                            &input_slice,
                            &mut kernel_grad
                                .slice_mut(s![out_f, in_f, .., ..]),
                            (1, 1)
                        );
                    };

                    // Error signal
                    // Flip over width and height dimensions (180 rotation)
                    let kernel_slice = self.kernels.values.slice(s![out_f, in_f, ..;-1, ..;-1]);
                    let padded = pad_2d(&error_slice, (kernel_height - 1, kernel_width - 1));
                    convolve2d(
                        &padded.view(), 
                        &kernel_slice, 
                        &mut batch_signal
                            .slice_mut(s![in_f, .., ..]), 
                        (1, 1)
                    );
                }

                // Compute bias gradients
                if let Some(bias) = bias_grad { 
                    bias[out_f] += delta.slice(s![b, out_f, .., ..]).sum();
                }
            }
        });

        // Collect full signals
        let mut error_signal = Array4::zeros((batch_size, in_features, signal_height, signal_width));
        for (b, batch) in batch_signals.into_iter().enumerate() {
            error_signal
                .slice_mut(s![b, .., .., ..])
                .assign(&batch);
        }
        for grad in kernel_grads.into_iter() {
            self.kernels.gradients += &grad;
        }
        if let Some(bias) = &mut self.bias {
            for grad in bias_grads.into_iter().flatten() {
                bias.gradients += &grad;
            }
        }

        // We need to crop the error signal to account for the padding added during the forward pass.
        // In the case padding was added there will be extra error values mapping to those positions, 
        // however they are not important for calculating the previous layer's error since they were
        // added to the data by this layer during the forward pass
        crop_4d(&error_signal.view(), (0, 0, signal_height - input_height, signal_width - input_width))
    }

    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> {
        let mut params = vec![self.kernels.as_learnable_parameter()];
        if let Some(bias) = &mut self.bias {
            params.push(bias.as_learnable_parameter());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        // (out_features, in_features, height, width)
        let kernels = Array4::from_shape_vec((2, 2, 2, 2), vec![
            // out 1, in 1
            1., 1.,
            1., 1.,

            // out 1, in 2
            1., 2.,
           -1., 1.,

            // out 2, in 1
            1., 1.,
            1., 1.,

            // out 2, in 2
            2., 1.,
            1., 2.,
        ]).unwrap();
        let mut conv = Convolutional2D::new_from_kernels(kernels, None, (1, 1), (0, 0));

        let input = Array4::<f32>::from_shape_vec((1, 2, 3, 3), vec![
            // Feature 1
            0., 1., 2., 
            3., 4., 5., 
            6., 7., 8.,

            // Feature 2
            0., 2., 4., 
            6., 8., 10., 
            12.,14.,16.,
        ]).unwrap();
        let output = conv.forward(&input, false);
        
        let target = Array4::<f32>::from_shape_vec((1, 2, 2, 2), vec![
            // Feature 1
            14., 24., 
            44., 54.,

            // Feature 2
            32., 48.,
            80., 96.,
        ]).unwrap();
        
        assert_eq!(output, target);
    }

    #[test]
    fn forward_stride_and_padding() {
        // (out_features, in_features, height, width)
        let kernels = Array4::from_shape_vec((1, 2, 2, 2), vec![
            // in 1
            1., 1.,
            1., 1.,

            // in 2
            1., 2.,
           -1., 1.,
        ]).unwrap();
        let biases = Array1::from_elem(1, 1.);
        let mut conv = Convolutional2D::new_from_kernels(kernels, Some(biases), (1, 2), (0, 1));

        let input = Array4::<f32>::from_shape_vec((1, 2, 3, 4), vec![
            // Feature 1
            0., 1., 2., 3., 
            4., 5., 6., 7., 
            8., 9., 10.,11.,

            // Feature 2
            0., 2., 4., 6., 
            8., 10.,12.,14.,
            16.,18.,20.,22.,
        ]).unwrap();
        let output = conv.forward(&input, false);
        
        let target = Array4::<f32>::from_shape_vec((1, 1, 2, 3), vec![
            13., 27., 3.,
            45., 67., 11.,
        ]).unwrap();
        
        assert_eq!(output, target);
    }

    #[test]
    fn forward_multibatch() {
        // (out_features, in_features, height, width)
        let kernels = Array4::from_shape_vec((1, 1, 2, 2), vec![
            // in 1
            1., 1.,
            1., 1.,
        ]).unwrap();
        let biases = Array1::from_elem(1, 1.);
        let mut conv = Convolutional2D::new_from_kernels(kernels, Some(biases), (1, 1), (0, 0));

        // (batch_size, features, height, width)
        let input = Array4::<f32>::from_shape_vec((2, 1, 2, 3), vec![
            // Batch 1
            0., 1., 2., 
            4., 5., 6., 

            // Batch 2
            0., 0., 0., 
            -1.,-1.,-2.,
        ]).unwrap();
        let output = conv.forward(&input, false);
        
        let target = Array4::<f32>::from_shape_vec((2, 1, 1, 2), vec![
            // Batch 1
            11., 15.,

            // Batch 2
            -1., -2.,
        ]).unwrap();
        
        assert_eq!(output, target);
    }

    #[test]
    fn backward() {
        // (out_features, in_features, height, width)
        let kernels = Array4::from_shape_vec((1, 2, 2, 2), vec![
            // in 1
            1., 1.,
            1., 1.,

            // in 2
            1., 2.,
           -1., 1.,
        ]).unwrap();
        let biases = Array1::from_elem(1, 1.);
        let mut conv = Convolutional2D::new_from_kernels(kernels, Some(biases), (1, 1), (0, 0));

        let input = Array4::<f32>::from_shape_vec((1, 2, 3, 3), vec![
            // Feature 1
            0., 1., 2., 
            3., 4., 5., 
            6., 7., 8.,

            // Feature 2
            0., 2., 4., 
            6., 8., 10., 
            12.,14.,16.,
        ]).unwrap();
        conv.forward(&input, false);

        let error = Array4::<f32>::from_shape_vec((1, 1, 2, 2), vec![
            // Feature 1
            1., 2., 
           -1.,-1.,
        ]).unwrap();
        let error_signal = conv.backward(&error, &input);

        let target_signal = Array4::<f32>::from_shape_vec((1, 2, 3, 3), vec![
            // Feature 1
            1., 3., 2., 
            0., 1., 1., 
           -1.,-2.,-1.,

            // Feature 2
            1., 4., 4., 
           -2.,-4., 0., 
            1., 0.,-1.,
        ]).unwrap();
        assert_eq!(error_signal, target_signal);

        let target_grads = Array4::<f32>::from_shape_vec((1, 2, 2, 2), vec![
            // Kernel for in 1
           -5., -4., 
           -2., -1., 

           // Kernel for in 2
           -10.,-8., 
           -4., -2.,
        ]).unwrap();
        assert_eq!(conv.kernels.gradients, target_grads);

        let target_b_grads = Array1::<f32>::from_shape_vec(1, vec![
            1.,
        ]).unwrap();
        assert_eq!(conv.bias.unwrap().gradients, target_b_grads);
    }

    #[test]
    fn backward_padding() {
        let kernels = Array4::from_shape_vec((1, 1, 2, 2), vec![
            1., 1.,
            1., 1.,
        ]).unwrap();
        let mut conv = Convolutional2D::new_from_kernels(kernels, None, (1, 1), (1, 1));

        let input = Array4::<f32>::from_shape_vec((1, 1, 2, 2), vec![
            0., 1., 
            2., 3., 
        ]).unwrap();
        conv.forward(&input, false);

        let error = Array4::<f32>::from_shape_vec((1, 1, 3, 3), vec![
            1., 2., 1.,
            2., 3., 2.,
            1., 2., 1.,
        ]).unwrap();
        let error_signal = conv.backward(&error, &input);

        let target_signal = Array4::<f32>::from_shape_vec((1, 1, 2, 2), vec![
            8., 8.,
            8., 8.,
        ]).unwrap();
        assert_eq!(error_signal, target_signal);

        // let target_grads = Array4::<f32>::from_shape_vec((1, 2, 2, 2), vec![
        //     // Kernel for in 1
        //    -7., -6., 
        //    -3., -2., 

        //    // Kernel for in 2
        //    -14.,-12., 
        //    -6., -4.,
        // ]).unwrap();
        // assert_eq!(conv.kernels.gradients, target_grads);
    }

    #[test]
    fn backward_stride_and_padding() {
        // (out_features, in_features, height, width)
        let kernels = Array4::from_shape_vec((1, 2, 2, 2), vec![
            // in 1
            1., 1.,
            1., 1.,

            // in 2
            1., 2.,
           -1., 1.,
        ]).unwrap();
        let biases = Array1::from_elem(1, 1.);
        let mut conv = Convolutional2D::new_from_kernels(kernels, Some(biases), (1, 2), (0, 1));

        let input = Array4::<f32>::from_shape_vec((1, 2, 3, 4), vec![
            // Feature 1
            0., 1., 2., 3., 
            4., 5., 6., 7., 
            8., 9.,10.,11.,

            // Feature 2
            0., 2., 4., 6., 
            8., 10.,12.,14.,
            16.,18.,20.,22.,
        ]).unwrap();
        conv.forward(&input, false);

        let error = Array4::<f32>::from_shape_vec((1, 1, 2, 3), vec![
            // Feature 1
            1., 2., 0.,
           -1.,-1., 0.,
        ]).unwrap();
        let error_signal = conv.backward(&error, &input);

        let target_signal = Array4::<f32>::from_shape_vec((1, 2, 3, 4), vec![
            // Feature 1
            1., 3., 2., 0., 
            0., 1., 1., 0.,
           -1.,-2.,-1., 0.,

            // Feature 2
            1., 4., 4., 0.,
           -2.,-4., 0., 0.,
            1., 0.,-1., 0.,
        ]).unwrap();
        assert_eq!(error_signal, target_signal);

        let target_grads = Array4::<f32>::from_shape_vec((1, 2, 2, 2), vec![
            // Kernel for in 1
           -7., -6., 
           -3., -2., 

           // Kernel for in 2
           -14.,-12., 
           -6., -4.,
        ]).unwrap();
        assert_eq!(conv.kernels.gradients, target_grads);

        let target_b_grads = Array1::<f32>::from_shape_vec(1, vec![
            1.,
        ]).unwrap();
        assert_eq!(conv.bias.unwrap().gradients, target_b_grads);
    }
}