use rand::Rng;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, Axis, Ix1, Ix2, Ix3, Ix4};
use serde::{Deserialize, Serialize};

use crate::{helpers::conv_helpers::{convolve1d, crop_3d, pad_1d, pad_3d}, helpers::initialize_weights::{kaiming_normal, SeedMode}};

use super::{RawLayer, LearnableParameter, ParameterGroup};

/// A convolutional layer that handles 1D spatial data. 
/// 
/// Convolutional layers are common in many areas of machine learning, from image processing, to audio
/// and even language. They are the backbone of processing data with spatial importance, where the values
/// of inputs may be related to those of their neighbouring inputs. 
/// 
/// The shape of the output is given as follows:
/// 
/// ```text
/// (batch_size, out_features, output_width)
/// where
/// output_width = floor((width - kernel_width + 2 * self.padding) / self.stride) + 1;
/// ```
/// 
/// Convolutions occur between each feature in the input, and the kernels of this convolutional layer, and then have a bias
/// added to each output as follows:
/// 
/// ```text
/// for each batch:
///     for each output feature:
///         for each feature in the input:
///             output[batch, output_feature, ..] = 
///                 convolve(input[batch, input_feature, ..], kernels[output_feature, input_feature, ..]) + bias[output_feature]
/// ```
/// 
/// - Input data shape: `(batch_size, features, width)`
/// - Kernels shape: `(out_features, in_features, kernel_width)`
/// - Bias shape: `(out_features)`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Convolutional1D {
    kernels: ParameterGroup<Ix3>,
    bias: Option<ParameterGroup<Ix1>>,

    stride: usize,
    padding: usize,
}

impl Convolutional1D {
    /// Initializes a new [`Convolutional1D`] layer with random kernels of the given shape and zero bias (if enabled) using the Kaiming Normal initialization.  
    /// 
    /// This is the standard way to initialize a convolutional layer for training. 
    /// 
    /// # Arguments
    /// - `in_features`: Number of input features. 
    /// - `out_features`: Number of output features. 
    /// - `kernel_width`: The width of each kernel. 
    /// - `use_bias`: Whether or not bias should be added to each output. 
    /// - `stride`: The stride to use during the convolutions between the kernels and input features. 
    /// - `padding`: The padding to add to the input before convolutions are performed. 
    /// 
    /// # Panics
    /// - If any of `in_features`, `out_features`, or `kernel_width` are zero.  
    /// - If `stride` is zero.
    pub fn new_from_rand(
        in_features: usize, 
        out_features: usize, 
        kernel_width: usize, 
        use_bias: bool,
        stride: usize, 
        padding: usize,
    ) -> Self {
        assert!(in_features > 0, "Invalid input feature count: {in_features}");
        assert!(out_features > 0, "Invalid output feature count: {out_features}");
        assert!(kernel_width > 0, "Invalid kernel width: {kernel_width}");

        let kernels = kaiming_normal((out_features, in_features, kernel_width), 1, SeedMode::Random);
        let bias = match use_bias {
            true => Some(Array1::zeros(out_features)),
            false => None,
        };
        Convolutional1D::new_from_kernels(kernels, bias, stride, padding)
    }

    /// Initializes a new [`Convolutional1D`] layer with given kernels and bias (if enabled).  
    /// 
    /// # Parameters
    /// - `kernels`: The kernels to use for the convolution. Should have the shape `(out_features, in_features, kernel_width)`.
    /// - `bias`: The bias to add to each output feature. Should have the shape `(out_features)`.
    /// - `stride`: The stride to use during the convolutions between the kernels and input features. 
    /// - `padding`: The padding to add to the input before convolutions are performed. 
    /// 
    /// # Panics
    /// - If `bias` and `kernels` do not share the same number of input features when bias exists (first dimension length). 
    /// - If `stride` is zero.
    pub fn new_from_kernels(
        kernels: Array3<f32>, 
        bias: Option<Array1<f32>>,
        stride: usize, 
        padding: usize,
    ) -> Self {
        assert!(stride > 0, "Invalid stride given: {stride}");
        if let Some(b) = &bias {
            assert!(b.dim() == kernels.dim().0, "Shape mismatch between kernels and bias: {:?}, {:?}", kernels.dim(), b.dim());
        }

        let kernels = ParameterGroup::new(kernels);
        let bias = bias.map(ParameterGroup::new);
        Convolutional1D { 
            kernels, 
            bias, 
            stride, 
            padding 
        }
    }
}

impl RawLayer for Convolutional1D {
    type Input = Ix3;
    type Output = Ix3;
    
    // Expected input shape: (batch_size, features, width)
    fn forward(&mut self, input: &Array3<f32>, _train: bool) -> Array3<f32> {
        let (batch_size, in_features, width) = input.dim();

        // (batch_size, in_features, width)
        // We only care about padding the width dimension
        let input = pad_3d(&input.view(), (0, 0, self.padding));

        // 1D convolution
        let (out_features, _, kernel_width) = self.kernels.values.dim();
        let output_width = ((width - kernel_width + (2 * self.padding)) / self.stride) + 1;
        let mut batch_outputs = vec![Array2::<f32>::zeros((out_features, output_width)); batch_size];

        batch_outputs
            .iter_mut()
            .enumerate()
            .for_each(|(b, batch_output)| {
            for out_f in 0..out_features {
                for in_f in 0..in_features {
                    let input_slice = input.slice(s![b, in_f, ..]);
                    let kernel_slice = self.kernels.values.slice(s![out_f, in_f, ..]);
                    convolve1d(
                        input_slice, 
                        kernel_slice, 
                        &mut batch_output
                            .slice_mut(s![out_f, ..]), 
                        self.stride);
                }
            }
        });

        let mut output = Array3::<f32>::zeros((batch_size, out_features, output_width));
        for (b, batch) in batch_outputs.into_iter().enumerate() {
            output
                .slice_mut(s![b, .., ..])
                .assign(&batch);
        }

        // Apply bias to the second dimension (features)
        if let Some(b) = &self.bias {
            output += &b.values.view()
                .insert_axis(Axis(0))
                .insert_axis(Axis(2))
                .broadcast(output.dim())
                .unwrap();
        }
        output
    }

    fn backward(&mut self, delta: &Array3<f32>, forward_input: &Array3<f32>) -> Array3<f32> {
        let (batch_size, in_features, input_width) = forward_input.dim();
        let (out_features, _, kernel_width) = self.kernels.values.dim();

        let output_width = ((input_width - kernel_width + (2 * self.padding)) / self.stride) + 1;
        let signal_width = output_width + kernel_width - 1;

        // Compute kernel gradients and error signal for backpropagation in a single step to save performance
        let mut batch_signals = vec![Array2::<f32>::zeros((in_features, signal_width)); batch_size];
        let mut kernel_grads = vec![Array3::<f32>::zeros(self.kernels.gradients.dim()); batch_size];
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
                    let input_slice = forward_input.slice(s![b, in_f, ..]);
                    let error_slice = delta.slice(s![b, out_f, ..]);

                    // In some cases, the loss may actually be larger than the input due to padding
                    // In these cases, we can swap the kernel and input to achieve the desired result
                    // without causing a shape error
                    if error_slice.dim() < input_slice.dim() {
                        convolve1d(
                            input_slice, 
                            error_slice, 
                            &mut kernel_grad
                                .slice_mut(s![out_f, in_f, ..]), 
                            1
                        );
                    } else {
                        convolve1d(
                            error_slice,
                            input_slice,
                            &mut kernel_grad
                                .slice_mut(s![out_f, in_f, ..]),
                            1
                        );
                    };

                    // Error signal
                    // Flip over width dimension (180 rotation)
                    let kernel_slice = self.kernels.values.slice(s![out_f, in_f, ..;-1]);
                    let padded = pad_1d(&error_slice, kernel_slice.dim() - 1);
                    convolve1d(
                        padded.view(),
                        kernel_slice,
                        &mut batch_signal
                            .slice_mut(s![in_f, ..]),
                        1
                    );
                }

                // Compute bias gradients
                if let Some(bias) = &mut *bias_grad { 
                    bias[out_f] += delta.slice(s![b, out_f, ..]).sum();
                }
            }
        });

        // Collect full signals
        let mut error_signal = Array3::zeros((batch_size, in_features, signal_width));
        for (b, batch) in batch_signals.into_iter().enumerate() {
            error_signal
                .slice_mut(s![b, .., ..])
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
        crop_3d(&error_signal.view(), (0, 0, signal_width - input_width))
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
        // (out_feature, in_feature, width)
        let kernels = Array3::from_shape_vec((2, 2, 2), vec![
            // out 1 in 1
            1., 1.,
            // out 1 in 2
            1., 1.,
            // out 2 in 1
            2., 2.,
            // out 2 in 2
            2., 2.,
        ]).unwrap();
        let mut conv = Convolutional1D::new_from_kernels(kernels, None, 1, 0);

        // (batch, in_feature, width)
        let input = Array3::<f32>::from_shape_vec((1, 2, 6), vec![
            // feature 1
            0., 1., 2., 3., 4., 5.,
            // feature 2
            0., 2., 4., 6., 8., 10.,
        ]).unwrap();
        let output = conv.forward(&input, false);
        
        let target = Array3::<f32>::from_shape_vec((1, 2, 5), vec![
            // feature 1
            3.,  9., 15., 21., 27.,
            // feature 2
            6., 18., 30., 42., 54.,
        ]).unwrap();
        
        assert_eq!(output, target);
    }

    #[test]
    fn forward_stride_and_padding() {
        let kernels = Array3::from_shape_fn((2, 2, 2), |(k, _in, _i)| if k == 0 { 2. } else { 1. });
        let biases = Array1::from_elem(2, 1.);

        let mut conv = Convolutional1D::new_from_kernels(kernels, Some(biases), 2, 1);
    
        let input = Array3::<f32>::from_shape_vec((1, 2, 7), vec![
            0., 1., 2., 3., 4., 5., 8.,
            0.,-2.,-4.,-6.,-8.,-10.,-12.,
        ]).unwrap();
        let output = conv.forward(&input, false);
        
        let target = Array3::<f32>::from_shape_vec((1, 2, 4), vec![
            1.,-5.,-13.,-17.,
            1.,-2., -6., -8.,
        ]).unwrap();

        assert_eq!(output, target);
    }

    #[test]
    fn forward_multibatch() {
        // (out_features, in_features, width)
        let kernels = Array3::from_shape_vec((1, 1, 2), vec![
            1., 1.,
        ]).unwrap();
        let biases = Array1::from_elem(1, 1.);
        let mut conv = Convolutional1D::new_from_kernels(kernels, Some(biases), 1, 0);

        // (batch_size, features, width)
        let input = Array3::<f32>::from_shape_vec((2, 1, 3), vec![
            // Batch 1
            0., 1., 2., 

            // Batch 2
            -1.,-1.,-2., 
        ]).unwrap();
        let output = conv.forward(&input, false);
        
        let target = Array3::<f32>::from_shape_vec((2, 1, 2), vec![
            // Batch 1
            2., 4.,

            // Batch 2
            -1., -2.,
        ]).unwrap();
        
        assert_eq!(output, target);
    }

    #[test]
    fn backward() {
        let kernels = Array3::from_elem((2, 2, 2), 1.);
        let bias = Array1::ones(2);
        let mut conv = Convolutional1D::new_from_kernels(kernels, Some(bias), 1, 0);
    
        let input = Array3::<f32>::from_shape_vec((1, 2, 7), vec![
            0., 1., 2., 3., 4., 5.,  6.,
            0., 2., 4., 6., 8., 10., 12.,
        ]).unwrap();
        conv.forward(&input, false);

        let error = Array3::<f32>::from_shape_vec((1, 2, 6), vec![
            3., 3., 3., 3., 3., 3.,
            0., 0., 0., 0., 0., 0.,
        ]).unwrap();
        let error_signal = conv.backward(&error, &input);

        let target_signal = Array3::<f32>::from_shape_vec((1, 2, 7), vec![
            3., 6., 6., 6., 6., 6., 3.,
            3., 6., 6., 6., 6., 6., 3.,
        ]).unwrap();
        assert_eq!(error_signal, target_signal);

        let target_grads = Array3::<f32>::from_shape_vec((2, 2, 2), vec![
            45., 63., 
            90., 126., 
            0., 0., 
            0., 0.,
        ]).unwrap();
        assert_eq!(conv.kernels.gradients, target_grads);

        let target_b_grads = Array1::<f32>::from_shape_vec(2, vec![
            18., 0.,
        ]).unwrap();
        assert_eq!(conv.bias.unwrap().gradients, target_b_grads);
    }

    #[test]
    fn backward_padding() {
        let kernels = Array3::from_shape_vec((1, 1, 2), vec![
            1., 1.,
        ]).unwrap();
        let mut conv = Convolutional1D::new_from_kernels(kernels, None, 1, 1);
    
        let input = Array3::<f32>::from_shape_vec((1, 1, 2), vec![
            0., 1.,
        ]).unwrap();
        conv.forward(&input, false);

        let error = Array3::<f32>::from_shape_vec((1, 1, 3), vec![
            1., 2., 1.,
        ]).unwrap();
        let error_signal = conv.backward(&error, &input);

        let target_signal = Array3::<f32>::from_shape_vec((1, 1, 2), vec![
            3., 3.,
        ]).unwrap();
        assert_eq!(error_signal, target_signal);

        let target_grads = Array3::<f32>::from_shape_vec((1, 1, 2), vec![
            2., 1.,
        ]).unwrap();
        assert_eq!(conv.kernels.gradients, target_grads);
    }

    #[test]
    fn backward_stride_and_padding() {
        let kernels = Array3::from_shape_vec((1, 2, 2), vec![
            // in 1
            0., 1.,

            // in 2
            2.,-1.,
        ]).unwrap();
        let bias = Array1::ones(1);
        let mut conv = Convolutional1D::new_from_kernels(kernels, Some(bias), 2, 1);
    
        let input = Array3::<f32>::from_shape_vec((1, 2, 4), vec![
            // Feature 1
            0., 1., 2., 3.,

            // Feature 2
            0., 2., 4., 6.,
        ]).unwrap();
        conv.forward(&input, false);

        let error = Array3::<f32>::from_shape_vec((1, 1, 3), vec![
            // Feature 1
            3., 3., -1.,
        ]).unwrap();
        let error_signal = conv.backward(&error, &input);

        let target_signal = Array3::<f32>::from_shape_vec((1, 2, 4), vec![
            // Feature 1
            0., 3., 3.,-1.,

            // Feature 2
            6., 3.,-5., 1.,
        ]).unwrap();
        assert_eq!(error_signal, target_signal);

        let target_grads = Array3::<f32>::from_shape_vec((1, 2, 2), vec![
            // Kernel for in 1
            1., 6.,

            // Kernel for in 2
            2., 12.,
        ]).unwrap();
        assert_eq!(conv.kernels.gradients, target_grads);

        let target_b_grads = Array1::<f32>::from_shape_vec(1, vec![
            5.
        ]).unwrap();
        assert_eq!(conv.bias.unwrap().gradients, target_b_grads);
    }
}