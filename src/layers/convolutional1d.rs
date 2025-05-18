use rand::Rng;
use ndarray::{s, Array1, Array3, ArrayView1, Axis, Ix1, Ix2, Ix3, Ix4};

use super::{RawLayer, LearnableParameter, ParameterGroup};

#[derive(Debug)]
pub struct Convolutional1D {
    kernels: ParameterGroup<Ix3>,
    bias: Option<ParameterGroup<Ix1>>,

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

        let kernels = Array3::from_shape_fn((out_features, in_features, kernel_size), |_| 
            rng.random_range(-1.0..1.)
        );
        let bias = match use_bias {
            true => Some(Array1::from_shape_fn(out_features, |_| rng.random_range(-1.0..1.))),
            false => None,
        };
        
        Convolutional1D::new_from_kernels(kernels, bias, stride, padding)
    }

    pub fn new_from_kernels(
        kernels: Array3<f32>, 
        bias: Option<Array1<f32>>,
        stride: usize, 
        padding: usize,
    ) -> Self {
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
    
    /// Expected input shape: (batch_size, features, width)
    fn forward(&mut self, input: &Array3<f32>, _train: bool) -> Array3<f32> {
        let (batch_size, in_features, width) = input.dim();

        // Pad the input
        // (batch_size, in_features, width)
        let input = {
            if self.padding > 0 {
                let mut padded = Array3::zeros((batch_size, in_features, width + self.padding * 2));
                padded.slice_mut(s![0..batch_size, 0..in_features, self.padding..width + self.padding]).assign(input);
                padded
            } else {
                input.to_owned()
            }
        };

        // 1D convolution
        let (out_features, _, kernel_width) = self.kernels.values.dim();
        let output_width = ((width - kernel_width + (2 * self.padding)) / self.stride) + 1;
        let mut output = Array3::<f32>::zeros((batch_size, out_features, output_width));
        for b in 0..batch_size {
            for out_f in 0..out_features {
                for in_f in 0..in_features {
                    let input_slice = input.slice(s![b, in_f, ..]);
                    let kernel_slice = self.kernels.values.slice(s![out_f, in_f, ..]);

                    let conv = convolve1d(input_slice, kernel_slice, self.stride);

                    output
                        .slice_mut(s![b, out_f, ..])
                        .scaled_add(1., &conv);
                }
            }
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

        // Compute kernel gradients
        for b in 0..batch_size {
            for out_f in 0..out_features {
                for in_f in 0..in_features {
                    // Align error slice with input slice
                    let input_slice = forward_input.slice(s![b, in_f, ..]);
                    let error_slice = delta.slice(s![b, out_f, ..]);

                    // In some cases, the loss may actually be larger than the input due to padding
                    // In these cases, we can swap the kernel and input to achieve the desired result
                    // without causing a shape error
                    let grad = if error_slice.dim() < input_slice.dim() {
                        convolve1d(input_slice, error_slice, 1)
                    } else {
                        convolve1d(error_slice, input_slice, 1)
                    };
                    self.kernels.gradients
                        .slice_mut(s![out_f, in_f, ..])
                        .scaled_add(1., &grad);
                }
            }
        }

        // Compute bias gradients
        if let Some(bias) = &mut self.bias { 
            for out_f in 0..out_features {
                bias.gradients[out_f] += delta.slice(s![.., out_f, ..]).sum();
            }
        }

        // Compute loss signal for backpropagation
        let output_width = ((input_width - kernel_width + (2 * self.padding)) / self.stride) + 1;
        let signal_width = output_width + kernel_width - 1;
        let mut error_signal = Array3::zeros((batch_size, in_features, signal_width));
        for b in 0..batch_size {
            for out_f in 0..out_features {
                for in_f in 0..in_features {
                    let delta_slice = delta.slice(s![b, out_f, ..]);

                    // Flip over width dimension (180 rotation)
                    let kernel_slice = self.kernels.values.slice(s![out_f, in_f, ..;-1]);

                    let padded = pad_1d(&delta_slice, kernel_slice.dim() - 1);
                    let conv = convolve1d(padded.view(), kernel_slice, 1);
                    error_signal
                        .slice_mut(s![b, in_f, ..])
                        .scaled_add(1., &conv);
                }
            }
        }

        // We need to crop the error signal to account for the padding added during the forward pass.
        // In the case padding was added there will be extra error values mapping to those positions, 
        // however they are not important for calculating the previous layer's error since they were
        // added to the data by this layer during the forward pass
        let crop = signal_width - input_width;
        let left = crop / 2;
        let right = crop - left;
        error_signal.slice(s![.., .., left..signal_width - right]).to_owned()
    }

    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> {
        let mut params = vec![self.kernels.as_learnable_parameter()];
        if let Some(bias) = &mut self.bias {
            params.push(bias.as_learnable_parameter());
        }
        params
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let kernels = Array3::from_shape_fn((2, 2, 2), |(k, _in, _i)| if k == 0 { 1. } else { 2. });
        let mut conv = Convolutional1D::new_from_kernels(kernels, None, 1, 0);

        let input = Array3::<f32>::from_shape_vec((1, 2, 7), vec![
            0., 1., 2., 3., 4., 5., 6.,
            0., 2., 4., 6., 8., 10., 12.,
        ]).unwrap();
        let output = conv.forward(&input, false);
        
        let target = Array3::<f32>::from_shape_vec((1, 2, 6), vec![
            3.,  9., 15., 21., 27., 33.,
            6., 18., 30., 42., 54., 66.,
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