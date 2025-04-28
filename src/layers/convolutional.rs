use rand::Rng;
use ndarray::{s, Array1, Array3, ArrayView1, Axis};

use super::{Layer, ParameterGroup, Tensor};

#[derive(Debug)]
pub struct Convolutional1D
{
    kernels: ParameterGroup,
    bias: Option<ParameterGroup>,

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
        let kernels = Tensor::T3D(Array3::from_shape_fn((out_features, in_features, kernel_size), |_| 1.));
        let bias = match use_bias {
            true => Some(Tensor::T1D(Array1::from_shape_fn(out_features, |_| rng.random_range(-1.0..1.)))),
            false => None,
        };
        
        Convolutional1D::new_from_kernel(kernels, bias, stride, padding)
    }

    pub fn new_from_kernel(
        kernels: Tensor, 
        bias: Option<Tensor>,
        stride: usize, 
        padding: usize,
    ) -> Self {
        let kernels = ParameterGroup::new(kernels);
        let bias = match bias {
            Some(b) => Some(ParameterGroup::new(b)),
            None => None
        };

        Convolutional1D { 
            kernels, 
            bias, 
            stride, 
            padding 
        }
    }
}

impl Layer for Convolutional1D {
    fn forward(&mut self, input: &Tensor, _train: bool) -> Tensor {
        let input = input.as_array3d();

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
        let (out_features, _, kernel_size) = self.kernels.values.as_array3d().dim();
        let output_width = ((width - kernel_size + (2 * self.padding)) / self.stride) + 1;
        let mut output = Array3::<f32>::zeros((batch_size, out_features, output_width));
        for b in 0..batch_size {
            for out_f in 0..out_features {
                for in_f in 0..in_features {
                    let input_slice = input.slice(s![b, in_f, ..]);
                    let kernel_slice = self.kernels.values.as_array3d().slice(s![out_f, in_f, ..]);

                    let conv = convolve1d(input_slice, kernel_slice, self.stride);

                    output
                        .slice_mut(s![b, out_f, ..])
                        .scaled_add(1., &conv);
                }
            }
        }

        // Apply bias to the second dimension (features)
        if let Some(b) = &self.bias {
            output += &b.values.as_array1d().view()
                .insert_axis(Axis(0))
                .insert_axis(Axis(2))
                .broadcast(output.dim())
                .unwrap();
        }

        Tensor::T3D(
            output
        )
    }

    fn backward(&mut self, delta: &Tensor, forward_input: &Tensor) -> Tensor {
        let delta = delta.as_array3d();
        let forward_input = forward_input.as_array3d();

        let (batch_size, in_features, _) = forward_input.dim();
        let (out_features, _, _) = self.kernels.values.as_array3d().dim();

        // Compute kernel gradients
        for b in 0..batch_size {
            for out_f in 0..out_features {
                for in_f in 0..in_features {
                    // Align error slice with input slice
                    let input_slice = forward_input.slice(s![b, in_f, ..]);
                    let error_slice = delta.slice(s![b, out_f, ..]);

                    // 1D convolution
                    let grad = convolve1d(input_slice, error_slice, self.stride);
                    self.kernels.gradients.as_array3d_mut()
                        .slice_mut(s![out_f, in_f, ..])
                        .scaled_add(1., &grad);
                }
            }
        }

        // TODO: Compute bias gradients

        // Compute loss signal for backpropagation
        let mut error_signal = Array3::zeros(forward_input.dim());
        for b in 0..batch_size {
            for out_f in 0..out_features {
                for in_f in 0..in_features {
                    let delta_slice = delta.slice(s![b, out_f, ..]);

                    // Flip over width dimension (180 rotation)
                    let kernel_slice = self.kernels.values.as_array3d().slice(s![out_f, in_f, ..;-1]);

                    let padded = pad_1d(&delta_slice, kernel_slice.dim() - 1);
                    let conv = convolve1d(padded.view(), kernel_slice, self.stride);
                    error_signal
                        .slice_mut(s![b, in_f, ..])
                        .scaled_add(1., &conv);
                }
            }
        }

        Tensor::T3D(
            error_signal
        )
    }

    fn get_learnable_parameters(&mut self) -> Vec<&mut ParameterGroup> {
        let mut params = vec![&mut self.kernels];
        if let Some(bias) = &mut self.bias {
            params.push(bias);
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