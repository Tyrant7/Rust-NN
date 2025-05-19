use rand::Rng;
use ndarray::{s, Array1, Array2, Array3, ArrayD, ArrayView2, Axis, Ix2, Ix3, IxDyn};

use crate::layers::{ParameterGroup, RawLayer};

#[derive(Debug)]
pub struct MaxPool1D {
    kernel_width: usize,
    stride: usize,
    padding: usize,

    max_indices: Option<Array3<usize>>
}

impl MaxPool1D {
    fn new_full(
        kernel_width: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        assert!(kernel_width > 0, "Kernel width must be positive");
        assert!(stride > 0, "Stride must be positive");
        MaxPool1D { 
            kernel_width, 
            stride, 
            padding, 
            max_indices: None,
        }
    }

    fn new(kernel_width: usize) -> Self {
        Self::new_full(kernel_width, kernel_width, 0)
    }
}

impl RawLayer for MaxPool1D {
    type Input = Ix3;
    type Output = Ix3;

    /// input shape: (batch_size, features, width)
    fn forward(&mut self, input: &Array3<f32>, _train: bool) -> Array3<f32> {
        let (batch_size, in_features, width) = input.dim();
        let output_width = ((width - self.kernel_width + (2 * self.padding)) / self.stride) + 1;
        let mut output = Array3::zeros((batch_size, in_features, output_width));

        // We'll track which indices we selected for pooling to propagate error only through those
        // indices during the backward pass
        let mut max_indices = Array3::<usize>::zeros((batch_size, in_features, output_width));

        for b in 0..batch_size {
            for in_f in 0..in_features {
                let input_slice = input.slice(s![b, in_f, ..]);
                let windows = input_slice.windows_with_stride(self.kernel_width, self.stride);
                for (i, window) in windows.into_iter().enumerate() {
                    let (max_idx, max_val) = window.iter()
                        .enumerate()
                        .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();
                    output[[b, in_f, i]] = *max_val;
                    max_indices[[b, in_f, i]] = i * self.stride + max_idx;
                }
            }
        }
        self.max_indices = Some(max_indices);

        output
    }

    fn backward(&mut self, error: &Array3<f32>, forward_input: &Array3<f32>) -> Array3<f32> {
        let (batch_size, in_features, _) = forward_input.dim();
        let (_, _, error_width) = error.dim();
        let mut signal = Array3::zeros(forward_input.dim());
        let max_indices = self.max_indices
            .as_ref()
            .expect("No indices stored during forward pass or forward pass never called!");
        for b in 0..batch_size {
            for in_f in 0..in_features {
                for i in 0..error_width {
                    let idx = max_indices[[b, in_f, i]];
                    signal[[b, in_f, idx]] = error[[b, in_f, i]];
                }
            }
        }
        signal
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;

    #[test]
    fn forward() {
        let mut maxpool = MaxPool1D::new(2);
        
        let input = Array3::<f32>::from_shape_vec((1, 2, 4), vec![
            // Feature 1
            1., 2., 3., 4.,
            // Feature 2
            5., 4., 3., 2.,
        ]).unwrap();
        let output = maxpool.forward(&input, false);

        let target = Array3::<f32>::from_shape_vec((1, 2, 2), vec![
            // Feature 1
            2., 4.,
            // Feature 2
            5., 3.,
        ]).unwrap();

        assert_eq!(output, target);
    }

    #[test]
    fn backward() {
        let mut maxpool = MaxPool1D::new(2);
        
        let input = Array3::<f32>::from_shape_vec((1, 2, 4), vec![
            // Feature 1
            1., 2., 3., 4.,
            // Feature 2
            5., 4., 3., 2.,
        ]).unwrap();
        maxpool.forward(&input, false);

        let error = Array3::<f32>::from_shape_vec((1, 2, 2), vec![
            -1., 1.,
            -1., 2.,
        ]).unwrap();
        let error_signal = maxpool.backward(&error, &input);

        let target_signal = Array3::<f32>::from_shape_vec((1, 2, 4), vec![
             0.,-1., 0., 1.,
            -1., 0., 2., 0.,
        ]).unwrap();

        assert_eq!(error_signal, target_signal);
    }
}