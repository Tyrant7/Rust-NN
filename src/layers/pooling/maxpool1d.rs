use ndarray::{s, Array3, Axis, Ix3};
use serde::{Deserialize, Serialize};

use crate::{helpers::conv_helpers::{crop_3d, pad_3d}, layers::{ParameterGroup, RawLayer}};

/// A layer that performs 1D max pooling over spatial or temporal data. 
/// 
/// Pooling layers are widely used in machine learning tasks involving spatial or temporal data, such
/// as images, audio and text. They are well-suited for reducing the size and complexity of spatial dimensions
/// while retaining important information about the data's spatial relationships. 
/// 
/// Expects input in the shape: `(batch_size, features, width)`, and 
/// the shape of the output is given as follows:
/// 
/// ```text
/// (batch_size, out_features, output_width)
/// where
/// output_width = floor((width - kernel_width + (2 * padding)) / stride) + 1;
/// ```
/// 
/// Max pooling occurs over each feature in the input. It works by sliding a kernel over the input by `stride` steps,
/// and reducing each set of overlapping data to the highest value currently overlapping the kernel. 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxPool1D {
    kernel_width: usize,
    stride: usize,
    padding: usize,

    /// Stores the indices of maximum elements selected during the forward pass
    /// for each window position. Used to route gradients in `backward`. 
    #[serde(skip)]
    max_indices: Option<Array3<usize>>
}

impl MaxPool1D {
    /// Initializes a new [`MaxPool1D`] layer with the given parameters. 
    /// 
    /// # Arguments
    /// - `kernel_width`: The width of the sliding window during pooling. 
    /// - `stride`: The step size of the sliding window during pooling. 
    /// - `padding`: The padding to add to either side of the input before pooling is performed. 
    /// 
    /// # Panics
    /// - If `kernel_width` is zero. 
    /// - If `stride` is zero. 
    fn new_full(
        kernel_width: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        assert!(kernel_width > 0, "Invalid kernel width: {kernel_width}");
        assert!(stride > 0, "Invalid stride given: {stride}");
        MaxPool1D { 
            kernel_width, 
            stride, 
            padding, 
            max_indices: None,
        }
    }

    /// Initializes a new [`MaxPool1D`] layer with the given `kernel_width`. This uses a stride equal to `kernel_width`, 
    /// and no padding. 
    /// 
    /// For more control over stride and padding, use [`MaxPool1D::new_full`]. 
    /// 
    /// # Arguments
    /// - `kernel_width`: The width of the sliding window during pooling. 
    /// 
    /// # Panics
    /// - If `kernel_width` is zero. 
    /// - If `stride` is zero. 
    fn new(kernel_width: usize) -> Self {
        Self::new_full(kernel_width, kernel_width, 0)
    }
}

impl RawLayer for MaxPool1D {
    type Input = Ix3;
    type Output = Ix3;

    // Input shape: (batch_size, features, width)
    fn forward(&mut self, input: &Array3<f32>, _train: bool) -> Array3<f32> {
        let (batch_size, in_features, width) = input.dim();
        let output_width = ((width - self.kernel_width + (2 * self.padding)) / self.stride) + 1;
        let mut output = Array3::zeros((batch_size, in_features, output_width));

        // (batch_size, in_features, width)
        let input = pad_3d(&input.view(), (0, 0, self.padding));

        // We'll track which indices we selected for pooling to propagate error only through those
        // indices during the backward pass
        let mut max_indices = Array3::<usize>::zeros(output.dim());

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
        let (batch_size, in_features, input_width) = forward_input.dim();
        let (_, _, error_width) = error.dim();

        let signal_width = (error_width - 1) * self.stride + self.kernel_width;
        let mut error_signal = Array3::zeros((batch_size, in_features, signal_width));
        let max_indices = self.max_indices
            .as_ref()
            .expect("No indices stored during forward pass or forward pass never called!");
        let (_, _, indices_width) = max_indices.dim();
        for b in 0..batch_size {
            for in_f in 0..in_features {
                for i in 0..indices_width {
                    // Indices were saved with padding, so may be incorrect
                    let idx = max_indices[[b, in_f, i]];
                    error_signal[[b, in_f, idx]] += error[[b, in_f, i]];
                }
            }
        }
        crop_3d(&error_signal.view(), (0, 0, signal_width - input_width))
    }
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
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
    fn forward_stride_and_padding() {
        let mut maxpool = MaxPool1D::new_full(2, 1, 1);
        
        let input = Array3::<f32>::from_shape_vec((1, 2, 2), vec![
            // Feature 1
            1., 2.,
            // Feature 2
            4., 3.,
        ]).unwrap();
        let output = maxpool.forward(&input, false);

        let target = Array3::<f32>::from_shape_vec((1, 2, 3), vec![
            // Feature 1
            1., 2., 2.,
            // Feature 2
            4., 4., 3.,
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

    #[test]
    fn backward_stride_and_padding() {
        let mut maxpool = MaxPool1D::new_full(2, 2, 1);
        
        let input = Array3::<f32>::from_shape_vec((1, 1, 4), vec![
            // Feature 1
            1., 2., 3., 4.,
        ]).unwrap();
        maxpool.forward(&input, false);

        let error = Array3::<f32>::from_shape_vec((1, 1, 3), vec![
            -1., 2., 1.,
        ]).unwrap();
        let error_signal = maxpool.backward(&error, &input);

        let target_signal = Array3::<f32>::from_shape_vec((1, 1, 4), vec![
            -1., 0., 2., 1.
        ]).unwrap();

        assert_eq!(error_signal, target_signal);
    }
}