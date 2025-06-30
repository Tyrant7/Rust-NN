use ndarray::{s, Array4, Axis, Ix4};
use serde::{Deserialize, Serialize};

use crate::{helpers::conv_helpers::{crop_4d, pad_4d}, layers::{ParameterGroup, RawLayer}};

/// A layer that performs 2D max pooling over spatial or temporal data. 
/// 
/// Pooling layers are widely used in machine learning tasks involving spatial or temporal data, such
/// as images, audio and text. They are well-suited for reducing the size and complexity of spatial dimensions
/// while retaining important information about the data's spatial relationships. 
/// 
/// Expects input in the shape: `(batch_size, features, height, width)`, and 
/// the shape of the output is given as follows:
/// 
/// ```text
/// (batch_size, out_features, output_height, output_width)
/// where
/// output_height = floor((height - kernel_height + (2 * padding.0)) / stride.0) + 1;
/// output_width  = floor((width  - kernel_width  + (2 * padding.1)) / stride.1) + 1;
/// ```
/// 
/// Max pooling occurs over each feature in the input. It works by sliding a kernel over the input by `stride` steps,
/// and reducing each set of overlapping data to the highest value currently overlapping the kernel. 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxPool2D {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),

    /// Stores the indices of maximum elements selected during the forward pass
    /// for each window position. Used to route gradients in `backward`. 
    #[serde(skip)]
    max_indices: Option<Array4<(usize, usize)>>
}

impl MaxPool2D {
    /// Initializes a new [`MaxPool2D`] layer with the given parameters. 
    /// 
    /// # Arguments
    /// - `kernel_size`: The size of the sliding window during pooling. 
    /// - `stride`: The step size of the sliding window during pooling. 
    /// - `padding`: The paddings to add to either sides of the input before pooling is performed. 
    /// 
    /// All pairs during initialization are expected in `(height, width)` format.
    /// 
    /// # Panics
    /// - If any of `kernel_size.0`, `kernel_size.1`, `stride.0` or `stride.1` is zero. 
    pub fn new_full(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        assert!(kernel_size.0 > 0 && kernel_size.1 > 0, "Invalid kernel size: {kernel_size:?}");
        assert!(stride.0 > 0 && stride.1 > 0, "Invalid stride given: {stride:?}");
        MaxPool2D { 
            kernel_size, 
            stride, 
            padding, 
            max_indices: None,
        }
    }

    /// Initializes a new [`MaxPool2D`] layer with the given `kernel_size`. This uses a stride equal to `kernel_size`,
    /// and no padding.  
    /// 
    /// For more control over stride and padding, use [`MaxPool2D::new_full`]. 
    /// 
    /// # Arguments
    /// - `kernel_size`: The size of the sliding window during pooling. Given in `(height, width)` format. 
    /// 
    /// # Panics
    /// - If `kernel_size.0` or `kernel_size.1` are zero.
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self::new_full(kernel_size, kernel_size, (0, 0))
    }
}

impl RawLayer for MaxPool2D {
    type Input = Ix4;
    type Output = Ix4;
    
        /// input shape: (batch_size, features, height, width)
    fn forward(&mut self, input: &Array4<f32>, _train: bool) -> Array4<f32> {
        let (batch_size, in_features, height, width) = input.dim();
        let output_width = ((width - self.kernel_size.1 + (2 * self.padding.1)) / self.stride.1) + 1;
        let output_height = ((height - self.kernel_size.0 + (2 * self.padding.0)) / self.stride.0) + 1;
        let mut output = Array4::zeros((batch_size, in_features, output_height, output_width));

        // (batch_size, in_features, height, width)
        let input = pad_4d(&input.view(), (0, 0, self.padding.0, self.padding.1));

        // We'll track which indices we selected for pooling to propagate error only through those
        // indices during the backward pass
        let mut max_indices = Array4::<(usize, usize)>::from_elem(output.dim(), (0, 0));

        for b in 0..batch_size {
            for in_f in 0..in_features {
                let input_slice = input.slice(s![b, in_f, .., ..]);
                let windows = input_slice.windows_with_stride(self.kernel_size, self.stride);
                for (i, window) in windows.into_iter().enumerate() {
                    let (max_idx, max_val) = window.iter()
                        .enumerate()
                        .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();
                    let (x, y) = (i % output_width, i / output_width); 
                    output[[b, in_f, y, x]] = *max_val;
                    let (max_x, max_y) = (max_idx % self.kernel_size.1, max_idx / self.kernel_size.1);
                    max_indices[[b, in_f, y, x]] = (y * self.stride.0 + max_y, x * self.stride.1 + max_x);
                }
            }
        }
        self.max_indices = Some(max_indices);

        output
    }

    fn backward(&mut self, error: &Array4<f32>, forward_input: &Array4<f32>) -> Array4<f32> {
        let (batch_size, in_features, input_height, input_width) = forward_input.dim();
        let (_, _, error_height, error_width) = error.dim();

        let signal_width = (error_width - 1) * self.stride.1 + self.kernel_size.1;
        let signal_height = (error_height - 1) * self.stride.0 + self.kernel_size.0;
        let mut error_signal = Array4::zeros((batch_size, in_features, signal_height, signal_width));
        let max_indices = self.max_indices
            .as_ref()
            .expect("No indices stored during forward pass or forward pass never called!");
        let (_, _, indices_height, indices_width) = max_indices.dim();
        for b in 0..batch_size {
            for in_f in 0..in_features {
                for y in 0..indices_height {
                    for x in 0..indices_width {
                        // Indices were saved with padding, so may be incorrect
                        let (idx_y, idx_x) = max_indices[[b, in_f, y, x]];
                        error_signal[[b, in_f, idx_y, idx_x]] += error[[b, in_f, y, x]];
                    }
                }
            }
        }
        crop_4d(&error_signal.view(), (0, 0, signal_height - input_height, signal_width - input_width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let mut maxpool = MaxPool2D::new((2, 2));
        
        let input = Array4::<f32>::from_shape_vec((1, 2, 2, 4), vec![
            // Feature 1
            1., 2., 4., 0.,
            5., 4., 3., 0.,
            // Feature 2
            1., 2., 4., 0.,
            5., 4., 3., 0.,
        ]).unwrap();
        let output = maxpool.forward(&input, false);

        let target = Array4::<f32>::from_shape_vec((1, 2, 1, 2), vec![
            // Feature 1
            5., 4.,
            // Feature 2
            5., 4.,
        ]).unwrap();

        assert_eq!(output, target);
    }

    #[test]
    fn forward_stride_and_padding() {
        let mut maxpool = MaxPool2D::new_full((2, 2), (2, 1), (1, 1));
        
        let input = Array4::<f32>::from_shape_vec((1, 1, 2, 3), vec![
            1., 2., -1.,
            4., 3., 5.,
        ]).unwrap();
        let output = maxpool.forward(&input, false);

        let target = Array4::<f32>::from_shape_vec((1, 1, 2, 4), vec![
            // Feature 1
            1., 2., 2., 0., 
            4., 4., 5., 5.,
        ]).unwrap();

        assert_eq!(output, target);
    }

    #[test]
    fn backward() {
        let mut maxpool = MaxPool2D::new((2, 2));
        
        let input = Array4::<f32>::from_shape_vec((1, 2, 2, 4), vec![
            // Feature 1
            1., 2., 4.,-1.,
            5., 4., 3., 0.,
            // Feature 2
            1., 2., 4., 0.,
            1., 3., 4., 5.,
        ]).unwrap();
        maxpool.forward(&input, false);

        let error = Array4::<f32>::from_shape_vec((1, 2, 1, 2), vec![
            // Feature 1
            -1., 1.,
            // Feature 2
            -1., 2.,
        ]).unwrap();
        let error_signal = maxpool.backward(&error, &input);

        let target_signal = Array4::<f32>::from_shape_vec((1, 2, 2, 4), vec![
            // Feature 1
             0., 0., 1., 0.,
            -1., 0., 0., 0.,
            // Feature 2
             0., 0., 0., 0.,
             0.,-1., 0., 2.,
        ]).unwrap();

        assert_eq!(error_signal, target_signal);
    }

    #[test]
    fn backward_stride_and_padding() {
        let mut maxpool = MaxPool2D::new_full((2, 2), (1, 2), (0, 1));
        
        let input = Array4::<f32>::from_shape_vec((1, 1, 2, 4), vec![
            1., 2., 3.,-8.,
            0., 3., 6.,-9.,
        ]).unwrap();
        maxpool.forward(&input, false);

        let error = Array4::<f32>::from_shape_vec((1, 1, 1, 3), vec![
            -1., 2., 1.,
        ]).unwrap();
        let error_signal = maxpool.backward(&error, &input);

        let target_signal = Array4::<f32>::from_shape_vec((1, 1, 2, 4), vec![
            -1., 0., 0., 0.,
             0., 0., 2., 0.,
        ]).unwrap();

        assert_eq!(error_signal, target_signal);
    }
}