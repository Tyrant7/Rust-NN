use ndarray::{s, Array4, Axis, Ix4};

use crate::{conv_helpers::{crop_4d, pad_4d}, layers::{ParameterGroup, RawLayer}};

#[derive(Debug)]
pub struct MaxPool2D {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),

    max_indices: Option<Array4<(usize, usize)>>
}

impl MaxPool2D {
    /// Pairs follow shape: (height, width)
    fn new_full(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        assert!(kernel_size.0 > 0 && kernel_size.1 > 0, "Kernel size must be positive on all dimensions");
        assert!(stride.0 > 0 && stride.1 > 0, "Stride must be positive on all dimensions");
        MaxPool2D { 
            kernel_size, 
            stride, 
            padding, 
            max_indices: None,
        }
    }

    fn new(kernel_size: (usize, usize)) -> Self {
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
        error.clone()
        // let (batch_size, in_features, input_width) = forward_input.dim();
        // let (_, _, error_width) = error.dim();

        // let signal_width = (error_width - 1) * self.stride + self.kernel_width;
        // let mut error_signal = Array3::zeros((batch_size, in_features, signal_width));
        // let max_indices = self.max_indices
        //     .as_ref()
        //     .expect("No indices stored during forward pass or forward pass never called!");
        // for b in 0..batch_size {
        //     for in_f in 0..in_features {
        //         for i in 0..error_width {
        //             // Indices were saved with padding, so may be incorrect
        //             let idx = max_indices[[b, in_f, i]];
        //             error_signal[[b, in_f, idx]] += error[[b, in_f, i]];
        //         }
        //     }
        // }
        // crop_3d(&error_signal.view(), (0, 0, signal_width - input_width))
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
        let mut maxpool = MaxPool2D::new_full((2, 2), (1, 1), (1, 1));
        
        let input = Array4::<f32>::from_shape_vec((1, 1, 2, 3), vec![
            1., 2., -1.,
            4., 3., 5.,
        ]).unwrap();
        let output = maxpool.forward(&input, false);

        let target = Array4::<f32>::from_shape_vec((1, 1, 3, 4), vec![
            // Feature 1
            1., 2., 2., 0., 
            4., 4., 5., 5.,
            4., 4., 5., 5.,
        ]).unwrap();

        assert_eq!(output, target);
    }

    // #[test]
    // fn backward() {
    //     let mut maxpool = MaxPool1D::new(2);
        
    //     let input = Array3::<f32>::from_shape_vec((1, 2, 4), vec![
    //         // Feature 1
    //         1., 2., 3., 4.,
    //         // Feature 2
    //         5., 4., 3., 2.,
    //     ]).unwrap();
    //     maxpool.forward(&input, false);

    //     let error = Array3::<f32>::from_shape_vec((1, 2, 2), vec![
    //         -1., 1.,
    //         -1., 2.,
    //     ]).unwrap();
    //     let error_signal = maxpool.backward(&error, &input);

    //     let target_signal = Array3::<f32>::from_shape_vec((1, 2, 4), vec![
    //          0.,-1., 0., 1.,
    //         -1., 0., 2., 0.,
    //     ]).unwrap();

    //     assert_eq!(error_signal, target_signal);
    // }

    // #[test]
    // fn backward_stride_and_padding() {
    //     let mut maxpool = MaxPool1D::new_full(2, 2, 1);
        
    //     let input = Array3::<f32>::from_shape_vec((1, 1, 4), vec![
    //         // Feature 1
    //         1., 2., 3., 4.,
    //     ]).unwrap();
    //     maxpool.forward(&input, false);

    //     let error = Array3::<f32>::from_shape_vec((1, 1, 3), vec![
    //         -1., 2., 1.,
    //     ]).unwrap();
    //     let error_signal = maxpool.backward(&error, &input);

    //     let target_signal = Array3::<f32>::from_shape_vec((1, 1, 4), vec![
    //         -1., 0., 2., 1.
    //     ]).unwrap();

    //     assert_eq!(error_signal, target_signal);
    // }
}