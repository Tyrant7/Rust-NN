use std::sync::Mutex;

use ndarray::{
    concatenate, s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, Axis, Ix1, Ix2, Ix3,
    Ix4,
};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::helpers::{
    conv_helpers::{col2im, convolve2d, crop_4d, im2col, pad_2d, pad_4d},
    initialize_weights::{kaiming_normal, SeedMode},
};

use super::{LearnableParameter, ParameterGroup, RawLayer};

/// A convolutional layer that handles 2D spatial data.
///
/// Convolutional layers are widely used in machine learning tasks involving spatial or temporal data, such
/// as images, audio, and text. They are well-suited for extracting local features by applying learnable kernels
/// over input features with spatial relationships.
///
/// The shape of the output is given as follows:
///
/// ```text
/// (batch_size, out_features, output_height, output_width)
/// where
/// output_height = floor((height - kernel_height + 2 * padding.0) / stride.0) + 1;
/// output_width =  floor((width  - kernel_width  + 2 * padding.1) / stride.1) + 1;
/// ```
///
/// Convolutions occur between each feature in the input, and the kernels of this convolutional layer, and then have a bias
/// added to each output as follows:
///
/// ```text
/// for each batch:
///     for each output feature:
///         for each feature in the input:
///             output[batch, output_feature, .., ..] =
///                 convolve(input[batch, input_feature, .., ..], kernels[output_feature, input_feature, .., ..]) + bias[output_feature]
/// ```
///
/// - Input data shape: `(batch_size, features, height, width)`
/// - Kernels shape: `(out_features, in_features, kernel_height, kernel_width)`
/// - Bias shape: `(out_features)`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Convolutional2D {
    kernels: ParameterGroup<Ix4>,
    bias: Option<ParameterGroup<Ix1>>,

    stride: (usize, usize),
    padding: (usize, usize),

    #[serde(skip)]
    cache: Option<(Array2<f32>, Array2<f32>)>,
}

impl Convolutional2D {
    /// Initializes a new [`Convolutional2D`] layer with random kernels of the given shape and zero bias (if enabled) using the Kaiming Normal initialization.  
    ///
    /// This is the standard way to initialize a convolutional layer for training.
    ///
    /// # Arguments
    /// - `in_features`: Number of input features.
    /// - `out_features`: Number of output features.
    /// - `kernel_size`: The size of each kernel.  
    /// - `use_bias`: Whether or not bias should be added to each output.
    /// - `stride`: The strides to use during the convolutions between the kernels and input features.
    /// - `padding`: The paddings to add to either sides of the input before convolutions are performed.
    ///
    /// All pairs during initialization are expected in `(height, width)` format.
    ///
    /// # Panics
    /// - If any of `in_features`, `out_features`, `kernel_size.0` or `kernel_size.1` are zero.
    /// - If `stride` is zero.
    pub fn new_from_rand(
        in_features: usize,
        out_features: usize,
        kernel_size: (usize, usize),
        use_bias: bool,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        assert!(
            in_features > 0,
            "Invalid input feature count: {in_features}"
        );
        assert!(
            out_features > 0,
            "Invalid output feature count: {out_features}"
        );
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0,
            "Invalid kernel size: {kernel_size:?}"
        );

        let kernels = kaiming_normal(
            (out_features, in_features, kernel_size.0, kernel_size.1),
            1,
            SeedMode::Random,
        );
        let bias = match use_bias {
            true => Some(Array1::zeros(out_features)),
            false => None,
        };
        Convolutional2D::new_from_kernels(kernels, bias, stride, padding)
    }

    /// Initializes a new [`Convolutional2D`] layer with given kernels and bias (if enabled).  
    ///
    /// # Parameters
    /// - `kernels`: The kernels to use for the convolution. Should have the shape `(out_features, in_features, kernel_height, kernel_width)`.
    /// - `bias`: The bias to add to each output feature. Should have the shape `(out_features)`.
    /// - `stride`: The stride to use during the convolutions between the kernels and input features.
    /// - `padding`: The paddings to add to either sides of the input before convolutions are performed.
    ///
    /// All pairs during initialization are expected in `(height, width)` format.
    ///
    /// # Panics
    /// - If `bias` and `kernels` do not share the same number of input features when bias exists (first dimension length).
    /// - If any dimension of `stride` is zero.
    pub fn new_from_kernels(
        kernels: Array4<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        assert!(
            stride.0 > 0 && stride.1 > 0,
            "Invalid stride given: {stride:?}"
        );
        if let Some(b) = &bias {
            assert!(
                b.dim() == kernels.dim().0,
                "Shape mismatch between kernels and bias: {:?}, {:?}",
                kernels.dim(),
                b.dim()
            );
        }

        let kernels = ParameterGroup::new(kernels);
        let bias = bias.map(ParameterGroup::new);
        Convolutional2D {
            kernels,
            bias,
            stride,
            padding,
            cache: None,
        }
    }
}

impl RawLayer for Convolutional2D {
    type Input = Ix4;
    type Output = Ix4;

    // Expected input shape: (batch_size, features, height, width)
    fn forward(&mut self, input: &Array4<f32>, train: bool) -> Array4<f32> {
        let (batch_size, in_features, height, width) = input.dim();

        let (out_features, _, kernel_height, kernel_width) = self.kernels.values.dim();
        let output_width = ((width - kernel_width + (2 * self.padding.1)) / self.stride.1) + 1;
        let output_height = ((height - kernel_height + (2 * self.padding.0)) / self.stride.0) + 1;

        // Transform the kernels into a single matrix of dimensions (out_features, k)
        // to prepare for a matrix multiplication
        let k = in_features * kernel_height * kernel_width;
        let kernel_matrix = self
            .kernels
            .values
            .clone()
            .into_shape_with_order((out_features, k))
            .expect("Kernel reshape failed");

        // We'll use im2col to do this for our input
        let input_matrix = im2col(
            input,
            (kernel_height, kernel_width),
            self.stride,
            self.padding,
        );

        // Matrix multiply to get output
        let mut output = kernel_matrix
            .dot(&input_matrix)
            .into_shape_clone((batch_size, out_features, output_height, output_width))
            .expect("Error when reshaping output");

        // Update the cache for our backward pass
        if train {
            self.cache = Some((kernel_matrix, input_matrix));
        }

        // Apply bias to the second dimension (features)
        if let Some(b) = &self.bias {
            output += &b
                .values
                .broadcast((batch_size, out_features, output_height, output_width))
                .unwrap();
        }
        output
    }

    fn backward(&mut self, delta: &Array4<f32>, forward_input: &Array4<f32>) -> Array4<f32> {
        let (batch_size, in_features, input_height, input_width) = forward_input.dim();
        let (out_features, _, kernel_height, kernel_width) = self.kernels.values.dim();

        // Get our kernel and input matrices constructed from the forward pass
        let (kernel_matrix, input_matrix) = self
            .cache
            .clone()
            .expect("Backward called before forward or outside of train mode");

        // Calculate bias grads
        if let Some(b) = &mut self.bias {
            b.gradients += &delta
                .sum_axis(Axis(0)) // Sum over batch
                .sum_axis(Axis(1)) // Channel is now axis 0, so Axis 1 is height
                .sum_axis(Axis(1)); // Axis 1 is now width
        }

        // Reshape delta
        let dout = delta
            .to_shape((batch_size * in_features, input_height * input_width))
            .expect("Error reshaping delta");
        let dout = dout
            .axis_chunks_iter(Axis(0), in_features)
            .collect::<Vec<_>>();
        let dout = concatenate(Axis(1), &dout).expect("Error concatenated delta into matrix");

        // Matrix multiplication to get kernel gradients
        self.kernels.gradients.scaled_add(
            1.,
            &dout
                .dot(&input_matrix.t())
                .into_shape_with_order((out_features, in_features, input_height, input_width))
                .expect("Failed to reshape kernel gradients"),
        );

        // Matrix multiplication to get new error signal matrix
        let error_matrix = kernel_matrix.t().dot(&dout);

        // Reshape back to image with col2im
        col2im(
            &error_matrix,
            forward_input.dim(),
            (kernel_height, kernel_width),
            self.stride,
            self.padding,
        )
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
#[rustfmt::skip]
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
        conv.forward(&input, true);

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
        conv.forward(&input, true);

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

        let target_grads = Array4::<f32>::from_shape_vec((1, 1, 2, 2), vec![
           15., 13., 
           11., 9., 
        ]).unwrap();
        assert_eq!(conv.kernels.gradients, target_grads);
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
        conv.forward(&input, true);

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
