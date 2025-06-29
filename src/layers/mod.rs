/// Trait for raw layers that directly transform data (e.g., activations, linear, conv, etc.). 
/// 
/// Implement this trait to define a layer type that participates in both forward and backward passes of a network. 
/// Raw layers operate on actual data arrays and may contain learnable parameters. 
/// 
/// Examples: [`ReLU`], [`Linear`], [`Convolutional2D`], [`Dropout`]. 
/// 
/// All input and output arrays are expected to include a leading batch dimension. 
/// 
/// # Required Associated Types:
/// - [`Self::Input`]: Dimension of the input data (must implement `ndarray::Dimension`). 
/// - [`Self::Output`]: Dimension of the output data (must implement `ndarray::Dimension`). 
/// 
/// # Required Methods: 
/// - [`Self::forward`]: Computes the layer's forward output given the input. 
/// - [`Self::backward`]: Computes the layer's backward gradient and optionally accumulates parameter gradients.  
/// 
/// # Optional Methods:
/// - [`Self::get_learnable_parameters`]: Returns the layer's parameters (if any). Should have a consistent order. 
pub trait RawLayer: std::fmt::Debug {
    /// Dimension of the forward input to this layer (must include batch dimension). 
    type Input: Dimension;

    /// Dimension of the output produced by this layer. 
    type Output: Dimension;

    /// Performs a forward pass through the layer. 
    /// 
    /// # Arguments
    /// - `input`: Input tensor with shape matching [`Self::Input`]. Must include batch as the first axis. 
    /// - `train`: Whether the forward pass is occuring during training (used by layers like [`Dropout`]). 
    ///  
    /// # Returns
    /// The output tensor with shape matching [`Self::Output`]. 
    fn forward(&mut self, input: &Array<f32, Self::Input>, train: bool) -> Array<f32, Self::Output>;

    /// Performs a backward pass through the layer. 
    /// 
    /// # Arguments
    /// - `error`: The error signal received from the next layer or loss function. Shape must match [`Self::Output`]. 
    /// - `forward_input`: The input originally passed to [`Self::forward`]. Typically provided by a [`Tracked`] wrapper. 
    /// 
    /// # Returns
    /// The gradient to be propagated to the previous layer. Shape must match [`Self::Input`]. 
    fn backward(&mut self, error: &Array<f32, Self::Output>, forward_input: &Array<f32, Self::Input>) -> Array<f32, Self::Input>;

    /// Optionally returns a vector of all learnable parameters in this layer.
    /// 
    /// This default implementation returns an empty list for layers with no learnable parameters.  
    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> { vec![] }
}


/// Trait for layers that handle other layers. 
/// 
/// Implement this trait to define a layer type that handles other layers within in a network. 
/// 
/// Examples: [`Chain`] and [`Tracked`]. 
/// 
/// All input and output arrays are expected to include a leading batch dimension. 
/// 
/// # Required Associated Types:
/// - [`Self::Input`]: Dimension of the input data (must implement `ndarray::Dimension`). 
/// - [`Self::Output`]: Dimension of the output data (must implement `ndarray::Dimension`). 
/// 
/// # Required Methods: 
/// - [`Self::forward`]: Computes the layer's forward output given the input. 
/// - [`Self::backward`]: Computes the layer's backward gradient and optionally accumulates parameter gradients. 
///   Note that this method does not depend on the forward input. If that is necessary for your implementation, 
///   you probably want to use the [`RawLayer`] trait. 
/// 
/// # Optional Methods:
/// - [`Self::get_learnable_parameters`]: Returns the layer's parameters (if any). Should have a consistent order. 
pub trait CompositeLayer: std::fmt::Debug {
    /// Dimension of the forward input to this layer (must include batch dimension). 
    type Input: Dimension;

    /// Dimension of the output produced by this layer. 
    type Output: Dimension;

    /// Performs a forward pass through the layer. 
    /// 
    /// # Arguments
    /// - `input`: Input tensor with shape matching [`Self::Input`]. Must include batch as the first axis. 
    /// - `train`: Whether the forward pass is occuring during training (used by layers like [`Dropout`]). 
    ///  
    /// # Returns
    /// The output tensor with shape matching [`Self::Output`]. 
    fn forward(&mut self, input: &Array<f32, Self::Input>, train: bool) -> Array<f32, Self::Output>;

    /// Performs a backward pass through the layer. 
    /// 
    /// # Arguments
    /// - `error`: The error signal received from the next layer or loss function. Shape must match [`Self::Output`]. 
    /// 
    /// # Returns
    /// The gradient to be propagated to the previous layer. Shape must match [`Self::Input`]. 
    fn backward(&mut self, error: &Array<f32, Self::Output>) -> Array<f32, Self::Input>;

    /// Optionally returns a vector of all learnable parameters in this layer.
    /// 
    /// This default implementation returns an empty list for layers with no learnable parameters.  
    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> { vec![] }
}

/// A view-based structure used to update layer's parameters during training. 
/// 
/// [`LearnableParameter`] wraps mutable views into a parameter's values and gradients,
/// allowing optimizers to modify them without knowing the underlying dimensionality. 
/// 
/// These are typically created by [`ParameterGroup::as_learnable_parameter`] and consumed by [`Optimizer`]s. 
#[derive(Debug)]
pub struct LearnableParameter<'a> {
    pub values: ArrayViewMutD<'a, f32>,
    pub gradients: ArrayViewMutD<'a, f32>,
}

impl<'a> LearnableParameter<'a> {
    /// Zeros the gradients of this parameter in place. 
    pub fn zero_grads(&mut self) {
        self.gradients.fill(0.)
    }
}

/// Stores a parameter tensor and its associated gradients. 
/// 
/// [`ParameterGroup`] manages both the values and the gradietns for a layer's learnable tensor,
/// and provides a method to expose these as a [`LearnableParameter`] for use by optimizers. 
/// 
/// This is typically used internally by layers with learnable weights. 
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParameterGroup<D>
where 
    D: Dimension
{
    pub values: Array<f32, D>,
    pub gradients: Array<f32, D>,
}

impl<D: Dimension> ParameterGroup<D>
{
    /// Creates a new [`ParameterGroup`] with the given initial values. 
    /// 
    /// Gradients are initialized to zero and will match the shape of `initial_values`. 
    pub fn new(initial_values: Array<f32, D>) -> Self {
        let gradients = Array::zeros(initial_values.raw_dim());
        ParameterGroup {
            values: initial_values,
            gradients
        }
    }

    /// Returns a [`LearnableParameter`] with mutable views into both values and gradients.  
    pub fn as_learnable_parameter(&mut self) -> LearnableParameter {
        LearnableParameter { 
            values: self.values.view_mut().into_dyn(), 
            gradients: self.gradients.view_mut().into_dyn() 
        }
    }
}

use std::vec;

use ndarray::{Array, ArrayViewMutD, Dimension, IntoDimension, OwnedRepr};

pub mod chain;
pub use chain::Chain;

pub mod tracked;
use serde::{Deserialize, Serialize};
pub use tracked::Tracked;

pub mod linear;
pub use linear::Linear;

pub mod dropout;
pub use dropout::Dropout;

pub mod convolutional1d;
pub use convolutional1d::Convolutional1D;

pub mod convolutional2d;
pub use convolutional2d::Convolutional2D;

pub mod pooling;
pub use pooling::maxpool1d::MaxPool1D;
pub use pooling::maxpool2d::MaxPool2D;

pub mod flatten;
pub use flatten::Flatten;

pub mod relu;
pub use relu::ReLU;
pub mod sigmoid;
pub use sigmoid::Sigmoid;

#[cfg(test)]
pub mod tests {
    use ndarray::{Array1, Dim, IxDyn, IxDynImpl};

    use super::RawLayer;

    pub fn test_activation_fn<T>(mut activation: T, input: Vec<f32>, expected_out: Vec<f32>, error: Vec<f32>, expected_err: Vec<f32>)
    where 
        T: RawLayer<Input = Dim<IxDynImpl>, Output = Dim<IxDynImpl>>
    {
        let epsilon = 1e-5;
        
        let input = Array1::<f32>::from_shape_vec(input.len(), input).unwrap().into_dyn();
        let output = activation.forward(&input, false);

        let target = Array1::<f32>::from_shape_vec(expected_out.len(), expected_out).unwrap().into_dyn();

        for (o, t) in output.iter().zip(target.iter()) {
            let diff = (*o - *t).abs();
            assert!(diff <= epsilon);
        }

        let error = Array1::<f32>::from_shape_vec(error.len(), error).unwrap().into_dyn();
        let error_signal = activation.backward(&error, &input);

        let target_signal = Array1::<f32>::from_shape_vec(expected_err.len(), expected_err).unwrap().into_dyn();

        for (e, t) in error_signal.iter().zip(target_signal.iter()) {
            let diff = (*e - *t).abs();
            assert!(diff <= epsilon);
        }
    }
}