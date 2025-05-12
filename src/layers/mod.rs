// Raw layers are layers that actually deal with data
// All activation functions, fully connected layers, or convolutions, etc. are raw layers
pub trait RawLayer: std::fmt::Debug
{
    type Input: Dimension;
    type Output: Dimension;

    fn forward(&mut self, input: &ArrayBase<OwnedRepr<f32>, Self::Input>, train: bool) -> ArrayBase<OwnedRepr<f32>, Self::Output>;
    fn backward(&mut self, 
        error: &ArrayBase<OwnedRepr<f32>, Self::Output>, 
        forward_input: &ArrayBase<OwnedRepr<f32>, Self::Input>
    ) -> ArrayBase<OwnedRepr<f32>, Self::Input>;

    // Not all layers have learnable parameters
    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> { vec![] }

    // TODO: "inspect()" method to print model layers in a prettier way
}

// Composite layers only handle piping data to raw layers, and don't actually deal with the data itself
// hence why these layers does not care about the forward input in backpropagation -> they will always
// delegate error calculations
pub trait CompositeLayer: std::fmt::Debug {
    type Input: Dimension;
    type Output: Dimension;

    fn forward(&mut self, input: &ArrayBase<OwnedRepr<f32>, Self::Input>, train: bool) -> ArrayBase<OwnedRepr<f32>, Self::Output>;
    fn backward(&mut self, error: &ArrayBase<OwnedRepr<f32>, Self::Output>) -> ArrayBase<OwnedRepr<f32>, Self::Input>;

    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> { vec![] }

    // TODO: "inspect()" method to print model layers in a prettier way
}

pub struct LearnableParameter<'a> {
    pub values: ArrayViewMutD<'a, f32>,
    pub gradients: ArrayViewMutD<'a, f32>,
}

impl<'a> LearnableParameter<'a> {
    pub fn zero_grads(&mut self) {
        self.gradients.fill(0.)
    }
}

#[derive(Debug)]
struct ParameterGroup<T>
where 
    T: Dimension
{
    pub values: ArrayBase<OwnedRepr<f32>, T>,
    pub gradients: ArrayBase<OwnedRepr<f32>, T>,
}

impl<T: Dimension> ParameterGroup<T>
{
    pub fn new(initial_values: ArrayBase<OwnedRepr<f32>, T>) -> Self {
        let gradients = Array::zeros(initial_values.raw_dim());
        ParameterGroup {
            values: initial_values,
            gradients
        }
    }

    pub fn as_learnable_parameter(&mut self) -> LearnableParameter {
        LearnableParameter { 
            values: self.values.view_mut().into_dyn(), 
            gradients: self.gradients.view_mut().into_dyn() 
        }
    }
}

use std::vec;

use ndarray::{Array, ArrayBase, ArrayViewMutD, Dimension, IntoDimension, OwnedRepr};

pub mod chain;
pub use chain::Chain;

pub mod tracked;
pub use tracked::Tracked;

pub mod linear;
pub use linear::Linear;

pub mod dropout;
pub use dropout::Dropout;

pub mod convolutional;
pub use convolutional::Convolutional1D;

pub mod relu;
pub use relu::ReLU;
pub mod sigmoid;
pub use sigmoid::Sigmoid;

#[cfg(test)]
mod tests;