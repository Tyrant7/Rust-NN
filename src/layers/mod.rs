pub trait Layer: std::fmt::Debug
{
    type Input;
    type Output;

    fn forward(&mut self, input: &Self::Input, train: bool) -> Self::Output;
    fn backward(&mut self, input: &Self::Output, forward_input: &Self::Input) -> Self::Input;

    // Not all layers have learnable parameters
    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> { Vec::new() }
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
}

pub mod linear;
pub use linear::Linear;

pub mod dropout;
pub use dropout::Dropout;

pub mod convolutional;
pub use convolutional::Convolutional1D;

pub mod relu;
use ndarray::{Array, ArrayBase, ArrayD, ArrayViewMutD, AsArray, Dimension, OwnedRepr};
pub use relu::ReLU;
pub mod sigmoid;
pub use sigmoid::Sigmoid;

#[cfg(test)]
mod tests;