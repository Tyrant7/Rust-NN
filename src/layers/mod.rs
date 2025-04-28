pub trait Layer: std::fmt::Debug
{
    fn forward(&mut self, input: &Tensor, train: bool) -> Tensor;
    fn backward(&mut self, input: &Tensor, forward_input: &Tensor) -> Tensor;

    // Not all layers have learnable parameters
    fn get_learnable_parameters(&mut self) -> Vec<ParameterGroup> { Vec::new() }
}

#[derive(Debug)]
pub struct ParameterGroup
{
    pub values: Tensor,
    pub gradients: Tensor,
}

impl ParameterGroup {
    pub fn new(initial_values: Tensor) -> Self {
        let gradients = initial_values.map(|_| 0.);
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
pub use relu::ReLU;
pub mod sigmoid;
pub use sigmoid::Sigmoid;

use crate::tensor::Tensor;

#[cfg(test)]
mod tests;