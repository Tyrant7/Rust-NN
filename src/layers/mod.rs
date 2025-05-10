pub trait Layer: std::fmt::Debug
{
    type Input;
    type Output;

    fn forward(&mut self, input: &Self::Input, train: bool) -> Self::Output;
    fn backward(&mut self, input: &Self::Output, forward_input: &Self::Input) -> Self::Input;

    // Not all layers have learnable parameters
    fn get_learnable_parameters(&mut self) -> Vec<&mut ParameterGroup> { Vec::new() }
}

#[derive(Debug)]
pub struct ParameterGroup
{
    pub values: ArrayD<f32>,
    pub gradients: ArrayD<f32>,
}

impl ParameterGroup {
    pub fn new(initial_values: ArrayD<f32>) -> Self {
        let gradients = ArrayD::zeros(initial_values.raw_dim());
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
use ndarray::ArrayD;
pub use relu::ReLU;
pub mod sigmoid;
pub use sigmoid::Sigmoid;

#[cfg(test)]
mod tests;