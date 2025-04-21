use ndarray::Array2;

// TODO: We need some way to support layers that take different input types
// -> Probably use an enum

pub trait Layer: std::fmt::Debug {
    fn forward(&mut self, input: &Array2<f32>, train: bool) -> Array2<f32>;
    fn backward(&mut self, input: &Array2<f32>, forward_input: &Array2<f32>) -> Array2<f32>;

    // Not all layers have learnable parameters
    fn get_learnable_parameters(&mut self) -> Vec<Parameter> { Vec::new() }
}

pub struct Parameter<'a> {
    pub value: &'a mut Array2<f32>,
    pub gradient: &'a mut Array2<f32>,
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

#[cfg(test)]
mod tests;