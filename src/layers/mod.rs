use ndarray::{Array2, ArrayBase, Dimension, Ix0, OwnedRepr, ShapeBuilder};

// TODO: We need some way to support layers that take different input types
// -> Probably use an enum

pub trait Layer: std::fmt::Debug {
    fn forward(&mut self, input: &Array2<f32>, train: bool) -> Array2<f32>;
    fn backward(&mut self, input: &Array2<f32>, forward_input: &Array2<f32>) -> Array2<f32>;

    // Not all layers have learnable parameters
    fn get_learnable_parameters(&mut self) -> Vec<ParameterGroup<Ix0>> { Vec::new() }
}

#[derive(Debug)]
pub struct ParameterGroup<D>
where
    D: Dimension,
{
    pub values: ArrayBase<OwnedRepr<f32>, D>,
    pub gradients: ArrayBase<OwnedRepr<f32>, D>,
}

impl<D: Dimension> ParameterGroup<D> {
    pub fn new(initial_values: ArrayBase<OwnedRepr<f32>, D>) -> Self {
        let raw_dim = initial_values.raw_dim();
        ParameterGroup { 
            values: initial_values, 
            gradients: ArrayBase::from_elem(raw_dim, 0.), 
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

#[cfg(test)]
mod tests;