use ndarray::{Array2, Array3, ArrayBase, Dimension, Ix1, Ix2, Ix3, OwnedRepr};

pub trait Layer: std::fmt::Debug
{
    fn forward(&mut self, input: &Tensor, train: bool) -> Tensor;
    fn backward(&mut self, input: &Tensor, forward_input: &Tensor) -> Tensor;

    // Not all layers have learnable parameters
    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> { Vec::new() }
}

#[derive(Debug)]
pub enum Tensor {
    T2D(Array2<f32>),
    T3D(Array3<f32>),
}

impl Tensor {
    fn into_array2d(&self) -> &Array2<f32> {
        match self {
            Self::T2D(data) => data,
            _ => panic!("Shape error: expected T2D but got {:?}", self)
        }
    }

    fn into_array3d(&self) -> &Array3<f32> {
        match self {
            Self::T3D(data) => data,
            _ => panic!("Shape error: expected T3D but got {:?}", self)
        }
    }
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

pub enum LearnableParameter<'a> {
    Param1D(&'a mut ParameterGroup<Ix1>),
    Param2D(&'a mut ParameterGroup<Ix2>),
    Param3D(&'a mut ParameterGroup<Ix3>),
}

macro_rules! match_param {
    ($self:expr, |$p:ident| $body:expr) => {
        match $self {
            LearnableParameter::Param1D($p) =>$body,
            LearnableParameter::Param2D($p) => $body,
            LearnableParameter::Param3D($p) => $body,
        }
    };
}

impl<'a> LearnableParameter<'a> {
    pub fn clone_shape<D: Dimension>(&self) -> ArrayBase<OwnedRepr<f32>, D> {
        match_param!(self, |p| ArrayBase::zeros(p.values.raw_dim()).into_dimensionality::<D>().unwrap())
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