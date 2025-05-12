use std::fmt::Debug;

use ndarray::{Array, Array2, ArrayD, Dimension, IntoDimension, IxDyn};

use crate::layers::{CompositeLayer, RawLayer, LearnableParameter};

#[derive(Debug)]
pub struct Chain<L1, L2>
where 
    L1: CompositeLayer<Output: Dimension>,
    L2: CompositeLayer<Input: Dimension>,
{
    inner: L1,
    next: L2,
}

impl<L1, L2> Chain<L1, L2> 
where 
    L1: CompositeLayer<Output: Dimension>,
    L2: CompositeLayer<Input: Dimension>,
{
    pub fn new(inner: L1, next: L2) -> Self {
        Self { 
            inner, 
            next,
        }
    }
}

impl<L1, L2> CompositeLayer for Chain<L1, L2>
where 
    L1: CompositeLayer<Output: Dimension>,
    L2: CompositeLayer<Input: Dimension>,
{
    type Input = L1::Input;
    type Output = L2::Output;

    fn forward(&mut self, input: &Array<f32, Self::Input>, train: bool) -> Array<f32, Self::Output> {
        let out1 = self.inner.forward(input, train);
        let resized_out1 = out1.into_dimensionality().expect("Incompatible array size!");
        self.next.forward(&resized_out1, train)
    }

    fn backward(&mut self, error: &Array<f32, Self::Output>) -> Array<f32, Self::Input> {
        let err1 = self.next.backward(error);
        let resized_err1 = err1.into_dimensionality().expect("Incompatible array size!");
        self.inner.backward(&resized_err1)
    }

    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> {
        // Recursively grab parameters of all layers in the chain
        let mut parameters = self.inner.get_learnable_parameters();
        parameters.extend(self.next.get_learnable_parameters());
        parameters
    }
}
