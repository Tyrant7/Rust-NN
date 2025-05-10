use std::fmt::Debug;

use ndarray::{ArrayD, IntoDimension};

use crate::layers::{CompositeLayer, RawLayer, LearnableParameter};

#[derive(Debug)]
pub struct Chain<L1, L2>
where 
    L1: CompositeLayer<Output: IntoDimension<Dim = L2::Input>>,
    L2: CompositeLayer<Input: IntoDimension<Dim = L1::Output>>,
{
    inner: L1,
    next: L2,
}

impl<L1, L2> Chain<L1, L2> 
where 
    L1: CompositeLayer<Output: IntoDimension<Dim = L2::Input>>,
    L2: CompositeLayer<Input: IntoDimension<Dim = L1::Output>>,
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
    L1: CompositeLayer<Output: IntoDimension<Dim = L2::Input>>,
    L2: CompositeLayer<Input: IntoDimension<Dim = L1::Output>>,
{
    type Input = L1::Input;
    type Output = L2::Output;

    fn forward(&mut self, input: &Self::Input, train: bool) -> Self::Output {
        let out1 = self.inner.forward(input, train);
        self.next.forward(&out1.into_dimension(), train)
    }

    fn backward(&mut self, error: &Self::Output) -> Self::Input {
        let err1 = self.next.backward(error);
        self.inner.backward(&err1.into_dimension())
    }

    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> {
        // Recursively grab parameters of all layers in the chain
        let mut parameters = self.inner.get_learnable_parameters();
        parameters.extend(self.next.get_learnable_parameters());
        parameters
    }
}
