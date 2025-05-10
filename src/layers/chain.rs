use std::fmt::Debug;

use ndarray::ArrayD;

use crate::layers::{CompositeLayer, RawLayer, LearnableParameter};

#[derive(Debug)]
pub struct Chain<L1, L2>
where 
    L1: CompositeLayer,
    L2: CompositeLayer<Input = L1::Output>,
{
    first: L1,
    second: L2,
}

impl<L1, L2> Chain<L1, L2> 
where 
    L1: CompositeLayer,
    L2: CompositeLayer<Input = L1::Output>,
{
    fn new(first: L1, second: L2) -> Self {
        Self { 
            first, 
            second,
        }
    }
}

impl<L1, L2> CompositeLayer for Chain<L1, L2>
where
    L1: CompositeLayer,
    L2: CompositeLayer<Input = L1::Output>,
{
    type Input = L1::Input;
    type Output = L2::Output;

    fn forward(&mut self, input: &Self::Input, train: bool) -> Self::Output {
        let out1 = self.first.forward(input, train);
        self.second.forward(&out1, train)
    }

    fn backward(&mut self, error: &Self::Output) -> Self::Input {
        let err1 = self.second.backward(error);
        self.first.backward(&err1)
    }

    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> {
        // Recursively grab parameters of all layers in the chain
        let mut parameters = self.first.get_learnable_parameters();
        parameters.extend(self.second.get_learnable_parameters());
        parameters
    }
}
