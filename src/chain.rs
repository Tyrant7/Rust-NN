use std::fmt::Debug;

use ndarray::ArrayD;

use crate::layers::{Layer, LearnableParameter};

#[derive(Debug)]
struct Chain<L1, L2>
where 
    L1: Layer,
    L2: Layer<Input = L1::Output>,
{
    first: L1,
    second: L2,
    first_forward_input: Option<L1::Input>,
    second_forward_input: Option<L1::Output>,
}

impl<L1, L2> Chain<L1, L2> 
where 
    L1: Layer,
    L2: Layer<Input = L1::Output>,
{
    fn new(first: L1, second: L2) -> Self {
        Self { 
            first, 
            second, 
            first_forward_input: None, 
            second_forward_input: None 
        }
    }
}

impl<L1, L2> Layer for Chain<L1, L2>
where
    L1: Layer,
    L2: Layer<Input = L1::Output>,
    L1::Input: Clone + Debug,
    L1::Output: Clone + Debug,
{
    type Input = L1::Input;
    type Output = L2::Output;

    fn forward(&mut self, input: &Self::Input, train: bool) -> Self::Output {
        let out1 = self.first.forward(input, train);
        self.first_forward_input = Some(input.clone());
        self.second_forward_input = Some(out1.clone());
        self.second.forward(&out1, train)
    }

    fn backward(&mut self, error: &Self::Output, _forward_input: &Self::Input) -> Self::Input {
        let second_input = self.second_forward_input.as_ref().expect("Backward called before forward or outside of train mode");
        let first_input = self.first_forward_input.as_ref().expect("Backward called before forward or outside of train mode");

        let err1 = self.second.backward(error, second_input);
        self.first.backward(&err1, first_input)
    }

    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> {
        // Recursively grab parameters of all layers in the chain
        let mut parameters = self.first.get_learnable_parameters();
        parameters.extend(self.second.get_learnable_parameters());
        return parameters
    }

    // TODO: "inspect()" method to print model layers in a prettier way
}
