use std::fmt::Debug;

use ndarray::{Array, ArrayD};

use crate::layers::{CompositeLayer, RawLayer, LearnableParameter};

#[derive(Debug)]
pub struct Tracked<L>
where 
    L: RawLayer
{
    inner: L,
    forward_input: Option<Array<f32, L::Input>>,
}

impl<L> Tracked<L>
where 
    L: RawLayer,
    L::Input: Clone,
{
    pub fn new(layer: L) -> Self {
        Self {
            inner: layer,
            forward_input: None,
        }
    }
}

impl<L> CompositeLayer for Tracked<L> 
where 
    L: RawLayer,
    L::Input: Clone + Debug,
{
    type Input = L::Input;
    type Output = L::Output;

    fn forward(&mut self, input: &Array<f32, Self::Input>, train: bool) -> Array<f32, Self::Output> {
        if train {
            self.forward_input = Some(input.clone());
        }
        self.inner.forward(input, train)
    }

    fn backward(&mut self, error: &Array<f32, Self::Output>) -> Array<f32, Self::Input> {
        let input = self.forward_input
            .as_ref()
            .expect("Backward called before forward or outside of training");
        self.inner.backward(error, input)
    }
}

