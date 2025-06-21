use rand::Rng;
use ndarray::{ArrayD, Axis, Dimension, IxDyn};
use serde::{Deserialize, Serialize};

use super::{RawLayer, LearnableParameter, ParameterGroup};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flatten {
    axis: usize,
}

impl Flatten {
    pub fn new(axis: usize) -> Self {
        Flatten { axis }
    }
}

impl RawLayer for Flatten 
{
    type Input = IxDyn;
    type Output = IxDyn;

    fn forward(&mut self, input: &ArrayD<f32>, _train: bool) -> ArrayD<f32> {
        let shape = input.shape();
        assert!(self.axis < shape.len() - 1, "Cannot flatten the last axis of input data!");

        let mut new_shape = shape.to_vec();
        let merged = new_shape[self.axis] * new_shape[self.axis + 1];
        new_shape.splice(self.axis..=self.axis + 1, [merged]);

        input.to_shape(IxDyn(&new_shape)).unwrap().to_owned()
    }

    fn backward(&mut self, error: &ArrayD<f32>, forward_input: &ArrayD<f32>) -> ArrayD<f32> {
        let forward_shape = forward_input.shape();
        error.to_shape(IxDyn(forward_shape)).unwrap().to_owned()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;

    #[test]
    fn forward() {
        // TODO
    }

    #[test]
    fn backward() {
        // TODO
    }
}