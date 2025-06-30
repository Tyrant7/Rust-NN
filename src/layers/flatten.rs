use rand::Rng;
use ndarray::{ArrayD, Axis, Dimension, IxDyn};
use serde::{Deserialize, Serialize};

use super::{RawLayer, LearnableParameter, ParameterGroup};

/// A flattening layer used to merge two adjacent axes of a tensor. 
/// 
/// This is typically used when transitioning from convolutional layers (which produce multi-dimensional outputs)
/// to fully connected (linear) layers that expect 2D inputs of shape `(batch, features)`.
/// 
/// For example, flattening axis 2 of a tensor with shape `(batch, channels, height, width)` would produce
/// a shape of `(batch, channels, height * width)`.
/// 
/// # Panics
/// Panics if `axis` refers to the last axis, as there is no adjacent axis to merge with. 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flatten {
    axis: usize,
}

impl Flatten {
    /// Creates a new [`Flatten`] layer that merges the specified axis with the one following it.
    /// 
    /// # Arguments
    /// - `axis`: The axis to flatten with its neighbour. For example, `axis = 1` on shape `(batch, 4, 5)`
    ///   produces shape `(batch, 20)`.
    /// 
    /// # Panics
    /// Will panic during [`Flatten::forward`] if `axis` is the last axis, since there is no adjacent axis to merge with. 
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
        error.to_shape(IxDyn(forward_shape)).expect(
            "Shape mismatch on error signal"
        ).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3};

    use super::*;

    #[test]
    fn forward() {
        let mut flatten = Flatten::new(1);

        let input = Array3::<f32>::zeros((1, 2, 3)).into_dyn();
        let output = flatten.forward(&input, false);

        assert_eq!(output.shape(), [1, 6]);
    }

    #[test]
    fn backward() {
        let mut flatten = Flatten::new(1);

        let input = Array3::<f32>::zeros((1, 2, 3)).into_dyn();
        flatten.forward(&input, false);

        let error = Array2::<f32>::zeros((1, 6)).into_dyn();
        let signal = flatten.backward(&error, &input);

        assert_eq!(signal.shape(), [1, 2, 3]);
    }
}