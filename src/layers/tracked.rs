use std::fmt::Debug;

use ndarray::{Array, ArrayD, Dimension};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::layers::{CompositeLayer, RawLayer, LearnableParameter};

/// A wrapper forward that records the forward inputs for later use in backpropagation.
/// 
/// `Tracked` is used to bridge [`RawLayer`]s into contexts where a [`CompositeLayer`] is expected, such as when building 
/// a [`Chain`] of layers. It enables proper gradient calculation by saving the input passed to [`RawLayer::forward`] during
/// training, so that it can be reused during the call to [`RawLayer::backward`]. 
/// 
/// This is especially useful for stateless layers that compute derivatives based on their forward input (e.g., activations). 
/// 
/// During inference (i.e., when `train = false`), the forward input is not saved to conserve memory. 
/// 
/// # Panics
/// Panics in [`CompositeLayer::backward`] if it is called before training-mode forward pass has occured. 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tracked<L>
where 
    L: RawLayer,
    L::Input: Serialize + DeserializeOwned
{
    inner: L,
    #[serde(skip)]
    forward_input: Option<Array<f32, L::Input>>,
}

impl<L> Tracked<L>
where 
    L: RawLayer,
    L::Input: Clone + Serialize + DeserializeOwned,
{
    /// Wraps the given layer in a new [`Tracked`] layer. 
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
    L::Input: Clone + Debug + Serialize + DeserializeOwned,
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

    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> {
        self.inner.get_learnable_parameters()
    }
}