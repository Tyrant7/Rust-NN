use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use ndarray::{Array, Array2, ArrayD, Dimension, IntoDimension, IxDyn};

use crate::layers::{CompositeLayer, LearnableParameter, RawLayer, Tracked};

/// A compositional wrapper for chaining multiple layers, allowing their forward and backward passes to be connected seamlessly.
///
/// [`Chain`] composes two [`CompositeLayer`]s, forwarding outputs from `inner` to `next`, and ensuring that
/// each layer is receiving its expected shapes from the last.
///
/// Chained layers are most often wrapped in a [`Tracked`] layer in order to allow [`RawLayer`]s to be chained.
///
/// # Panics
/// Panics in both [`Chain::forward`] and [`Chain::backward`] if the output shape of `inner` can not be converted to the input
/// shape of `next`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chain<L1, L2>
where
    L1: CompositeLayer,
    L2: CompositeLayer,
{
    inner: L1,
    next: L2,
}

impl<L1, L2> Chain<L1, L2>
where
    L1: CompositeLayer,
    L2: CompositeLayer,
{
    /// Creates a [`Chain`] with the given layers.
    ///
    /// It is recommended to use the [`chain!`] macro instead of calling this method directly, as it simplifies
    /// chaining and ensures layer are wrapped in [`Tracked`] when needed.   
    pub fn new(inner: L1, next: L2) -> Self {
        Self { inner, next }
    }

    /// Returns a reference to the `inner` layer of this [`Chain`].
    pub fn inner(&self) -> &L1 {
        &self.inner
    }
}

impl<L1, L2> CompositeLayer for Chain<L1, L2>
where
    L1: CompositeLayer,
    L2: CompositeLayer,
{
    type Input = L1::Input;
    type Output = L2::Output;

    fn forward(
        &mut self,
        input: &Array<f32, Self::Input>,
        train: bool,
    ) -> Array<f32, Self::Output> {
        let out1 = self.inner.forward(input, train);
        let resized_out1 = out1.into_dimensionality().expect(
            "Incompatible dimensions between L1 Output and L2 Input during call to forward method",
        );
        self.next.forward(&resized_out1, train)
    }

    fn backward(&mut self, error: &Array<f32, Self::Output>) -> Array<f32, Self::Input> {
        let err1 = self.next.backward(error);
        let resized_err1 = err1.into_dimensionality().expect(
            "Incompatible dimensions between L1 Output and L2 Input during call to backward method",
        );
        self.inner.backward(&resized_err1)
    }

    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> {
        // Recursively grab parameters of all layers in the chain
        let mut parameters = self.inner.get_learnable_parameters();
        parameters.extend(self.next.get_learnable_parameters());
        parameters
    }
}

/// This macro recursively chains together any number of layers, wrapping each one in a [`Tracked`] layer to
/// enable automatic input tracking. It simplifies the construction of deep networks without manually nesting
/// [`Chain`] and [`Tracked`] calls.
#[macro_export]
macro_rules! chain {
    ($a:expr) => {
        Tracked::new($a)
    };
    ($a:expr, $b:expr) => {
        Chain::new(Tracked::new($a), Tracked::new($b))
    };
    ($a:expr, $($rest:expr),+ $(,)?) => {
        Chain::new(Tracked::new($a), chain!($($rest),+))
    };
}
