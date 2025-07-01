use ndarray::{ArrayD, IxDyn};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use super::RawLayer;

/// A dropout layer that randomly disables a subset of inputs during training.
///
/// Dropout is a regularization technique used to prevent overfitting in neural networks,
/// especially between fully connected layers. During the forward pass in trainng mode, each input
/// value has a probability (`rate`) of being set to zero. This forces the network to not
/// rely too heavily on any single neuron.
///
/// To preserve the expected magnitude of the signal, remaining (non-zeroed) values are scaled
/// by `1 / (1 - rate)` during both forward and backward passes.
///
/// During inference (when `train = false`), dropout is disabled and the input is passed through unchanged.
///
/// # Parameters
/// - `rate`: The probability of dropping each element `[0.0, 1.0]`.
///   Typical values are `0.2` for small models and up to `0.5` for deeper architectures.
///   Values above `0.5` are generally discouraged and may affect the model's ability to generalize effectively.
///
/// # Panics
/// Panics if `rate` is not within the inclusive range `[0.0, 1.0]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dropout {
    rate: f32,
    #[serde(skip)]
    forward_mask: Option<ArrayD<f32>>,
}

impl Dropout {
    /// Creates a new [`Dropout`] layer with the given drop `rate`.
    ///
    /// # Arguments:
    /// - `rate`: Probability of dropping each input unit (must be in `[0.0, 1.0]`).
    ///
    /// # Panics
    /// Panics if `rate` is outside the valid range.
    pub fn new(rate: f32) -> Dropout {
        assert!((0.0..=1.).contains(&rate), "Invalid rate provided: {rate}");
        Dropout {
            rate,
            forward_mask: None,
        }
    }
}

impl RawLayer for Dropout {
    type Input = IxDyn;
    type Output = IxDyn;

    fn forward(&mut self, input: &ArrayD<f32>, train: bool) -> ArrayD<f32> {
        if !train {
            return input.clone();
        }

        let mask = input.map(|_| {
            if rand::random::<f32>() > self.rate {
                1.
            } else {
                0.
            }
        });
        // Here `1. - self.rate` acts as a scaling factor to ensure that the dropout of certain pathways
        // doesn't affect the mangitude of the signal moving through the network
        // The same principle applies during the backward pass
        let output = input * &mask / (1. - self.rate);
        self.forward_mask = Some(mask);
        output
    }

    fn backward(&mut self, delta: &ArrayD<f32>, _forward_input: &ArrayD<f32>) -> ArrayD<f32> {
        let mask = self
            .forward_mask
            .as_ref()
            .expect("No mask created during forward pass or forward never called");
        delta * mask / (1. - self.rate)
    }
}
