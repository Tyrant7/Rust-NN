use ndarray::{Array, ArrayD};

use super::Optimizer;
use crate::layers::LearnableParameter;

/// Stochastic Gradient Descent (SGD)
///
/// A traditional optimization algorithm that updates parameters using gradients
/// with optional momentum support to smooth updates over time.
///
/// Updates are computed using the following formulas:
///
/// ```text
/// v = v_prev * m + ∇p * lr
/// p = p - v
/// ```
///
/// where:
/// - `v` is the velocity (update step),
/// - `v_prev` is the previous velocity,
/// - `m` is the momentum coefficient,
/// - `∇p` is the gradient of the parameter,
/// - `lr` is the constant learning rate coefficient,
/// - `p` is the parameter being updated.
#[allow(clippy::upper_case_acronyms)]
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    velocities: Vec<ArrayD<f32>>,
}

impl SGD {
    /// Creates a new [`SGD`] optimizer for the given parameters, learning rate, and momentum.
    ///
    /// # Arguments
    ///
    /// * `parameters` - A slice of [`LearnableParameter`]s used to initialize the shape of velocity buffers.
    /// * `learning_rate` - The learning rate used to scale gradient updates.
    /// * `momentum` - The momentum used to scale prior velocity contributions.
    pub fn new(parameters: &[LearnableParameter], learning_rate: f32, momentum: f32) -> SGD {
        let velocities = parameters
            .iter()
            .map(|p| Array::zeros(p.gradients.raw_dim()))
            .collect();
        SGD {
            learning_rate,
            momentum,
            velocities,
        }
    }
}

impl Optimizer for SGD {
    /// Applies gradient updates to the given parameters using SGD with momentum.
    ///
    /// Updates are saved internally for momentum to take effect on future steps.
    fn step(&mut self, parameters: &mut [LearnableParameter]) {
        for (i, param) in parameters.iter_mut().enumerate() {
            let update =
                &self.velocities[i] * self.momentum + &param.gradients * self.learning_rate;
            param.values -= &update;

            self.velocities[i] = update;
        }
    }
}
