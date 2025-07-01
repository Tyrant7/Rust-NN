use crate::layers::LearnableParameter;

/// Trait for optimization algorithms used during model training.
/// Inspired by PyTorch's `Optimizer` class.
///
/// Optimizers adjust model parameters using gradients computed during backpropagation.
///
/// Implement this trait to define an optimizer for training.
/// It provides core methods to update parameters and zero gradients.
///
/// # Required Methods:
/// - [`Self::step`]: Applies the gradients calculated during backpropagation to each given [`LearnableParameter`].
pub trait Optimizer {
    /// Updates each [`LearnableParameter`] using its stored gradients and the optimizer's internal rules
    /// (e.g. learning rate, momentum, etc.).
    fn step(&mut self, parameters: &mut [LearnableParameter]);

    /// Zeros the gradients for all given [`LearnableParameter`]s. This function will typically be called
    /// after a call to `step` to reset gradients after application.
    ///
    /// Overwriting this method is not recommended, as it provides a standard implementation for resetting
    /// gradients and ensures consistency across optimizers.
    fn zero_gradients(&self, parameters: &mut [LearnableParameter]) {
        parameters.iter_mut().for_each(|p| p.zero_grads());
    }
}

pub mod stochastic_gradient_descent;
pub use stochastic_gradient_descent::SGD;
