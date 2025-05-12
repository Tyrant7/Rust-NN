use crate::layers::LearnableParameter;

pub trait Optimizer {
    fn step(&mut self, parameters: &mut [LearnableParameter], n_samples: usize);
    fn zero_gradients(&self, parameters: &mut [LearnableParameter]) {
        parameters.iter_mut().for_each(|p| p.zero_grads());
    }
}

pub mod stochastic_gradient_descent;
pub use stochastic_gradient_descent::SGD;
