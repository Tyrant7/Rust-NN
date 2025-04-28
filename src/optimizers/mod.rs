use crate::layers::ParameterGroup;

pub trait Optimizer {
    fn step(&mut self, parameters: &mut [ParameterGroup], n_samples: usize);
    fn zero_gradients(&self, parameters: &mut [ParameterGroup]) {
        for param in parameters.iter_mut() {
            param.gradients = param.gradients.map(|_| 0.);
        }
    }
}

pub mod stochastic_gradient_descent;
pub use stochastic_gradient_descent::SGD;
