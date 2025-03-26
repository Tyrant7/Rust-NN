use crate::layers::Parameter;

pub trait Optimizer {
    fn step(&mut self, parameters: &mut [Parameter], n_samples: usize);
    fn zero_gradients(&self, parameters: &mut [Parameter]) {
        for param in parameters.iter_mut() {
            *param.gradient = 0.;
        }
    }
}

pub mod stochastic_gradient_descent;
pub use stochastic_gradient_descent::SGD;
