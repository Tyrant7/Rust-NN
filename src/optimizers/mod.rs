use crate::layers::ValueGradPair;

pub trait Optimizer {
    fn step(&mut self, parameters: &mut [ValueGradPair], n_samples: usize);
    fn zero_gradients(&self, parameters: &mut [ValueGradPair]) {
        for param in parameters.iter_mut() {
            param.gradients = param.gradients.map(|_| 0.);
        }
    }
}

pub mod stochastic_gradient_descent;
pub use stochastic_gradient_descent::SGD;

