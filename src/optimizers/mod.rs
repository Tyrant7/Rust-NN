use crate::Model;

pub trait Optimizer {
    fn step(&self, network: &mut Model);
}

pub mod stochastic_gradient_descent;
pub use stochastic_gradient_descent::SGD;
