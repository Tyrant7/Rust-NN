use crate::model::Model;

pub trait Optimizer {
    fn step(&mut self, network: &mut Model);
}

pub mod stochastic_gradient_descent;
pub use stochastic_gradient_descent::SGD;
