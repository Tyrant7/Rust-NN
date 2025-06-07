use ndarray::{Array, ArrayD};

use crate::layers::LearnableParameter;

use super::Optimizer;

#[allow(clippy::upper_case_acronyms)]
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    velocities: Vec<ArrayD<f32>>,
}

impl SGD {
    pub fn new(parameters: &[LearnableParameter], learning_rate: f32, momentum: f32) -> SGD {
        let velocities = parameters.iter().map(|p| Array::zeros(p.gradients.raw_dim())).collect();
        SGD {
            learning_rate,
            momentum,
            velocities
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [LearnableParameter]) {        
        for (i, param) in parameters.iter_mut().enumerate() {
            let update = &self.velocities[i] * self.momentum + &param.gradients * self.learning_rate;
            param.values -= &update;

            self.velocities[i] = update;
        }
    }
}
