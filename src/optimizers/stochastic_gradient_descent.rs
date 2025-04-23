use ndarray::{Array2, ArrayBase};

use crate::layers::{LearnableParameter, ParameterGroup};
use super::Optimizer;

#[allow(clippy::upper_case_acronyms)]
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    velocities: Vec<Array2<f32>>,
}

impl SGD {
    pub fn new(parameters: &[LearnableParameter], learning_rate: f32, momentum: f32) -> SGD {
        let velocities = parameters.iter().map(|p| p.clone_shape());
        SGD {
            learning_rate,
            momentum,
            velocities
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [LearnableParameter], n_samples: usize) {
        for (i, param) in parameters.iter_mut().enumerate() {
            let grad = &*param.gradient / n_samples as f32;
            let update = self.learning_rate * grad + self.momentum * &self.velocities[i];
            *param.value -= &update;

            self.velocities[i] = update;
        }
    }
}
