use ndarray::Array2;

use crate::layers::ParameterGroup;
use super::Optimizer;

#[allow(clippy::upper_case_acronyms)]
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    velocities: Vec<Array2<f32>>,
}

impl SGD {
    pub fn new(parameters: &[ParameterGroup], learning_rate: f32, momentum: f32) -> SGD {
        let velocities = parameters.iter().map(|p| Array2::zeros(p.value.raw_dim())).collect();
        SGD {
            learning_rate,
            momentum,
            velocities
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [ParameterGroup], n_samples: usize) {
        for (i, param) in parameters.iter_mut().enumerate() {
            let grad = &*param.gradient / n_samples as f32;
            let update = self.learning_rate * grad + self.momentum * &self.velocities[i];
            *param.value -= &update;

            self.velocities[i] = update;
        }
    }
}
