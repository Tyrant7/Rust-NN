use crate::{layers::ParameterGroup, tensor::Tensor};
use super::Optimizer;

#[allow(clippy::upper_case_acronyms)]
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    velocities: Vec<Tensor>,
}

impl SGD {
    pub fn new(parameters: &[ParameterGroup], learning_rate: f32, momentum: f32) -> SGD {
        let velocities = parameters.iter().map(|p| p.gradients.map(|_| 0.)).collect();
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
            let grad = param.gradients.map(|x| x / (n_samples as f32));
            let update = &self.velocities[i] * self.momentum + grad * self.learning_rate;
            param.values -= &update;

            self.velocities[i] = update;
        }
    }
}
