use crate::layers::Parameter;
use super::Optimizer;

#[allow(clippy::upper_case_acronyms)]
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    velocities: Vec<f32>,
}

impl SGD {
    pub fn new(parameters: &[Parameter], learning_rate: f32, momentum: f32) -> SGD {
        let velocities = vec![0.; parameters.len()];
        SGD {
            learning_rate,
            momentum,
            velocities
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [Parameter], n_samples: usize) {
        for (i, param) in parameters.iter_mut().enumerate() {
            let grad = *param.gradient / n_samples as f32;
            let update = self.learning_rate * grad + self.momentum * self.velocities[i];
            *param.value -= update;

            self.velocities[i] = update;
        }
    }
}
