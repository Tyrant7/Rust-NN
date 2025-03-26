use ndarray::Array2;

use crate::layers::Parameter;
use super::Optimizer;

#[allow(clippy::upper_case_acronyms)]
pub struct SGD<'a> {
    parameters: Vec<Parameter<'a>>,
    learning_rate: f32,
    momentum: f32,
    velocities: Vec<Array2<f32>>,
}

impl<'a> SGD<'a> {
    pub fn new(parameters: Vec<Parameter<'_>>, learning_rate: f32, momentum: f32) -> SGD {
        let mut velocities = Vec::new();
        for parameter in parameters.iter() {
            velocities.push(Array2::zeros(parameter.value.raw_dim()));
        }
        SGD {
            parameters,
            learning_rate,
            momentum,
            velocities,
        }
    }
}

impl<'a> Optimizer for SGD<'a> {
    fn step(&mut self, parameters: &mut [Parameter], n_samples: usize) {
        for (i, param) in parameters.iter_mut().enumerate() {
            let grad = &*param.gradient / n_samples as f32;
            let update = grad * self.learning_rate + &self.velocities[i] * self.momentum;
            *param.value -= &update;

            self.velocities[i] = update;
        }
    }
}
