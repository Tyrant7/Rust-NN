use ndarray::Array2;
use super::{Layer, NoForwardError};

pub struct Sigmoid {
    forward_z: Option<Array2<f32>>,
}

impl Sigmoid {
    pub fn new() -> Sigmoid {
        Sigmoid { forward_z: None }
    }

    fn sigmoid(input: &Array2<f32>) -> Array2<f32> {
        input.clone().mapv_into(|x| 1. / (1. + (-x).exp()))
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Array2<f32>, train: bool) -> Array2<f32> {
        if train {
            self.forward_z = Some(input.clone());
        }
        Sigmoid::sigmoid(input)
    }

    fn backward(&mut self, error: &Array2<f32>) -> Result<Array2<f32>, NoForwardError> {
        let forward_z = match self.forward_z.as_ref() {
            Some(z) => z,
            None => return Err(NoForwardError)
        };

        let sig = Sigmoid::sigmoid(forward_z);
        let activation_derivative = &sig * (1. - &sig);

        Ok(error * activation_derivative)
    }
}
