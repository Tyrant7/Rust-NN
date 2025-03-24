use ndarray::Array2;
use super::Layer;

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
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward_z = Some(input.clone());
        Sigmoid::sigmoid(input)
    }

    fn backward(&mut self, error: &Array2<f32>) -> Array2<f32> {
        let forward_z = self.forward_z.as_ref().expect("Backward called before forward");

        let sig = Sigmoid::sigmoid(forward_z);
        let activation_derivative = &sig * (1. - &sig);

        error * activation_derivative
    }
}
