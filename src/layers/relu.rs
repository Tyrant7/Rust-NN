use ndarray::Array2;
use super::Layer;

pub struct ReLU {
    forward_z: Option<Array2<f32>>,
}

impl ReLU {
    pub fn new() -> ReLU {
        ReLU { forward_z: None }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Keep track of the input before activation to handle it during backprop
        self.forward_z = Some(input.clone());
        input.clone().mapv_into(|x| x.max(0.))
    }

    fn backward(&mut self, error: &Array2<f32>) -> Array2<f32> {
        let forward_z = self.forward_z.as_ref().expect("Backward called before forward");
        let activation_derivative = forward_z.clone().mapv_into(|x| if x <= 0. { 0. } else { 1. });
        error * activation_derivative
    }
}
