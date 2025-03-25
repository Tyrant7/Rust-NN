use ndarray::Array2;

use crate::layers::Layer;

pub struct Model {
    layers: Vec<Box<dyn Layer>>,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Model {
        Model {
            layers,
        }
    }

    pub fn forward(&mut self, mut input: Array2<f32>) -> Array2<f32> {
        for layer in self.layers.iter_mut() {
            input = layer.forward(&input);
        }
        input
    }

    pub fn backward(&mut self, mut error: Array2<f32>) {
        for layer in self.layers.iter_mut().rev() {
            error = layer.backward(&error);
        }
    }

    pub fn apply_gradients(&mut self, lr: f32, batch_size: usize) {
        for layer in self.layers.iter_mut() {
            layer.apply_gradients(lr, batch_size);
            layer.zero_gradients();
        }
    }
}
