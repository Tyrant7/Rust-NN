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

    pub fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        let mut forward_signal = input;
        for layer in self.layers.iter_mut() {
            forward_signal = layer.forward(&forward_signal);
        }
        forward_signal
    }

    pub fn backward(&mut self, loss_fn) {
        
    }
}
