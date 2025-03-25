use ndarray::Array2;

use crate::layers::{Layer, Parameter};

pub struct Model {
    layers: Vec<Box<dyn Layer>>,
    samples: usize,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Model {
        Model {
            layers,
            samples: 0,
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
        self.samples += 1;
    }

    pub fn collect_parameters(&mut self) -> Vec<Parameter> {
        let mut parameters = Vec::new();
        for layer in self.layers.iter_mut() {
            parameters.extend(layer.get_learnable_parameters());
        }
        parameters
    }

    pub fn get_samples(&self) -> usize {
        self.samples
    }

    pub fn zero_gradients(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.zero_gradients();
        }
        self.samples = 0;
    }
}
