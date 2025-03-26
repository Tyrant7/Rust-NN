use ndarray::Array2;

use crate::layers::{Layer, Parameter};

pub struct Model {
    layers: Vec<Box<dyn Layer>>,
    train: bool,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Model {
        let forward_signals = Vec::with_capacity(layers.len());
        Model {
            layers,
            train: true,
        }
    }

    pub fn forward(&mut self, mut input: Array2<f32>) -> Array2<f32> {
        for layer in self.layers.iter_mut() {
            input = layer.forward(&input, self.train);
        }
        input
    }

    pub fn backward(&mut self, mut error: Array2<f32>) {
        for layer in self.layers.iter_mut().rev() {
            error = match layer.backward(&error) {
                Ok(error) => error,
                Err(_) => panic!("Backward called before forward or outside of train mode"),
            };
        }
    }

    pub fn collect_parameters(&mut self) -> Vec<Parameter> {
        let mut parameters = Vec::new();
        for layer in self.layers.iter_mut() {
            parameters.extend(layer.get_learnable_parameters());
        }
        parameters
    }

    pub fn set_train_mode(&mut self, train: bool) {
        self.train = train
    }

    pub fn is_training(&self) -> bool {
        self.train
    }
}
