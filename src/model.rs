use ndarray::Array2;

use crate::layers::{Layer, Parameter, SequentialLayer};

pub struct Model {
    layers: Vec<SequentialLayer<Layer>>,
    train: bool,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Model {
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
            error = layer.backward(&error);
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
