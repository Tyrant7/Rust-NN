use ndarray::Array2;

use crate::layers::{Layer, Parameter};

pub struct Model {
    layers: Vec<Box<dyn Layer>>,
    forward_inputs: Vec<Option<Array2<f32>>>,
    train: bool,
    samples: usize,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Model {
        let forward_inputs = Vec::from_iter(layers.iter().map(|_| None));
        Model {
            layers,
            forward_inputs,
            train: true,
            samples: 0,
        }
    }

    pub fn forward(&mut self, mut input: Array2<f32>) -> Array2<f32> {
        for (layer, layer_input) in self.layers.iter_mut().zip(self.forward_inputs.iter_mut()) {
            *layer_input = Some(input.clone());
            input = layer.forward(&input, self.train);
        }
        input
    }

    pub fn backward(&mut self, mut error: Array2<f32>) {
        for (layer, layer_input) in self.layers.iter_mut().zip(self.forward_inputs.iter_mut()).rev() {
            error = match layer_input {
                Some(forward) => layer.backward(&error, forward),
                None => panic!("Backward called before forward or outside of train mode"),
            }
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
