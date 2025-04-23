use ndarray::{Array2, Dimension};

use crate::layers::{Layer, LearnableParameter};

#[derive(Debug)]
pub struct Model {
    layers: Vec<Box<dyn Layer<D: Dimension>>>,
    forward_inputs: Vec<Option<Array2<f32>>>,
    pub train: bool,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Model {
        let forward_inputs = Vec::from_iter(layers.iter().map(|_| None));
        Model {
            layers,
            forward_inputs,
            train: true,
        }
    }

    pub fn forward(&mut self, mut input: Array2<f32>) -> Array2<f32> {
        for (layer, layer_input) in self.layers.iter_mut().zip(self.forward_inputs.iter_mut()) {
            if self.train {
                *layer_input = Some(input.clone());
            }
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

    pub fn collect_parameters(&mut self) -> Vec<LearnableParameter> {
        let mut parameters = Vec::new();
        for layer in self.layers.iter_mut() {
            parameters.extend(layer.get_learnable_parameters());
        }
        parameters
    }

    // TODO: "inspect()" method to print model layers in a prettier way
}
