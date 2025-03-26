use ndarray::Array2;

pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>, train: bool) -> Array2<f32>;
    fn backward(&mut self, input: &Array2<f32>, forward_input: &Array2<f32>) -> Array2<f32>;

    // Not all layers have learnable parameters
    fn get_learnable_parameters(&mut self) -> Vec<Parameter> { Vec::new() }
}

pub struct SequentialLayer<T: Layer> {
    layer: T,
    forward_input: Option<Array2<f32>>,
    samples: usize,
}

impl<T: Layer> SequentialLayer<T> {
    fn new(layer: T) -> Self {
        SequentialLayer { 
            layer, 
            forward_input: None,
            samples: 0,
        }
    }

    fn forward(&mut self, input: &Array2<f32>, train: bool) -> Array2<f32> {
        self.forward_input = Some(input.clone());
        self.layer.forward(input, train)
    }

    fn backward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match &self.forward_input {
            Some(forward) => self.layer.backward(input, forward),
            None => panic!("Backward called before forward or outside of train mode"),
        }
    }
}

pub struct Parameter<'a> {
    pub value: &'a mut Array2<f32>,
    pub gradient: &'a mut Array2<f32>,
}

pub mod linear;
pub use linear::Linear;

pub mod dropout;
pub use dropout::Dropout;

pub mod relu;
pub use relu::ReLU;
pub mod sigmoid;
pub use sigmoid::Sigmoid;