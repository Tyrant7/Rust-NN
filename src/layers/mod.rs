use ndarray::Array2;

pub trait Layer {
    // Each layer has a custom state where it can store its forward computations
    type State;

    fn forward(&mut self, input: &Array2<f32>, train: bool) -> (Array2<f32>, Self::State);
    fn backward(&mut self, input: &Array2<f32>, state: Self::State) -> Array2<f32>;

    // Not all layers have learnable parameters
    fn get_learnable_parameters(&mut self) -> Vec<Parameter> { Vec::new() }
}

enum LayerState<S> {
    NotComputed,
    ForwardComputed(S),
}

struct SequentialLayer<T: Layer> {
    layer: T,
    state: LayerState<T::State>,
    samples: usize,
}

impl<T: Layer> SequentialLayer<T> {
    fn new(layer: T) -> Self {
        SequentialLayer { 
            layer, 
            state: LayerState::NotComputed,
            samples: 0,
        }
    }

    fn forward(&mut self, input: &Array2<f32>, train: bool) -> Array2<f32> {
        let (output, state) = self.layer.forward(input, train);
        self.state = LayerState::ForwardComputed(state);
        output
    }

    fn backward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match &self.state {
            LayerState::ForwardComputed(state) => self.layer.backward(input, state),
            LayerState::NotComputed => panic!("Backward called before forward or while not in train mode"),
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