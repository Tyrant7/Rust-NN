use rand::Rng;
use ndarray::{Array2, ArrayBase, Axis, Data, Dimension};

use super::{Layer, Parameter};

pub struct Convolutional<T, D>
where 
    T: Data<Elem = f32>,
    D: Dimension,
{
    dimensions: ArrayBase<T, D>,

    weights: Array2<f32>,
    bias: Array2<f32>,
    wgrads: Array2<f32>,
    bgrads: Array2<f32>,
}

impl Convolutional {
    pub fn new_from_rand() -> Convolutional {
        unimplemented!();
    }
}

impl Layer for Convolutional {
    fn forward(&mut self, input: &Array2<f32>, _train: bool) -> Array2<f32> {
        unimplemented!();
    }

    // Here, we'll be fed the delta after the activation derivative has been applied,
    // since the activation functions will handle that portion themselves
    fn backward(&mut self, delta: &Array2<f32>, forward_input: &Array2<f32>) -> Array2<f32> {
        unimplemented!();
    }

    fn get_learnable_parameters(&mut self) -> Vec<Parameter> {
        unimplemented!();
    }
}
