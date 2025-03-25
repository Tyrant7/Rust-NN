use ndarray::Array2;

pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, error: &Array2<f32>) -> Array2<f32>;

    // Not all layers have learnable parameters
    fn zero_gradients(&mut self) {}
    fn get_learnable_parameters(&mut self) -> Vec<Parameter> { Vec::new() }
}

pub struct Parameter<'a> {
    pub value: &'a mut f32,
    pub gradient: &'a mut f32,
}

pub mod linear;
pub use linear::Linear;

pub mod relu;
pub use relu::ReLU;
pub mod sigmoid;
pub use sigmoid::Sigmoid;