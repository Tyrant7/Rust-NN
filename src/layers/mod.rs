use ndarray::Array2;

pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>, train: bool) -> Array2<f32>;
    fn backward(&mut self, error: &Array2<f32>) -> Result<Array2<f32>, NoForwardError>;

    // Not all layers have learnable parameters
    fn get_learnable_parameters(&mut self) -> Vec<Parameter> { Vec::new() }
}

pub struct NoForwardError;

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