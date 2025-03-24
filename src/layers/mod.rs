use ndarray::Array2;

pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, error: &Array2<f32>) -> Array2<f32>;

    // Not all layers have learnable parameters
    fn apply_gradients(&mut self, _lr: f32, _batch_size: usize) {}
    fn zero_gradients(&mut self) {}
}

pub mod linear;
pub use linear::Linear;

pub mod relu;
pub use relu::ReLU;
pub mod sigmoid;
pub use sigmoid::Sigmoid;