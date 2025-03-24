use ndarray::Array2;

pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, error: &Array2<f32>) -> Array2<f32>;
    fn apply_gradients(&mut self, lr: f32, batch_size: usize);
    fn zero_gradients(&mut self);
}

pub mod linear;
pub use linear::Linear;
