use ndarray::{Array1, Array2};

pub trait LossFunction {
    /// Data should be given in (batch, width) format.
    /// Returns an array of loss for each batch. 
    fn original(preds: &Array2<f32>, labels: &Array2<f32>) -> Array1<f32>;
    fn derivative(preds: &Array2<f32>, labels: &Array2<f32>) -> Array2<f32>;
}

pub mod binary_cross_entropy_loss;
pub use binary_cross_entropy_loss::BCELoss;

pub mod cross_entroy_loss;
pub use cross_entroy_loss::CrossEntropyWithLogitsLoss;

pub mod mean_squared_error_loss;
pub use mean_squared_error_loss::MSELoss;
