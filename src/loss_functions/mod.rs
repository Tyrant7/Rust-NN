use ndarray::{Array1, Array2};

pub trait LossFunction {
    /// Data should be given in (batch, width) format.
    /// Returns the total, unaveraged loss for all sample in the batch. 
    fn original(preds: &Array2<f32>, labels: &Array2<f32>) -> f32;
    /// Data should be given in (batch, width) format.
    /// Returns the average gradient over the batch to avoid scaling gradients with larger batch sizes. 
    fn derivative(preds: &Array2<f32>, labels: &Array2<f32>) -> Array2<f32>;
}

pub mod binary_cross_entropy_loss;
pub use binary_cross_entropy_loss::BCELoss;

pub mod cross_entroy_loss;
pub use cross_entroy_loss::CrossEntropyWithLogitsLoss;

pub mod mean_squared_error_loss;
pub use mean_squared_error_loss::MSELoss;
