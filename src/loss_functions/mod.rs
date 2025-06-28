use ndarray::{Array1, Array2};

/// Trait for loss functions used in neural networks. 
/// 
/// Implement this trait to define a loss function for training. 
/// It provides the core methods required during the forward and backward passes
/// of backpropagation. 
/// 
/// All inputs and outputs are expected to be in `(batch, width)` format, where each row
/// corresponds to one sample in the batch. 
/// 
/// # Required Methods:
/// - [`Self::original`]: Computes the total (unaveraged) loss for all output samples in a batch. 
/// - [`Self::derivative`]: Computes the gradient of the loss with respect to each prediction in a batch. 
pub trait LossFunction {
    /// Computes the total (unaveraged) loss over the batch. 
    /// 
    /// # Arguments
    /// - `preds`: The model's predictions, shape `(batch, width)`.
    /// - `labels`: The ground-truth labels, shape `(batch, width)`.
    fn original(preds: &Array2<f32>, labels: &Array2<f32>) -> f32;

    /// Computes the gradients of the loss with respect to `preds` for the entire batch. 
    /// 
    /// # Arguments
    /// - `preds`: The model's predictions, shape `(batch, width)`.
    /// - `labels`: The ground-truth labels, shape `(batch, width)`.
    /// 
    /// # Returns
    /// A matrix of shape `(batch, width)` containing the per-sample gradients. 
    fn derivative(preds: &Array2<f32>, labels: &Array2<f32>) -> Array2<f32>;
}

pub mod binary_cross_entropy_loss;
pub use binary_cross_entropy_loss::BCELoss;

pub mod cross_entroy_loss;
pub use cross_entroy_loss::CrossEntropyWithLogitsLoss;

pub mod mean_squared_error_loss;
pub use mean_squared_error_loss::MSELoss;
