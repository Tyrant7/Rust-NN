use ndarray::{Array1, Array2, Axis};
use super::LossFunction;


/// Cross Entropy Loss with Logits (CEL). 
/// 
/// Used for multi-class classification tasks where the output is a distribution over classes. 
/// This loss function expects **raw logits**, not probabilities. It applies the softmax function internally
/// to improve numerical stability and efficiency. 
/// 
/// CEL penalizes confident incorrect predictions and is commonly used with softmax activations. 
/// 
/// The formula for a single sample is defined as:
/// 
/// ```text
/// -target_y * ln(softmax_p)
/// ```
/// 
/// where: 
/// - `target_y` is a one-hot encoded vector containing the true label (0 or 1), 
/// - `softmax_p` is a vector of predicted probabilities for each class computed via softmax over the logits.
/// 
/// The simplified derivative (used for backpropagation) is:
/// 
/// ```text
/// softmax_p - target_y
/// ```
/// 
/// Inputs must be shaped as `(batch, num_classes)`.
pub struct CrossEntropyWithLogitsLoss;

impl LossFunction for CrossEntropyWithLogitsLoss {
    fn original(pred: &Array2<f32>, label: &Array2<f32>) -> f32 {
        // (batch_size)
        let max = pred.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        // (batch_size, num_classes)
        let shifted = pred - max.insert_axis(Axis(1));

        // (batch_size)
        let log_sum_exp = shifted.map_axis(Axis(1), |row| row.exp().sum().ln());
        // (batch_size)
        let target_logit = (shifted * label).sum_axis(Axis(1));
        (log_sum_exp - target_logit).sum()
    }

    fn derivative(pred: &Array2<f32>, label: &Array2<f32>) -> Array2<f32> {
        // (batch_size)
        let max = pred.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));

        // (batch_size, num_classes)
        let shifted = pred - max.insert_axis(Axis(1));

        // Apply softmax row-wise
        let exp = shifted.exp();
        let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
        let soft = exp / sum;

        // Gradient of cross-entropy with softmax = softmax - label
        soft - label
    }
}
