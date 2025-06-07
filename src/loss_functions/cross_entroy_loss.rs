use ndarray::{Array1, Array2, Axis};
use super::LossFunction;

pub struct CrossEntropyWithLogitsLoss;

impl LossFunction for CrossEntropyWithLogitsLoss {
    /// This loss function expects raw logits, performing the softmax step as part of itself in order to simplify inner calculations
    fn original(pred: &Array2<f32>, label: &Array2<f32>) -> f32 {
        // (batch_size)
        let max = pred.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        // (batch_size, num_classes)
        let shifted = pred - max.insert_axis(Axis(1));

        // (batch_size)
        let log_sum_exp = shifted.map_axis(Axis(1), |row| row.exp().sum().ln());
        // (batch_size)
        let target_logit = (shifted * label).sum_axis(Axis(1));
        (log_sum_exp - target_logit).sum_axis(Axis(1)).mean().unwrap()
    }

    /// This loss function expects raw logits, performing the softmax step as part of itself in order to simplify inner calculations
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
        (soft - label) / (pred.dim().0 as f32)
    }
}
