use ndarray::{stack, Array1, Array2, ArrayD, Axis};
use super::LossFunction;

pub struct CrossEntropyLoss;

impl LossFunction for CrossEntropyLoss {
    /// This function expects raw logits, performaing the softmax step as part of itself in order to simplify inner calculations
    fn original(pred: &Array2<f32>, label: &Array2<f32>) -> Array1<f32> {
        // (batch_size)
        let max = pred.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        // (batch_size, num_classes)
        let shifted = pred - max.insert_axis(Axis(1));

        // (batch_size)
        let log_sum_exp = shifted.map_axis(Axis(1), |row| row.exp().sum().ln());
        // (batch_size)
        let target_logit = (shifted * label).sum_axis(Axis(1));
        log_sum_exp - target_logit
    }

    /// This function expects raw logits, performaing the softmax step as part of itself in order to simplify inner calculations
    fn derivative(pred: &Array2<f32>, label: &Array2<f32>) -> Array2<f32> {
        pred - label
    }
}
