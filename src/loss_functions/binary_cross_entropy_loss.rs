use ndarray::{Array1, Array2, ArrayD, Axis};
use super::LossFunction;

/// Binary Cross Entropy Loss (BCE). 
/// 
/// Commonly used for binary classification tasks (i.e. "yes or no" problems). 
/// BCE is generally paired with a sigmoid activation function and expects 
/// predicted probabilities in the range `[0, 1]`. 
/// 
/// BCE strongly penalizes confident incorrect predictions. 
/// 
/// The loss for a single data point is defined as:
/// 
/// ```text
/// -[y * ln(p) + (1 - y) * ln(1 - p)]
/// ```
/// 
/// The derivative (used for backpropagation) is: 
/// 
/// ```text
/// -(y / p) + (1 - y) / (1 - p)
/// ```
/// 
/// where: 
/// - `y` is the true label (0 or 1),
/// - `p` is the predicted probability of the positive class (between 0 and 1).
/// 
/// Inputs must be shaped as `(batch, width)`.
pub struct BCELoss;

impl LossFunction for BCELoss {
    fn original(pred: &Array2<f32>, label: &Array2<f32>) -> f32 {
        // To prevent log(0)
        let pred = pred.mapv(|x| x.clamp(f32::EPSILON, 1. - f32::EPSILON));
        let loss = -(label * pred.ln() + (1. - label) * (1. - pred).ln());
        loss.sum_axis(Axis(1)).sum()
    }

    fn derivative(pred: &Array2<f32>, label: &Array2<f32>) -> Array2<f32> {
        let pred = pred.mapv(|x| x.clamp(f32::EPSILON, 1. - f32::EPSILON));
        -(label / &pred) + ((1. - label) / (1. - &pred))
    }
}
