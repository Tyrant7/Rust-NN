use ndarray::{Array1, Array2, ArrayD, Axis};
use super::LossFunction;


/// Mean Squared Error Loss (MSE). 
/// 
/// Used for regression tasks, where the output is not a classification but rather a numerical value
/// representing something. 
/// 
/// MSE penalizes large errors more severely than small ones due to squarring the difference. 
/// 
/// The loss for a single value is defined as:
/// 
/// ```text
/// (y - p)^2
/// ```
/// 
/// The total loss is the sum over all samples in the batch. 
/// 
/// where: 
/// - `y` is the target value,   
/// - `p` is the predicted value.
/// 
/// The derivative with respect to the prediction is:
/// 
/// ```text
/// 2(p - y)
/// ```
/// 
/// Inputs must be shaped as `(batch, width)`.
pub struct MSELoss;

impl LossFunction for MSELoss {
    fn original(preds: &Array2<f32>, labels: &Array2<f32>) -> f32 {
        let mut output = Array1::zeros(preds.dim().0);
        for (b, (pred, label)) in preds.axis_iter(Axis(0)).zip(labels.axis_iter(Axis(0))).enumerate() {
            output[b] = (&label - &pred).pow2().sum();
        }
        output.sum()
    }

    fn derivative(pred: &Array2<f32>, label: &Array2<f32>) -> Array2<f32> {
        // With respect to 'pred'
        (pred - label) * 2.
    }
}
