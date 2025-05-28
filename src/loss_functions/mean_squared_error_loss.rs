use ndarray::{Array1, Array2, ArrayD, Axis};
use super::LossFunction;

pub struct MSELoss;

impl LossFunction for MSELoss {
    fn original(pred: &Array2<f32>, label: &Array2<f32>) -> Array1<f32> {
        (label - pred).pow2().sum_axis(Axis(0))
    }

    fn derivative(pred: &Array2<f32>, label: &Array2<f32>) -> Array2<f32> {
        // With respect to 'pred'
        -(label - pred) * 2.
    }
}
