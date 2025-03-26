use ndarray::Array2;
use super::LossFunction;

pub struct MSELoss;

impl LossFunction for MSELoss {
    fn original(pred: &Array2<f32>, label: &Array2<f32>) -> f32 {
        (label - pred).pow2().sum()
    }

    fn derivative(pred: &Array2<f32>, label: &Array2<f32>) -> Array2<f32> {
        // With respect to 'pred'
        -(label - pred) * 2.
    }
}
