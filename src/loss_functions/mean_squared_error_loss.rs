use ndarray::ArrayD;
use super::LossFunction;

pub struct MSELoss;

impl LossFunction for MSELoss {
    fn original(pred: &ArrayD<f32>, label: &ArrayD<f32>) -> f32 {
        (label - pred).pow2().sum()
    }

    fn derivative(pred: &ArrayD<f32>, label: &ArrayD<f32>) -> ArrayD<f32> {
        // With respect to 'pred'
        -(label - pred) * 2.
    }
}
