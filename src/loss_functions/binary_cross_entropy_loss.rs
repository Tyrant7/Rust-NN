use ndarray::ArrayD;
use super::LossFunction;

pub struct BCELoss;

impl LossFunction for BCELoss {
    fn original(pred: &ArrayD<f32>, label: &ArrayD<f32>) -> f32 {
        // To prevent log(0)
        let epsilon = 1e-12;
        -(label * (pred + epsilon).ln() + (1. - label) * (1. - pred + epsilon).ln()).sum()
    }

    fn derivative(pred: &ArrayD<f32>, label: &ArrayD<f32>) -> ArrayD<f32> {
        let epsilon = 1e-12;
        (pred - label) / ((pred * (1. - pred)) + epsilon)
    }
}
