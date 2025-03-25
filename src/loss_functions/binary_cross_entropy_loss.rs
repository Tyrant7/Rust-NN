use ndarray::Array2;
use super::LossFunction;

pub struct BCELoss;

impl LossFunction for BCELoss {
    fn original(pred: &Array2<f32>, label: &Array2<f32>) -> f32 {
        // To prevent log(0)
        let epsilon = 1e-12;
        -(label * (pred + epsilon).ln() + (1. - label) * (1. - pred + epsilon).ln()).sum()
    }

    fn derivative(pred: &Array2<f32>, label: &Array2<f32>) -> Array2<f32> {
        let epsilon = 1e-12;
        (pred - label) / ((pred * (1. - pred)) + epsilon)
    }
}
