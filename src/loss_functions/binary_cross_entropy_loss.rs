use ndarray::Array2;
use crate::tensor::Tensor;

use super::LossFunction;

pub struct BCELoss;

impl LossFunction for BCELoss {
    fn original(pred: &Tensor, label: &Tensor) -> f32 {
        // To prevent log(0)
        let epsilon = 1e-12;
        -(label * (pred + epsilon).ln() + (1. - label) * (1. - pred + epsilon).ln()).sum()
    }

    fn derivative(pred: &Tensor, label: &Tensor) -> Tensor {
        let epsilon = 1e-12;
        (pred - label) / ((pred * (1. - pred)) + epsilon)
    }
}
