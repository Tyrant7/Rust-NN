use ndarray::Array2;
use crate::tensor::Tensor;

use super::LossFunction;

pub struct MSELoss;

impl LossFunction for MSELoss {
    fn original(pred: &Tensor, label: &Tensor) -> f32 {
        (label - pred).pow2().sum()
    }

    fn derivative(pred: &Tensor, label: &Tensor) -> Tensor {
        // With respect to 'pred'
        -(label - pred) * 2.
    }
}
