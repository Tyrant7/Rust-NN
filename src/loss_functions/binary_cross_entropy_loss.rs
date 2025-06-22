use ndarray::{Array1, Array2, ArrayD, Axis};
use super::LossFunction;

pub struct BCELoss;

impl LossFunction for BCELoss {
    fn original(pred: &Array2<f32>, label: &Array2<f32>) -> f32 {
        // To prevent log(0)
        let epsilon = 1e-12;
        let pred = pred.mapv(|x| x.clamp(epsilon, 1. - epsilon));
        let loss = -(label * pred.ln() + (1. - label) * (1. - pred).ln());
        loss.sum_axis(Axis(1)).sum()
    }

    fn derivative(pred: &Array2<f32>, label: &Array2<f32>) -> Array2<f32> {
        let epsilon = 1e-12;
        let pred = pred.mapv(|x| x.clamp(epsilon, 1. - epsilon));
        -(label / &pred) + ((1. - label) / (1. - &pred)) / (pred.dim().0 as f32)
    }
}
