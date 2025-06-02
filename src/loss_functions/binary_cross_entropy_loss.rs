use ndarray::{Array1, Array2, ArrayD, Axis};
use super::LossFunction;

pub struct BCELoss;

impl LossFunction for BCELoss {
    fn original(pred: &Array2<f32>, label: &Array2<f32>) -> Array1<f32> {
        // To prevent log(0)
        let epsilon = 1e-12;
        let pred = pred.mapv(|x| x.clamp(epsilon, 1. - epsilon));
        let loss = -(label * pred.ln() + (1. - label) * (1. - pred).ln());
        loss.sum_axis(Axis(1))
    }

    fn derivative(pred: &Array2<f32>, label: &Array2<f32>) -> Array2<f32> {
        let epsilon = 1e-12;
        let pred = pred.mapv(|x| x.clamp(epsilon, 1. - epsilon));
        -(label / &pred) + ((1. - label) / (1. - &pred))        
    }
}
