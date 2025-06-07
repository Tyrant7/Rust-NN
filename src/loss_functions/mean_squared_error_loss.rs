use ndarray::{Array1, Array2, ArrayD, Axis};
use super::LossFunction;

pub struct MSELoss;

impl LossFunction for MSELoss {
    fn original(preds: &Array2<f32>, labels: &Array2<f32>) -> f32 {
        let mut output = Array1::zeros(preds.dim().0);
        for (b, (pred, label)) in preds.axis_iter(Axis(0)).zip(labels.axis_iter(Axis(0))).enumerate() {
            output[b] = (&label - &pred).pow2().sum();
        }
        output.mean().unwrap()
    }

    fn derivative(pred: &Array2<f32>, label: &Array2<f32>) -> Array2<f32> {
        // With respect to 'pred'
        (pred - label) * 2. / (pred.dim().0 as f32)
    }
}
