use ndarray::Array2;

pub trait LossFunction {
    fn original(pred: &Array2<f32>, label: &Array2<f32>) -> f32;
    fn derivative(pred: &Array2<f32>, label: &Array2<f32>) -> Array2<f32>;
}

mod binary_cross_entropy_loss;
use binary_cross_entropy_loss::BCELoss;
