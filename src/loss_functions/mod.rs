use ndarray::ArrayD;

pub trait LossFunction {
    fn original(pred: &ArrayD<f32>, label: &ArrayD<f32>) -> f32;
    fn derivative(pred: &ArrayD<f32>, label: &ArrayD<f32>) -> ArrayD<f32>;
}

pub mod binary_cross_entropy_loss;
pub use binary_cross_entropy_loss::BCELoss;

pub mod mean_squared_error_loss;
pub use mean_squared_error_loss::MSELoss;