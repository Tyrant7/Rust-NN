use ndarray::{ArrayD, IxDyn};

use super::RawLayer;

#[derive(Debug)]
pub struct ReLU;

impl RawLayer for ReLU {
    type Input = IxDyn;
    type Output = IxDyn;

    fn forward(&mut self, input: &ArrayD<f32>, _train: bool) -> ArrayD<f32> {
        input.map(|x| x.max(0.))
    }

    fn backward(&mut self, error: &ArrayD<f32>, forward_z: &ArrayD<f32>) -> ArrayD<f32> {
        let activation_derivative = forward_z.map(|x| if *x <= 0. { 0. } else { 1. });
        activation_derivative * error
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array3;

    use super::*;

    #[test]
    fn test_relu() {
        let input = Array3::<f32>::from_shape_vec((2, 2, 2), vec![
            -2., -100.,
            0.,   5.,
            0.,   0.,
            1.,   200.,
        ]).unwrap().into_dyn();
        let mut relu = ReLU;
        let output = relu.forward(&input, false);

        let target = Array3::<f32>::from_shape_vec((2, 2, 2), vec![
            0., 0.,
            0., 5.,
            0., 0.,
            1., 200.,
        ]).unwrap().into_dyn();

        assert_eq!(output, target);
    }
}