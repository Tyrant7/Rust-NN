use ndarray::{ArrayD, IxDyn};

use super::RawLayer;

#[derive(Debug)]
pub struct Sigmoid;

impl Sigmoid {
    fn sigmoid(input: &ArrayD<f32>) -> ArrayD<f32> {
        input.map(|x| 1. / (1. + (-x).exp()))
    }
}

impl RawLayer for Sigmoid {
    type Input = IxDyn;
    type Output = IxDyn;

    fn forward(&mut self, input: &ArrayD<f32>, _train: bool) -> ArrayD<f32> {
        Sigmoid::sigmoid(input)
    }

    fn backward(&mut self, error: &ArrayD<f32>, forward_z: &ArrayD<f32>) -> ArrayD<f32> {
        let sig = Sigmoid::sigmoid(forward_z);
        let one_minus = sig.map(|x| 1. - x);
        let activation_derivative = sig * &one_minus;
        activation_derivative * error
    }
}

#[cfg(test)]
mod tests {
    use crate::layers::tests::test_activation_fn;

    use super::*;

    #[test]
    fn test() {
        test_activation_fn(Sigmoid, 
            vec![-1., 0., 1.,],
            vec![0.268941, 0.5, 0.731059,],
            vec![-1., 0., 1.,],
            vec![-0.196612, 0., 0.196612,]
        );
    }
}