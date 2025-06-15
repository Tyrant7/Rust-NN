// Raw layers are layers that actually deal with data
// All activation functions, fully connected layers, or convolutions, etc. are raw layers
pub trait RawLayer: std::fmt::Debug
{
    type Input: Dimension;
    type Output: Dimension;

    fn forward(&mut self, input: &Array<f32, Self::Input>, train: bool) -> Array<f32, Self::Output>;
    fn backward(&mut self, error: &Array<f32, Self::Output>, forward_input: &Array<f32, Self::Input>) -> Array<f32, Self::Input>;

    // Not all layers have learnable parameters
    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> { vec![] }

    // TODO: "inspect()" method to print model layers in a prettier way
}

// Composite layers only handle piping data to raw layers, and don't actually deal with the data itself
// hence why these layers does not care about the forward input in backpropagation -> they will always
// delegate error calculations
pub trait CompositeLayer: std::fmt::Debug {
    type Input: Dimension;
    type Output: Dimension;

    fn forward(&mut self, input: &Array<f32, Self::Input>, train: bool) -> Array<f32, Self::Output>;
    fn backward(&mut self, error: &Array<f32, Self::Output>) -> Array<f32, Self::Input>;

    fn get_learnable_parameters(&mut self) -> Vec<LearnableParameter> { vec![] }

    // TODO: "inspect()" method to print model layers in a prettier way
}

pub struct LearnableParameter<'a> {
    pub values: ArrayViewMutD<'a, f32>,
    pub gradients: ArrayViewMutD<'a, f32>,
}

impl<'a> LearnableParameter<'a> {
    pub fn zero_grads(&mut self) {
        self.gradients.fill(0.)
    }
}

#[derive(Debug)]
struct ParameterGroup<D>
where 
    D: Dimension
{
    pub values: Array<f32, D>,
    pub gradients: Array<f32, D>,
}

impl<D: Dimension> ParameterGroup<D>
{
    pub fn new(initial_values: Array<f32, D>) -> Self {
        let gradients = Array::zeros(initial_values.raw_dim());
        ParameterGroup {
            values: initial_values,
            gradients
        }
    }

    pub fn as_learnable_parameter(&mut self) -> LearnableParameter {
        LearnableParameter { 
            values: self.values.view_mut().into_dyn(), 
            gradients: self.gradients.view_mut().into_dyn() 
        }
    }
}

use std::vec;

use ndarray::{Array, ArrayViewMutD, Dimension, IntoDimension, OwnedRepr};

pub mod chain;
pub use chain::Chain;

pub mod tracked;
pub use tracked::Tracked;

pub mod linear;
pub use linear::Linear;

pub mod dropout;
pub use dropout::Dropout;

pub mod convolutional1d;
pub use convolutional1d::Convolutional1D;

pub mod convolutional2d;
pub use convolutional2d::Convolutional2D;

pub mod pooling;
pub use pooling::maxpool1d::MaxPool1D;
pub use pooling::maxpool2d::MaxPool2D;

pub mod flatten;
pub use flatten::Flatten;

pub mod relu;
pub use relu::ReLU;
pub mod sigmoid;
pub use sigmoid::Sigmoid;

#[cfg(test)]
pub mod tests {
    use ndarray::{Array1, Dim, IxDyn, IxDynImpl};

    use super::RawLayer;

    pub fn test_activation_fn<T>(mut activation: T, input: Vec<f32>, expected_out: Vec<f32>, error: Vec<f32>, expected_err: Vec<f32>)
    where 
        T: RawLayer<Input = Dim<IxDynImpl>, Output = Dim<IxDynImpl>>
    {
        let epsilon = 1e-5;
        
        let input = Array1::<f32>::from_shape_vec(input.len(), input).unwrap().into_dyn();
        let output = activation.forward(&input, false);

        let target = Array1::<f32>::from_shape_vec(expected_out.len(), expected_out).unwrap().into_dyn();

        for (o, t) in output.iter().zip(target.iter()) {
            let diff = (*o - *t).abs();
            assert!(diff <= epsilon);
        }

        let error = Array1::<f32>::from_shape_vec(error.len(), error).unwrap().into_dyn();
        let error_signal = activation.backward(&error, &input);

        let target_signal = Array1::<f32>::from_shape_vec(expected_err.len(), expected_err).unwrap().into_dyn();

        for (e, t) in error_signal.iter().zip(target_signal.iter()) {
            let diff = (*e - *t).abs();
            assert!(diff <= epsilon);
        }
    }
}