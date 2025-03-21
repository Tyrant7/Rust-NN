use ndarray::Array2;
use rand::Rng;

fn main() {
    let network = [
        Layer::new_from_rand(2, 3, relu, relu_derivative),
        Layer::new_from_rand(3, 1, relu, relu_derivative),
    ];
    let mut x = Array2::from_shape_vec((2, 1), [1., 1.].to_vec()).unwrap();
    println!("Input: {x}");
    for layer in network {
        x = layer.forward(x);
    }
    println!("Result from forward pass: {x}")
}

fn relu(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|x| x.max(0.))
}

fn relu_derivative(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|x| if x < 0. { 0. } else { 1. })
}

struct Layer {
    inputs: usize,
    outputs: usize,
    activation: fn(Array2<f32>) -> Array2<f32>,
    activation_derivative: fn(Array2<f32>) -> Array2<f32>,
    weights: Array2<f32>,
    bias: Array2<f32>,
}

impl Layer {
    fn new_from_rand(
        inputs: usize, 
        outputs: usize, 
        activation: fn(Array2<f32>) -> Array2<f32>, 
        activation_derivative: fn(Array2<f32>) -> Array2<f32>
    ) -> Layer {
        let mut rng = rand::rng();
        let weights = Array2::from_shape_fn((inputs, outputs), |(_i, _j)| rng.random_range(-1.0..1.0));
        let bias = Array2::from_shape_fn((1, outputs), |(_i, _j)| rng.random_range(-1.0..1.0));
        Layer {
            inputs,
            outputs,
            activation,
            activation_derivative,
            weights,
            bias,
        }
    }

    fn forward(&self, input: Array2<f32>) -> Array2<f32> {
        (self.activation)(input * self.weights.t() + &self.bias)
    }

    /*
    fn backward(&self, input: Array2<f32>) -> Array2<f32> {

    }
     */
}

