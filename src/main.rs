use ndarray::{Array, Array2, Axis};
use rand::Rng;

fn main() {
    let mut network = [
        Layer::new_from_rand(2, 3, relu, relu_derivative),
        Layer::new_from_rand(3, 1, relu, relu_derivative),
    ];

    let inputs = Array2::from_shape_vec((2, 4), vec![
        0., 0.,
        1., 0.,
        0., 1.,
        1., 1.,
    ]);
    let labels = Array2::from_shape_vec((1, 4), vec![
        0., 
        1., 
        1., 
        0., 
    ]);

    let lr = 0.01;
    let epochs = 10;

    for i in 0..epochs {
        for (x, label) in inputs.iter().zip(labels.iter()) {
            println!("Input: \n{}", x);
            println!("Label: \n{}", label);

            let mut forward_signal = x.clone();
            for layer in network.iter_mut() {
                forward_signal = layer.forward(&forward_signal);
            }

            println!("Result from forward pass: \n{}", forward_signal);
            let cost = (label - x).pow2().sum();
            println!("Cost: {}", cost)

            println!("\nBeginning backward pass...");
            let mut error = label - &forward_signal;

            let mut wgrads: Vec<Array2<f32>> = Vec::new();
            let mut bgrads: Vec<Array2<f32>> = Vec::new();

            for layer in network.iter_mut().rev() {
                let wgrad: Array2<f32>;
                let bgrad: Array2<f32>;
                (error, wgrad, bgrad) = layer.backward(&error);
                wgrads.push(wgrad);
                bgrads.push(bgrad);
            }

            println!("Applying gradients!");
            for layer in network.iter_mut() {
                let w = wgrads.pop().unwrap();
                let b = bgrads.pop().unwrap();
                layer.weights += &(&w.t() * lr);
                layer.bias += &(&b.t() * lr);
            }
        }
    }
}

fn relu(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|x| x.max(0.))
}

fn relu_derivative(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|x| if x < 0. { 0. } else { 1. })
}

struct Layer {
    activation: fn(Array2<f32>) -> Array2<f32>,
    activation_derivative: fn(Array2<f32>) -> Array2<f32>,
    weights: Array2<f32>,
    bias: Array2<f32>,
    forward_input: Option<Array2<f32>>,
    forward_activations: Option<Array2<f32>>,
}

impl Layer {
    fn new_from_rand(
        inputs: usize, 
        outputs: usize, 
        activation: fn(Array2<f32>) -> Array2<f32>, 
        activation_derivative: fn(Array2<f32>) -> Array2<f32>
    ) -> Layer {
        let mut rng = rand::rng();
        let weights = Array2::from_shape_fn((outputs, inputs), |(_i, _j)| rng.random_range((0.)..1.));
        let bias = Array2::from_shape_fn((1, outputs), |(_i, _j)| rng.random_range((0.)..1.));
        Layer {
            activation,
            activation_derivative,
            weights,
            bias,
            forward_input: None,
            forward_activations: None,
        }
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        /*
            println!();
            println!("trans: \n{:?}", input.dot(&self.weights.t()));
            println!("bias:  \n{:?}", &self.bias);
            println!("out:   \n{:?}", input.dot(&self.weights.t()) + &self.bias);
        */
        self.forward_input = Some(input.clone());
        let activation = (self.activation)(input.dot(&self.weights.t()) + &self.bias);
        self.forward_activations = Some(activation.clone());
        activation
    }

    fn backward(&mut self, error: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {

        let forward_act = self.forward_activations.as_ref().expect("Backward called before forward");
        let forward_input = self.forward_input.as_ref().expect("Backward called before forward");


        println!();

        println!("error:   {}", &error);


        let delta = error * (self.activation_derivative)(forward_act.clone());

        println!("forward: {}", &forward_act);
        println!("weight:  {}", &self.weights);

        println!("delta:   {}", &delta);

        let wgrad = forward_input.t().dot(&delta);

        println!("bias:    {}", &self.bias);
        let bgrad = delta.sum_axis(Axis(0)).insert_axis(Axis(1));

        let new_error = delta.dot(&self.weights);

        println!("wgrad:   {}", &wgrad);
        println!("bgrad:   {}", &bgrad);

        (new_error, wgrad, bgrad)
    }
}

