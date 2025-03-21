use ndarray::{Array2, Axis};
use rand::Rng;

fn main() {
    let mut network = [
        Layer::new_from_rand(2, 3, relu, relu_derivative),
        Layer::new_from_rand(3, 1, relu, relu_derivative),
    ];
    let mut x = Array2::from_shape_vec((1, 2), [1., 1.].to_vec()).unwrap();
    println!("Input: \n{x}");
    for layer in network.iter_mut() {
        x = layer.forward(&x);
    }
    println!("Result from forward pass: \n{x}");

    println!("\nBeginning backward pass...");
    let target = Array2::from_shape_vec((1, 1), [0.].to_vec()).unwrap();
    let mut error = target - &x;
    let mut signal = x.clone();

    let mut wgrads: Vec<Array2<f32>> = Vec::new();
    let mut bgrads: Vec<Array2<f32>> = Vec::new();
    for layer in network.iter_mut().rev() {
        let wgrad: Array2<f32>;
        let bgrad: Array2<f32>;
        (error, signal, wgrad, bgrad) = layer.backward(&error, &signal);
        wgrads.push(wgrad);
        bgrads.push(bgrad);
    }
    println!("Final gradients: ");
    println!("weights: {:?}", wgrads);
    println!("biases:  {:?}", bgrads);

    println!("wgrad shape: {:?}", wgrads.iter().map(|x| x.shape().to_vec()).collect::<Vec<_>>());
    println!("bgrad shape: {:?}", bgrads.iter().map(|x| x.shape().to_vec()).collect::<Vec<_>>());

    println!("Applying gradients!");
    // TODO
}

fn relu(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|x| x.max(0.))
}

fn relu_derivative(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|x| if x < 0. { 0. } else { 1. })
}

fn mse(targets: &Array2<f32>, actual: &Array2<f32>) -> Array2<f32> {
    (targets - actual).pow2()
}

struct Layer {
    activation: fn(Array2<f32>) -> Array2<f32>,
    activation_derivative: fn(Array2<f32>) -> Array2<f32>,
    weights: Array2<f32>,
    bias: Array2<f32>,
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
            forward_activations: None
        }
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        /*
            println!();
            println!("trans: \n{:?}", input.dot(&self.weights.t()));
            println!("bias:  \n{:?}", &self.bias);
            println!("out:   \n{:?}", input.dot(&self.weights.t()) + &self.bias);
        */
        self.forward_activations = Some(input.clone());
        (self.activation)(input.dot(&self.weights.t()) + &self.bias)
    }

    fn backward(&mut self, error: &Array2<f32>, signal: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        let forward_act = self.forward_activations.as_ref().expect("Backward called before forward");

        println!();

        println!("forward: {}", &forward_act);
        println!("weight:  {}", &self.weights);

        let delta = error * (self.activation_derivative)(forward_act.clone());

        println!("delta:   {}", &delta);

        let wgrad = delta.t().dot(&signal.to_owned());

        println!("bias:    {}", &self.bias);
        let bgrad = delta.sum_axis(Axis(0)).insert_axis(Axis(0));

        println!("error:   {}", &error);
        println!("signal:  {}", &signal);
        let new_error = self.weights.dot(&delta.t());

        println!("new err: {}", &new_error);

        println!("wgrad:   {}", &wgrad);
        println!("bgrad:   {}", &bgrad);

        (new_error, forward_act.clone(), wgrad, bgrad)
    }
}

