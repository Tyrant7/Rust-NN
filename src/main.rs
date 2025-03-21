use ndarray::Array2;
use rand::Rng;

fn main() {
    let mut network = [
        Layer::new_from_rand(2, 3, relu, relu_derivative),
        Layer::new_from_rand(3, 1, relu, relu_derivative),
    ];
    let mut x = Array2::from_shape_vec((1, 2), [1., 1.].to_vec()).unwrap();
    println!("Input: \n{x}");
    for layer in network.iter_mut() {
        x = layer.forward(x);
    }
    println!("Result from forward pass: \n{x}");

    println!("\nBeginning backward pass...");
    let mut target = Array2::from_shape_vec((1, 1), [0.].to_vec()).unwrap();

    let mut wgrads: Vec<Array2<f32>> = Vec::new();
    let mut bgrads: Vec<Array2<f32>> = Vec::new();
    for layer in network.iter_mut().rev() {
        let (error, wgrad, bgrad) = layer.backward(target);
        target = error;

        wgrads.push(wgrad);
        bgrads.push(bgrad);
    }
    println!("Final gradients: ");
    println!("weights: {:?}", wgrads);
    println!("biases:  {:?}", bgrads);
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

    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        /*
            println!();
            println!("trans: \n{:?}", input.dot(&self.weights.t()));
            println!("bias:  \n{:?}", &self.bias);
            println!("out:   \n{:?}", input.dot(&self.weights.t()) + &self.bias);
        */
        let activation = (self.activation)(input.dot(&self.weights.t()) + &self.bias);
        self.forward_activations = Some(activation.clone());
        activation
    }

    fn backward(&mut self, target: Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let forward_act = self.forward_activations.as_mut().expect("Backward called before forward");
        println!();
        println!("target:  {}", &target);
        println!("forward: {}", &forward_act);

        let error = target - &*forward_act;

        println!("error:   {}", &error);

        let wgrad = &self.weights * &error;
        let bgrad = &self.weights * &error;

        println!("wgrad:   {}", &wgrad);
        println!("bgrad:   {}", bgrad);

        (error, wgrad, bgrad)
    }
}

