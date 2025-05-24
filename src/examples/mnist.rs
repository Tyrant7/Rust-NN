use std::env;
use std::fs;

use ndarray::Array4;
use ndarray::Axis;
use ndarray::{s, Array1, Array3};

use crate::chain;
use crate::layers::CompositeLayer;
use crate::layers::Convolutional1D;
use crate::layers::Convolutional2D;
use crate::layers::Flatten;
use crate::layers::Linear;
use crate::layers::MaxPool2D;
use crate::layers::ReLU;

use crate::Chain;
use crate::Tracked;

#[allow(unused)]

// Dataset taken from: https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download

pub fn run() {
    let train_data_path = "data/t10k-images.idx3-ubyte";
    let train_labels_path = "data/t10k-labels.idx1-ubyte";
    let (train_data, train_labels) = read_data(train_data_path, train_labels_path);

    let test_data_path = "data/train-images.idx3-ubyte";
    let test_labels_path = "data/train-labels.idx1-ubyte";
    let (test_data, test_labels) = read_data(test_data_path, test_labels_path);

    // Example usage of the library solving the MNIST handwritten digit dataset
    let mut network = chain!(
        // batch, 1, 28, 28
        Convolutional2D::new_from_rand(1, 32, (3, 3), true, (1, 1), (1, 1)),
        ReLU,
        // batch, 5, 28, 28
        MaxPool2D::new((2, 2)),
        // batch, 32, 14, 14
        Convolutional2D::new_from_rand(32, 64, (3, 3), true, (1, 1), (1, 1)),
        ReLU,
        // batch, 64, 14, 14
        MaxPool2D::new((2, 2)),
        // batch, 64, 7, 7
        Flatten::new(2),
        // batch, 64, 7*7=49
        Flatten::new(1),
        // batch, 64*7*7=3136
        Linear::new_from_rand(3136, 128),
        ReLU,
        // batch, 128
        Linear::new_from_rand(128, 10),
        // batch, 10
    );

    // shape: (28, 28)
    let sample = train_data.slice(s![0, .., ..]).to_owned();
    // shape: (batch, 1, 28, 28)
    let expanded = sample.insert_axis(Axis(0)).insert_axis(Axis(0));
    let expanded_f32 = expanded.map(|v| *v as f32 / 255.);
    let output = network.forward(&expanded_f32, false);

    println!("output: {output}");

    // let mut optimizer = SGD::new(&network.get_learnable_parameters(), 0.01, 0.9);
    // let epochs = 10000;

    // let mut avg_costs = Vec::new();
    // let mut max_costs = Vec::new();

    // let time = std::time::Instant::now();

    // for epc in 0..epochs {
    //     let mut avg_cost = 0.;
    //     let mut max_cost = 0.;

    //     /* thread::spawn(|| { */
    //     // Iterate over our entire dataset to collect gradients before applying them
    //     for (x, label) in data.iter() {
    //         let x = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
    //         let label = Array2::from_shape_fn((1, 1), |_| *label).into_dyn();

    //         let pred = network.forward(&x, true);

    //         let cost = MSELoss::original(&pred, &label);
    //         avg_cost += cost;
    //         max_cost = cost.max(max_cost);

    //         // Back propagation
    //         network.backward(&MSELoss::derivative(&pred, &label));
    //     }
    //     /* }); */

    //     avg_costs.push(avg_cost);
    //     max_costs.push(max_cost);

    //     // Gradient application
    //     optimizer.step(&mut network.get_learnable_parameters(), data.len());

    //     // Zero gradients before next epoch
    //     optimizer.zero_gradients(&mut network.get_learnable_parameters());

    //     println!("Epoch {} avg cost: {}", epc + 1, avg_cost / data.len() as f32);
    // }

    // println!("{}", format!("Completed training in {} seconds", time.elapsed().as_secs()).green());

    // println!("Generating costs chart");
    // let _ = graphs::costs_candle(&avg_costs, &max_costs);
}

fn read_data(data_path: &str, labels_path: &str) -> (Array3<u8>, Array1<u8>) {
    let contents = fs::read(data_path)
        .expect("Should have been able to read the file");
    let mut contents = contents.into_iter();

    // Explanation derived from: https://medium.com/theconsole/do-you-really-know-how-mnist-is-stored-600d69455937

    // MNIST dataset format:
    // magic number
    // size in dimension 0
    // size in dimension 1
    // ...
    // size in dimension N
    // data

    // The magic number is a 4 byte integer in big endian with its first two bytes
    // set as 0 and the other two bytes describing:
    // - The basic data type used (third byte)
    // - The number of dimensions of the stored arrays (fourth byte)

    // The basic data type used (third byte) can be as follows:
    // 0x08: unsigned byte
    // 0x09: signed byte
    // 0x0B: short (2 bytes)
    // 0x0C: int (4 bytes)
    // 0x0D: float (4 bytes)
    // 0x0E: double (8 bytes)

    // The sizes for dimensions are 32 bit integers in big endian format

    let zero_bytes = (
        contents.next().expect("Missing file metadata"), 
        contents.next().expect("Missing file metadata")
    );
    assert!(zero_bytes.0 == 0 && zero_bytes.1 == 0, "First 2 bytes in file should be zero");

    let datatype = contents.next().expect("File missing datatype");
    assert!(datatype == 0x08, "datatype should be u8 for MNIST");

    let ndims = contents.next().expect("File missing dimensions");
    assert!(ndims == 3, "ndims should be 3 for MNIST");

    let mut dims = Vec::new();
    for _ in 0..ndims {
        let dim_bytes = (0..4)
            .map(|_| contents.next().expect("Missing dim byte"))
            .collect::<Vec<u8>>();

        let dim = i32::from_be_bytes(dim_bytes[0..4].try_into().unwrap());
        dims.push(dim as usize);
    }

    let images = Array3::from_shape_vec((dims[0], dims[1], dims[2]), contents.collect())
        .expect("Shape mismatch");
    
    
    // Read back labels too
    
    let contents = fs::read(labels_path)
        .expect("Should have been able to read the file");
    let mut contents = contents.into_iter();

    let zero_bytes = (
        contents.next().expect("Missing file metadata"), 
        contents.next().expect("Missing file metadata")
    );
    assert!(zero_bytes.0 == 0 && zero_bytes.1 == 0, "First 2 bytes in file should be zero");

    let datatype = contents.next().expect("File missing datatype");
    assert!(datatype == 0x08, "datatype should be u8 for MNIST labels");

    let ndims = contents.next().expect("File missing dimensions");
    assert!(ndims == 1, "ndims should be 1 for MNIST labels");

    let dim_bytes = (0..4)
        .map(|_| contents.next().expect("Missing dim byte"))
        .collect::<Vec<u8>>();
    let dim = i32::from_be_bytes(dim_bytes[0..4].try_into().unwrap()) as usize;

    let labels = Array1::from_shape_vec(dim, contents.collect())
        .expect("Shape mismatch");
    
    (images, labels)
}