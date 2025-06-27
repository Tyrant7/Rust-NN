use std::env;
use std::fs;

use colored::Colorize;
use ndarray::Array2;
use ndarray::Array4;
use ndarray::Axis;
use ndarray::{s, Array1, Array3};

use crate::chain;
use crate::graphs;
use crate::layers::Chain;
use crate::layers::Tracked;
use crate::layers::CompositeLayer;
use crate::layers::Convolutional1D;
use crate::layers::Convolutional2D;
use crate::layers::Flatten;
use crate::layers::Linear;
use crate::layers::MaxPool2D;
use crate::layers::ReLU;

use crate::layers::Sigmoid;
use crate::loss_functions::LossFunction;
use crate::loss_functions::MSELoss;
use crate::optimizers::Optimizer;
use crate::optimizers::SGD;

use super::benchmark;

pub fn run() {
    let train_data_path = "data/t10k-images.idx3-ubyte";
    let train_labels_path = "data/t10k-labels.idx1-ubyte";
    let (train_data, train_labels) = read_data(train_data_path, train_labels_path);

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

    let batch_size = 50;
    let samples = train_data.shape()[0];

    assert!(samples % batch_size == 0, "TODO: Fill empty space with zeroes. For now will error");
    let num_batches = samples / batch_size;

    let new_shape = (num_batches, batch_size, train_data.shape()[1], train_data.shape()[2]);
    let new_label_shape = (num_batches, batch_size);

    let reshaped_train = train_data.to_shape(new_shape).unwrap();
    let reshaped_labels = train_labels.to_shape(new_label_shape).unwrap();

    benchmark(&mut |i| {
        println!("Iteration {i}");

        // Only do 10 batches for sampling
        for (x, labels) in reshaped_train.axis_iter(Axis(0)).zip(reshaped_labels.axis_iter(Axis(0))).take(10) {
            let mut label_encoded = Array2::<f32>::zeros((batch_size, 10));
            for (i, &label) in labels.iter().enumerate() {
                label_encoded[[i, label as usize]] = 1.;
            }

            // go from shape (28, 28) to (batch, 1, 28, 28)
            let expanded = x.insert_axis(Axis(1));
            let expanded_f32 = expanded.map(|v| *v as f32 / 255.);

            let pred = network.forward(&expanded_f32, true);
            network.backward(&MSELoss::derivative(&pred, &label_encoded));
        }
    }, 10, "forward + backward 10 batches");
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