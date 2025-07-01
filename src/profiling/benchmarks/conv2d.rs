use core::panic;
use std::env;
use std::fs;

use colored::Colorize;
use ndarray::Array;
use ndarray::Array2;
use ndarray::Array4;
use ndarray::ArrayD;
use ndarray::Axis;
use ndarray::{s, Array1, Array3};
use ndarray_stats::QuantileExt;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use crate::chain;
use crate::data_management::data_augmentation::DataAugmentation;
use crate::data_management::dataloader::DataLoader;
use crate::graphs;
use crate::helpers::save_load;
use crate::layers::Chain;
use crate::layers::CompositeLayer;
use crate::layers::Convolutional1D;
use crate::layers::Convolutional2D;
use crate::layers::Dropout;
use crate::layers::Flatten;
use crate::layers::Linear;
use crate::layers::MaxPool2D;
use crate::layers::ReLU;
use crate::layers::Tracked;

use crate::layers::Sigmoid;
use crate::loss_functions::CrossEntropyWithLogitsLoss;
use crate::loss_functions::LossFunction;
use crate::loss_functions::MSELoss;
use crate::optimizers::Optimizer;
use crate::optimizers::SGD;

// Dataset taken from: https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download

/// Example usage of the library solving the MNIST handwritten digit dataset with a test
/// accuracy of about 96%
pub fn run() {
    let (train_data, train_labels) = read_data(
        "data/train-images.idx3-ubyte",
        "data/train-labels.idx1-ubyte",
    );
    // Go from (N, 28, 28) from 0-255 u8 to (N, 1, 28, 28) from 0-1 f32
    let train_data = train_data.insert_axis(Axis(1)).map(|&v| v as f32 / 255.);

    let train_data_pairs = train_data
        .outer_iter()
        .zip(train_labels.outer_iter())
        .collect::<Vec<_>>();

    let num_classes = 10;
    let batch_size = 64;

    let train_dataloader = DataLoader::new(
        &train_data_pairs.as_slice()[..(100 * batch_size)],
        None,
        batch_size,
        true,
        true,
    );

    let network = chain!(
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
        // batch, 3136
        // Dropout::new(0.05),
        ReLU,
        // batch, 128
        Linear::new_from_rand(128, 10),
        // batch, 10
    );

    let time = std::time::Instant::now();
    for (i, (x, labels)) in train_dataloader.iter().enumerate() {
        let batch_time = std::time::Instant::now();

        let mini_batch_size: usize =
            batch_size.div_ceil(std::thread::available_parallelism().unwrap().into());
        x
            .axis_chunks_iter(Axis(0), mini_batch_size)
            .zip(labels.axis_chunks_iter(Axis(0), mini_batch_size))
            .map(|d| (d, network.clone()))
            .collect::<Vec<_>>()
            .into_par_iter()
            .for_each(|((mini_batch, mini_batch_labels), mut network)| {
                let mini_batch_length = mini_batch.dim().0;
                let pred = network.forward(&mini_batch.to_owned(), true);

                let mut label_encoded = Array2::<f32>::zeros((mini_batch_length, num_classes));
                for (i, &label) in mini_batch_labels.iter().enumerate() {
                    label_encoded[[i, *label.into_scalar() as usize]] = 1.;
                }

                // Back propagation
                let back = CrossEntropyWithLogitsLoss::derivative(&pred, &label_encoded);
                network.backward(&back);
            });
        println!(
            "Batch {i:>3} | time: {:.0}ms",
            batch_time.elapsed().as_millis()
        );
    }

    println!(
        "{}",
        format!("Completed benchmark in {} ms", time.elapsed().as_millis()).green()
    );
}

fn read_data(data_path: &str, labels_path: &str) -> (Array3<u8>, Array1<u8>) {
    let contents = fs::read(data_path).expect("Should have been able to read the file");
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
        contents.next().expect("Missing file metadata"),
    );
    assert!(
        zero_bytes.0 == 0 && zero_bytes.1 == 0,
        "First 2 bytes in file should be zero"
    );

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

    let contents = fs::read(labels_path).expect("Should have been able to read the file");
    let mut contents = contents.into_iter();

    let zero_bytes = (
        contents.next().expect("Missing file metadata"),
        contents.next().expect("Missing file metadata"),
    );
    assert!(
        zero_bytes.0 == 0 && zero_bytes.1 == 0,
        "First 2 bytes in file should be zero"
    );

    let datatype = contents.next().expect("File missing datatype");
    assert!(datatype == 0x08, "datatype should be u8 for MNIST labels");

    let ndims = contents.next().expect("File missing dimensions");
    assert!(ndims == 1, "ndims should be 1 for MNIST labels");

    let dim_bytes = (0..4)
        .map(|_| contents.next().expect("Missing dim byte"))
        .collect::<Vec<u8>>();
    let dim = i32::from_be_bytes(dim_bytes[0..4].try_into().unwrap()) as usize;

    let labels = Array1::from_shape_vec(dim, contents.collect()).expect("Shape mismatch");

    (images, labels)
}
