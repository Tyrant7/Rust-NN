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
use crate::data_management::dataloader::DataLoader;
use crate::graphs;
use crate::helpers::save_load;
use crate::layers::CompositeLayer;
use crate::layers::Convolutional1D;
use crate::layers::Convolutional2D;
use crate::layers::Flatten;
use crate::layers::Linear;
use crate::layers::MaxPool2D;
use crate::layers::ReLU;

use crate::layers::Sigmoid;
use crate::loss_functions::CrossEntropyWithLogitsLoss;
use crate::loss_functions::LossFunction;
use crate::loss_functions::MSELoss;
use crate::optimizers::Optimizer;
use crate::optimizers::SGD;
use crate::Chain;
use crate::Tracked;


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

    let train_data_pairs = train_data.outer_iter()
        .zip(train_labels.outer_iter())
        .collect::<Vec<_>>();

    let (test_data, test_labels) = read_data(
        "data/t10k-images.idx3-ubyte", 
        "data/t10k-labels.idx1-ubyte"
    );
    let test_data = test_data.insert_axis(Axis(1)).map(|&v| v as f32 / 255.);

    let test_data_pairs = test_data.outer_iter()
        .zip(test_labels.outer_iter())
        .collect::<Vec<_>>();

    let num_classes = 10;

    let batch_size = 50;
    let train_dataloader = DataLoader::new(train_data_pairs.as_slice(), batch_size, true, true);
    let test_dataloader = DataLoader::new(test_data_pairs.as_slice(), batch_size, false, true);

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

    let mut optimizer = SGD::new(&network.get_learnable_parameters(), 0.001, 0.9);
    let epochs = 10;
    
    let mut avg_train_costs = Vec::new();
    let mut avg_train_accuracies = Vec::new();
    let train_batches = train_dataloader.len();

    let mut avg_test_costs = Vec::new();
    let mut avg_test_accuracies = Vec::new();
    let test_batches = test_dataloader.len();

    let time = std::time::Instant::now();

    for epc in 0..epochs {
        let epoch_time = std::time::Instant::now();
        let mut avg_cost = 0.;
        let mut avg_acc = 0.;

        for (i, (x, labels)) in train_dataloader.iter().enumerate() {
            let batch_time = std::time::Instant::now();

            let mini_batch_size: usize = batch_size.div_ceil(std::thread::available_parallelism().unwrap().into());
            let mini_batch_results = x.axis_chunks_iter(Axis(0), mini_batch_size)
                .zip(labels.axis_chunks_iter(Axis(0), mini_batch_size))
                .map(|d| (d, network.clone()))
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|((mini_batch, mini_batch_labels), mut network)| {
                    let mini_batch_length = mini_batch.dim().0;
                    let pred = network.forward(&mini_batch.to_owned(), true);

                    let mut label_encoded = Array2::<f32>::zeros((mini_batch_length, num_classes));
                    for (i, &label) in mini_batch_labels.iter().enumerate() {
                        label_encoded[[i, *label.into_scalar() as usize]] = 1.;
                    }

                    let cost = CrossEntropyWithLogitsLoss::original(&pred.clone(), &label_encoded);

                    // Compute accuracy for this minibatch
                    let mut mini_batch_acc = 0.;
                    for (&label, preds) in mini_batch_labels.iter().zip(pred.axis_iter(Axis(0))) {
                        if preds.argmax().unwrap() == *label.into_scalar() as usize {
                            mini_batch_acc += 1.;
                        }
                    }

                    // Back propagation
                    let back = CrossEntropyWithLogitsLoss::derivative(&pred, &label_encoded);
                    network.backward(&back);

                    let gradients = network.get_learnable_parameters()
                        .into_iter()
                        .map(|param| param.gradients.to_owned())
                        .collect::<Vec<_>>();
                    (cost, mini_batch_acc, gradients)
                })
                .reduce_with(|a, b| {
                    let grads = a.2.into_iter()
                        .zip(b.2)
                        .map(|(p1, p2)| p1 + p2)
                        .collect::<Vec<_>>();
                    (a.0 + b.0, a.1 + b.1, grads)
                })
                .unwrap();
      
            let (batch_cost, batch_acc, grads) = mini_batch_results;
            let batch_cost = batch_cost / batch_size as f32;
            let batch_acc = batch_acc / batch_size as f32;
            let grads = grads.into_iter()
                .map(|g| g / batch_size as f32)
                .collect::<Vec<_>>();

            // Average gradients from the entire minibatch into our main network
            let mut main_params = network.get_learnable_parameters();
            for (param, grad) in main_params.iter_mut().zip(grads) {
                param.gradients.assign(&grad);
            }

            println!("Batch {i:>3} | avg loss: {batch_cost:>7.6} | avg acc: {:>6.2}% | time: {:.0}ms", batch_acc * 100., batch_time.elapsed().as_millis());
            avg_cost += batch_cost;
            avg_acc += batch_acc;

            // Gradient application
            optimizer.step(&mut main_params);
            
            // Zero gradients before next epoch
            optimizer.zero_gradients(&mut main_params);
        }

        avg_cost /= train_batches as f32;
        avg_acc /= train_batches as f32;
        avg_train_costs.push(avg_cost);
        avg_train_accuracies.push(avg_acc);

        println!("{}", "-".repeat(50));
        println!("Starting test");
        println!("{}", "-".repeat(50));

        let mut avg_test_cost = 0.;
        let mut avg_test_acc = 0.;

        // TODO: Parallelize test
        for (i, (x, labels)) in test_dataloader.iter().enumerate() {
            let batch_time = std::time::Instant::now();

            let mut label_encoded = Array2::<f32>::zeros((batch_size, num_classes));
            for (i, &label) in labels.iter().enumerate() {
                label_encoded[[i, *label.into_scalar() as usize]] = 1.;
            }

            let pred = network.forward(&x, false);
            let cost = CrossEntropyWithLogitsLoss::original(&pred.clone(), &label_encoded.clone());
            avg_test_cost += cost;

            // Compute accuracy for this batch
            let mut batch_acc = 0.;
            for (&label, preds) in labels.iter().zip(pred.axis_iter(Axis(0))) {
                if preds.argmax().unwrap() == *label.into_scalar() as usize {
                    batch_acc += 1.;
                }
            }
            batch_acc /= batch_size as f32;
            avg_test_acc += batch_acc;

            println!("Batch {i:>3} | avg loss: {cost:>7.6} | avg acc: {:>6.2}% | time: {:.0}ms", batch_acc * 100., batch_time.elapsed().as_millis());
        }

        avg_test_cost /= test_batches as f32;
        avg_test_acc /= test_batches as f32;
        avg_test_costs.push(avg_test_cost);
        avg_test_accuracies.push(avg_test_acc);

        println!("{}", "-".repeat(50));
        println!("Epoch {} | avg loss: {:.6} | avg acc: {:.2}% | avg test loss: {:.6} | avg test acc: {:.2}% | time: {:.0}ms", 
            epc + 1, 
            avg_cost, 
            avg_acc * 100., 
            avg_test_cost, 
            avg_test_acc * 100., 
            epoch_time.elapsed().as_millis()
        );
        println!("{}", "-".repeat(50));
    }

    println!("{}", format!("Completed training in {} seconds", time.elapsed().as_secs()).green());

    println!("Generating costs chart");
    let _ = graphs::costs_candle(&avg_train_costs, &avg_test_costs);
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