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

pub fn run() {
    let train_data_path = "data/train-images.idx3-ubyte";
    let train_labels_path = "data/train-labels.idx1-ubyte";
    let (train_data, train_labels) = read_data(train_data_path, train_labels_path);

    let test_data_path = "data/t10k-images.idx3-ubyte";
    let test_labels_path = "data/t10k-labels.idx1-ubyte";
    let (test_data, test_labels) = read_data(test_data_path, test_labels_path);

    let num_classes = 10;

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

    let mut optimizer = SGD::new(&network.get_learnable_parameters(), 0.001, 0.9);
    let epochs = 10;

    let mut avg_train_costs = Vec::new();
    let mut avg_train_accuracies = Vec::new();

    let mut avg_test_costs = Vec::new();
    let mut avg_test_accuracies = Vec::new();

    let batch_size = 200;
    let samples = train_data.shape()[0];

    assert!(samples % batch_size == 0, "TODO: Fill empty space with zeroes. For now will error");
    let num_batches = samples / batch_size;

    let new_shape = (num_batches, batch_size, train_data.shape()[1], train_data.shape()[2]);
    let new_label_shape = (num_batches, batch_size);

    let reshaped_train = train_data.to_shape(new_shape).unwrap();
    let reshaped_labels = train_labels.to_shape(new_label_shape).unwrap();

    let test_samples = test_data.shape()[0];
    assert!(test_samples % batch_size == 0, "TODO: Fill empty space with zeroes. For now will error");
    let num_test_batches = test_samples / batch_size;

    let new_test_shape = (num_test_batches, batch_size, test_data.shape()[1], test_data.shape()[2]);
    let new_test_label_shape = (num_test_batches, batch_size);

    let reshaped_test = test_data.to_shape(new_test_shape).unwrap();
    let reshaped_test_labels = test_labels.to_shape(new_test_label_shape).unwrap();

    let time = std::time::Instant::now();

    for epc in 0..epochs {
        let epoch_time = std::time::Instant::now();
        let mut avg_cost = 0.;
        let mut avg_acc = 0.;

        for (i, (x, labels)) in reshaped_train.axis_iter(Axis(0)).zip(reshaped_labels.axis_iter(Axis(0))).enumerate() {
            let batch_time = std::time::Instant::now();

            // Go from (batch, 28, 28) to (batch, 1, 28, 28)
            let expanded = x.insert_axis(Axis(1)).map(|&v| v as f32 / 255.);

            let mini_batch_size: usize = batch_size.div_ceil(std::thread::available_parallelism().unwrap().into());
            let mini_batch_results = expanded.axis_chunks_iter(Axis(0), mini_batch_size)
                .zip(labels.axis_chunks_iter(Axis(0), mini_batch_size))
                .map(|d| (d, network.clone()))
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|((mini_batch, mini_batch_labels), mut network)| {
                    let mini_batch_length = mini_batch.dim().0;
                    let pred = network.forward(&mini_batch.to_owned(), true);

                    let mut label_encoded = Array2::<f32>::zeros((mini_batch_length, num_classes));
                    for (i, &label) in mini_batch_labels.iter().enumerate() {
                        label_encoded[[i, label as usize]] = 1.;
                    }

                    let cost = CrossEntropyWithLogitsLoss::original(&pred.clone(), &label_encoded);

                    // Compute accuracy for this minibatch
                    let mut mini_batch_acc = 0.;
                    for (&label, preds) in mini_batch_labels.iter().zip(pred.axis_iter(Axis(0))) {
                        if preds.argmax().unwrap() == label as usize {
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

            // Gradient application
            optimizer.step(&mut main_params);
            
            // Zero gradients before next epoch
            optimizer.zero_gradients(&mut main_params);
        }

        avg_cost /= num_batches as f32;
        avg_acc /= num_batches as f32;
        avg_train_costs.push(avg_cost);
        avg_train_accuracies.push(avg_acc);

        println!("Starting test");

        let mut avg_test_cost = 0.;
        let mut avg_test_acc = 0.;

        // TODO: Parallelize test
        for (i, (x, labels)) in reshaped_test.axis_iter(Axis(0)).zip(reshaped_test_labels.axis_iter(Axis(0))).enumerate() {
            let batch_time = std::time::Instant::now();

            let mut label_encoded = Array2::<f32>::zeros((batch_size, num_classes));
            for (i, &label) in labels.iter().enumerate() {
                label_encoded[[i, label as usize]] = 1.;
            }

            // Go from (batch, 28, 28) to (batch, 1, 28, 28)
            let expanded = x.insert_axis(Axis(1)).map(|&v| v as f32 / 255.);

            let pred = network.forward(&expanded, false);
            let cost = CrossEntropyWithLogitsLoss::original(&pred.clone(), &label_encoded.clone());
            avg_test_cost += cost;

            // Compute accuracy for this batch
            let mut batch_acc = 0.;
            for (&label, preds) in labels.iter().zip(pred.axis_iter(Axis(0))) {
                if preds.argmax().unwrap() == label as usize {
                    batch_acc += 1.;
                }
            }
            batch_acc /= batch_size as f32;
            avg_test_acc += batch_acc;

            println!("Batch {i:>3} | avg loss: {cost:>7.6} | avg acc: {:>6.2}% | time: {:.0}ms", batch_acc * 100., batch_time.elapsed().as_millis());
        }

        avg_test_cost /= num_batches as f32;
        avg_test_acc /= num_batches as f32;
        avg_test_costs.push(avg_test_cost);
        avg_test_accuracies.push(avg_test_acc);

        println!("{}", "-".repeat(50));
        println!("Epoch {} | avg loss: {:.6} | avg acc: {:.2}% | avg test loss: {:.6} | avg test acc: {:.2}% | time: {:.0}ms", 
            epc + 1, 
            avg_cost, 
            avg_acc * 100., 
            avg_test_cost, 
            avg_test_acc, 
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