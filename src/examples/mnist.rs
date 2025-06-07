use std::env;
use std::fs;

use colored::Colorize;
use ndarray::Array2;
use ndarray::Array4;
use ndarray::Axis;
use ndarray::{s, Array1, Array3};
use ndarray_stats::QuantileExt;

use crate::chain;
use crate::graphs;
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

#[allow(unused)]
pub fn run() {
    let train_data_path = "data/t10k-images.idx3-ubyte";
    let train_labels_path = "data/t10k-labels.idx1-ubyte";
    let (train_data, train_labels) = read_data(train_data_path, train_labels_path);

    let test_data_path = "data/train-images.idx3-ubyte";
    let test_labels_path = "data/train-labels.idx1-ubyte";
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

    let mut optimizer = SGD::new(&network.get_learnable_parameters(), 0.03, 0.9);
    let epochs = 10;

    let mut avg_costs = Vec::new();
    let mut avg_accuracies = Vec::new();

    let batch_size = 5;
    let samples = train_data.shape()[0];

    assert!(samples % batch_size == 0, "TODO: Fill empty space with zeroes. For now will error");
    let num_batches = samples / batch_size;

    let new_shape = (num_batches, batch_size, train_data.shape()[1], train_data.shape()[2]);
    let new_label_shape = (num_batches, batch_size);

    let reshaped_train = train_data.to_shape(new_shape).unwrap();
    let reshaped_labels = train_labels.to_shape(new_label_shape).unwrap();

    let time = std::time::Instant::now();

    for epc in 0..epochs {
        let mut avg_cost = 0.;
        let mut avg_acc = 0.;

        for (i, (x, labels)) in reshaped_train.axis_iter(Axis(0)).zip(reshaped_labels.axis_iter(Axis(0))).enumerate() {
            println!("batch {i}");

            let mut label_encoded = Array2::<f32>::zeros((batch_size, num_classes));
            for (i, &label) in labels.iter().enumerate() {
                label_encoded[[i, label as usize]] = 1.;
            }

            // Go from (batch, 28, 28) to (batch, 1, 28, 28)
            let expanded = x.insert_axis(Axis(1)).map(|&v| v as f32 / 255.);

            let pred = network.forward(&expanded, true);
            let cost = CrossEntropyWithLogitsLoss::original(&pred.clone(), &label_encoded.clone());

            // The problem is here
            println!("preds: {:?}", pred);
            // println!("model: {:?}", network.inner());

            // println!("pred-0: {:?}", pred.slice(s![0, ..]));
            // println!("labels: {:?}", &label_encoded.slice(s![0, ..]));

            // println!("cavg: {:?}", cost);

            avg_cost += cost;
            
            // Compute accuracy for this batch
            let mut batch_acc = 0.;
            for (&label, preds) in labels.iter().zip(pred.axis_iter(Axis(0))) {
                if preds.argmax().unwrap() == label as usize {
                    batch_acc += 1.;
                }
            }
            batch_acc /= batch_size as f32;
            avg_acc += batch_acc;

            println!("aavg: {:.2}%", batch_acc * 100.);

            // Back propagation
            let back = CrossEntropyWithLogitsLoss::derivative(&pred, &label_encoded);

            println!("back: {:#?}", back);

            network.backward(&back);

            // Gradient application
            optimizer.step(&mut network.get_learnable_parameters());

            // Zero gradients before next epoch
            optimizer.zero_gradients(&mut network.get_learnable_parameters());
        }

        avg_cost /= num_batches as f32;
        avg_acc /= num_batches as f32;
        avg_costs.push(avg_cost);
        avg_accuracies.push(avg_acc);

        println!("Epoch {} | Avg cost: {:.6} | Avg acc: {:.2}%", epc + 1, avg_cost, avg_acc * 100.);
    }

    println!("{}", format!("Completed training in {} seconds", time.elapsed().as_secs()).green());

    println!("Generating costs chart");
    let _ = graphs::costs_candle(&avg_costs, &vec![0.; avg_costs.len()]);
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