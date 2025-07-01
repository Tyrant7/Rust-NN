#![allow(dead_code)]
#![allow(unused_imports)]
// #![warn(missing_docs)]

//! A lightweight deep learning framework for machine learning in Rust.
//!
//! This crate provides utilities for building, training, and evaluating custom neural network architectures using `ndarray`-based tensors.
//!
//! Note: This crate is in early development; APIs may change.
//!
//! ## Features
//! - A variety of layer types and activation functions which can be composed in any way using the `chain!` macro.  
//! - Utilities for data management, including `DataLoader`s for automated batching, shuffling, and augmenting training data.
//! - Built-in data augmentation strategies, easily integrated into the data pipeline.
//! - Multiple optimizers and loss functions to support backpropagation-based training.
//! - Basic model serialization and deserialization system for saving/loading models weights.
//! - Minimal charting tools to visualize training and testing loss over epochs or batches.
//!
//! ## Examples
//!
//! Examples are provided under the `examples/` directory and currently include:
//!
//! - A sample convolutional neural network (CNN) trained on the MNIST dataset.
//! - Additional examples coming soon.
//!
//! ## Getting Started
//!
//! This framework is designed to follow a PyTorch-like workflow. To get started:
//!
//! - Prepare your dataset and wrap it in a `DataLoader` for batching and shuffling.
//! - Define a model as a sequence of layers using the `chain` macro.
//! - Choose an `Optimizer` and `LossFunction` to use for training.
//! - Write your training loop.
//!     - (Convenience utilities for training loops are coming soon, and will include multithreading capabilities using Rayon.)
//!     - (For now, an example of a multithreaded training loop using Rayon can be found in `examples/mnist.rs`.)
//! - Plot training to monitor progress.
//!     - (Charting tools are a heavy work-in-progress and will continue to improve.)

mod data_management;
mod helpers;
mod layers;
mod loss_functions;
mod optimizers;

mod examples;
mod graphs;
mod profiling;

use examples::mnist;

fn main() {
    profiling::benchmarks::conv2d::run();

    // mnist::run();
}
