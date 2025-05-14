# TODO

## Currently working on:

-   Finish up and organize unit tests for currently implemented features
    -   Also put unit tests in files with their code, as per the Rust best practices
-   Finish Conv1D bias backprop implementation

## Next up:

-   Rename `main.rs` to `lib.rs` and do whatever else needs to be done to convert from
    an application to a library
-   Pooling layers
    -   Min pooling
    -   Max pooling
    -   Avg pooling

## Layer types:

-   2D convolutional layers
-   3D convolutional layers
-   Batch norm
-   RNN support
    -   LSTM support

## Other features:

-   Gradient clipping
-   Pretty model and layer printouts
-   Finish tests for all features and layer types
    -   Figure out how tests should actually work
-   Additional sample problems, harder than XOR
-   More loss functions (MAE, Huber Loss, etc.)
-   More activation functions (leaky ReLU, Tanh, etc.)
-   Helper modules for things like Softmax
-   Helpers for dataloading and data management
-   CPU multithreading
