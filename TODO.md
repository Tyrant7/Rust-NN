# TODO

## Currently working on:

-   Max pooling 2d
-   Avg pooling for 1d and 2d
-   Global pooling for max and avg for 1d and 2d

## Next up:

-   Rename `main.rs` to `lib.rs` and do whatever else needs to be done to convert from
    an application to a library
-   More safety assertions throughout codebase, especially within layers
-   Solve MNIST as a sample problem to show off some cool capability
-   CPU multithreading

## Layer types:

-   3D convolutional layers
-   Batch norm
-   RNN support
    -   LSTM support

## Other features:

-   Gradient clipping
-   Pretty layer printouts
-   Additional sample problems, harder than XOR
-   More loss functions (MAE, Huber Loss, etc.)
-   More activation functions (leaky ReLU, Tanh, etc.)
-   Helper modules for things like Softmax
-   Helpers for dataloading and data management
-   Full docstrings complete with examples for entire API (should definitely include expected shapes to any methods which take them)

## Fixes

-   None at the moment
