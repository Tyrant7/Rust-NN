# TODO

## Currently working on:

-   Finish up convolutional unit tests for backward + stride + padding

## Next up:

-   Rename `main.rs` to `lib.rs` and do whatever else needs to be done to convert from
    an application to a library
-   More safety assertions throughout codebase, especially within layers
-   Pooling layers
    -   Min pooling
    -   Max pooling
    -   Avg pooling

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
-   CPU multithreading
-   Full docstrings complete with examples for entire API (should definitely include expected shapes to any methods which take them)
