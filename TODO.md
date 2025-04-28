# TODO

## Currently working on:

-   Fix up tensor code to support basic ndarray methods and avoid constant calls to
    .to_arrayXd using macros, then clean up all code in the project to utilize these

## Next up:

-   Finish up and organize unit tests for currently implemented features
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
