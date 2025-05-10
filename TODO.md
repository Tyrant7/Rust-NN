# TODO

## Currently working on:

-   Switch back to each layer tracking its own internal forward input state, but use some sort of intermediate
    abstraction to avoid having the same error checking in each layer

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
