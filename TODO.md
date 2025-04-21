# TODO

## Currently working on:

-   Finish 1D convolutional layers
    -   Unit tests
-   Stateless layers
    -   "Parameters" object which holds all parameters and gradients for a given model, then the network indexes
        into params object to retrieve specific parameters for each layer and feeds it as an input parmeter

## Next up:

-   Better layer abstraction, potentially using an enum

## Layer types:

-   2D convolutional layers
-   3D convolutional layers
-   Max pooling
-   Avg pooling
-   Min pooling
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
