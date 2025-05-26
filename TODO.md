# TODO

## Currently working on:

-   Softmax: good resource [here](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
-   CrossEntropyLoss

## Next up:

-   Rename `main.rs` to `lib.rs` and do whatever else needs to be done to convert from
    an application to a library
-   More safety assertions throughout codebase, especially within layers
-   Generic IDX file reading solution

## Layer types:

-   3D convolutional layers
-   Batch norm
-   RNN support
    -   LSTM support
-   More pooling layers
    -   First, create default "pooling layer" type
    -   Then, can add different layers by defining their "choose" method
    -   Avg pooling

## Other features:

-   Gradient clipping
-   Pretty layer printouts
-   More loss functions (MAE, Huber Loss, etc.)
-   More activation functions (leaky ReLU, Tanh, etc.)
-   Full docstrings complete with examples for entire API (should definitely include expected shapes to any methods which take them)

## Fixes

-   Adapt quick convolution for 1D
