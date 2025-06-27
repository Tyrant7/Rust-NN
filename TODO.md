# TODO

## Currently working on:

-   More safety assertions throughout codebase, especially within layers
-   Full docstrings complete with examples for entire API (should definitely include expected shapes to any methods which take them)

## Next up:

-   Rename `main.rs` to `lib.rs` and do whatever else needs to be done to convert from
    an application to a library
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
-   Add more debug features
    -   Pretty layer printouts
    -   Complete random seeding, too
    -   Additional logging for debug mode
-   More loss functions (MAE, Huber Loss, etc.)
-   More activation functions (leaky ReLU, Tanh, etc.)
-   Better graphing features
    -   Line charts, naming the charts, choosing the path, etc.

## Fixes

-   Adapt quick convolution for 1D
