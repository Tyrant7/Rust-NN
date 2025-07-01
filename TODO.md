# TODO

## Currently working on:

-   Optimize conv layers even more (and rewrite benchmarks)
-   "README" on GitHub explaining the library

## Next up:

-   Parallelization utilities and adapt MNIST to use these

## Layer types:

-   3D convolutional layers
-   Batch norm
-   RNN support
    -   LSTM support
-   More pooling layers
    -   First, create default "pooling layer" type
    -   Then, can add different layers by defining their "choose" method
    -   Avg pooling
-   Residual block support
    -   Not sure how to do this yet. Many options on how they could work

## Other features:

-   Generic IDX file reading solution
-   Gradient clipping
-   Add more debug features
    -   Pretty layer printouts
    -   Complete random seeding, too
    -   Additional logging for debug mode
    -   Expose more fields within layers publicly to read for debugging
-   More loss functions (MAE, Huber Loss, etc.)
-   More activation functions (leaky ReLU, Tanh, etc.)
-   More data augmentation options (other noise types, blurring, etc.)
-   Better graphing features
    -   Line charts, naming the charts, choosing the path, etc.
-   GPU training and testing support

## For publishing:

-   Rename `main.rs` to `lib.rs` and do whatever else needs to be done to convert from
    an application to a library
-   Move `examples/` directory to outside of `src/` and test everything works there
    -   This will help to ensure that all exports are working right
    -   Also comment and cleanup example code
-   Publish to Crates.io

## Fixes

-   Adapt quick convolution for 1D
