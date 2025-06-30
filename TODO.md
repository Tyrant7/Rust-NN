# TODO

## Currently working on:

-   Remaining docstrings -> conv, pooling, and chain left
-   More safety assertions throughout codebase, especially within layers
-   Fully comment example code (don't overdo it, though)

## Next up:

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
-   Residual block support
    -   Not sure how to do this yet. Many options on how they could work

## Other features:

-   Gradient clipping
-   Add more debug features
    -   Pretty layer printouts
    -   Complete random seeding, too
    -   Additional logging for debug mode
-   More loss functions (MAE, Huber Loss, etc.)
-   More activation functions (leaky ReLU, Tanh, etc.)
-   More data augmentation options (other noise types, blurring, etc.)
-   Better graphing features
    -   Line charts, naming the charts, choosing the path, etc.
-   GPU training and testing support

## For publishing:

-   "README" on GitHub explaining the library
-   Rename `main.rs` to `lib.rs` and do whatever else needs to be done to convert from
    an application to a library
-   Full docstrings complete with examples for entire API (should definitely include expected shapes to any methods which take them)
-   Format (but make sure not to reformat a lot of tests for clarity)
-   Move `examples/` directory to outside of `src/` and test everything works there
    -   This will help to ensure that all exports are working right
-   Publish to Crates.io

## Fixes

-   Adapt quick convolution for 1D
