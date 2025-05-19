use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4};

pub fn pad_1d(input: &ArrayView1<f32>, padding: usize) -> Array1<f32> {
    assert!(padding > 0);

    let mut padded = Array1::zeros(input.dim() + padding * 2);
    padded
        .slice_mut(s![padding..input.dim() + padding])
        .assign(input);
    padded
}

pub fn convolve1d(input: ArrayView1<f32>, kernel: ArrayView1<f32>, output_size: usize, stride: usize) -> Array1<f32> {
    let mut output = Array1::zeros(output_size);
    let windows = input.windows_with_stride(kernel.dim(), stride);
    for (i, window) in windows.into_iter().enumerate() {
        output[i] += (&window * &kernel).sum();
    }
    output
}

/// padding: (height, width)
pub fn pad_2d(input: &ArrayView2<f32>, padding: (usize, usize)) -> Array2<f32> {
    let dim = input.dim();
    let mut padded = Array2::zeros(
        (dim.0 + padding.0 * 2, dim.1 + padding.1 * 2)
    );
    padded.slice_mut(
        s![(padding.0)..dim.0 + padding.0, (padding.1)..dim.1 + padding.1]
    ).assign(input);
    padded
}

/// output_size and stride: (height, width)
pub fn convolve2d(input: ArrayView2<f32>, kernel: ArrayView2<f32>, output_size: (usize, usize), stride: (usize, usize)) -> Array2<f32> {
    let mut output = Array2::zeros(output_size);
    let windows = input.windows_with_stride(kernel.dim(), stride);
    for (i, window) in windows.into_iter().enumerate() {
        let (x, y) = (i % output_size.1, i / output_size.1); 
        output[[y, x]] += (&window * &kernel).sum();
    }
    output
}

pub fn pad_3d(input: &ArrayView3<f32>, padding: (usize, usize, usize)) -> Array3<f32> {
    let dim = input.dim();
    let mut padded = Array3::zeros(
        (dim.0 + padding.0 * 2, dim.1 + padding.1 * 2, dim.2 + padding.2 * 2)
    );
    padded.slice_mut(
        s![(padding.0)..dim.0 + padding.0, (padding.1)..dim.1 + padding.1, (padding.2)..dim.2 + padding.2]
    ).assign(input);
    padded
}

pub fn pad_4d(input: &ArrayView4<f32>, padding: (usize, usize, usize, usize)) -> Array4<f32> {
    let dim = input.dim();
    let mut padded = Array4::zeros(
        (dim.0 + padding.0 * 2, dim.1 + padding.1 * 2, dim.2 + padding.2 * 2, dim.3 + padding.3 * 2)
    );
    padded.slice_mut(
        s![(padding.0)..dim.0 + padding.0, (padding.1)..dim.1 + padding.1, (padding.2)..dim.2 + padding.2, (padding.3)..dim.3 + padding.3]
    ).assign(input);
    padded
}