use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4};

pub fn pad_1d(input: &ArrayView1<f32>, padding: usize) -> Array1<f32> {
    assert!(padding > 0);

    let mut padded = Array1::zeros(input.dim() + padding * 2);
    padded
        .slice_mut(s![padding..input.dim() + padding])
        .assign(input);
    padded
}

pub fn convolve1d(input: ArrayView1<f32>, kernel: ArrayView1<f32>, stride: usize) -> Array1<f32> {
    let i_w = input.dim();
    let k_w = kernel.dim();
    
    let output_w = (i_w - k_w) / stride + 1;
    let mut output = Array1::zeros(output_w);
    for out in 0..output_w {
        let mut acc = 0.0;
        for k in 0..k_w {
            let in_x = out * stride + k;
            acc += input[in_x] * kernel[k];
        }
        output[out] = acc;
    }

    output
}

pub fn crop_1d(input: &ArrayView1<f32>, crop: usize) -> Array1<f32> {
    let left = crop / 2;
    let right = crop - left;
    input.slice(s![left..input.dim() - right]).to_owned()
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
pub fn convolve2d(input: &ArrayView2<f32>, kernel: &ArrayView2<f32>, stride: (usize, usize)) -> Array2<f32> {
    let (i_h, i_w) = input.dim();
    let (k_h, k_w) = kernel.dim();
    let (s_y, s_x) = stride;
    
    let output_h = (i_h - k_h) / s_y + 1;
    let output_w = (i_w - k_w) / s_x + 1;
    let mut output = Array2::zeros((output_h, output_w));
    for out_y in 0..output_h {
        for out_x in 0..output_w {
            let mut acc = 0.0;
            for ky in 0..k_h {
                for kx in 0..k_w {
                    let in_y = out_y * s_y + ky;
                    let in_x = out_x * s_x + kx;
                    acc += input[(in_y, in_x)] * kernel[(ky, kx)];
                }
            }
            output[(out_y, out_x)] = acc;
        }
    }

    output
}

/// crop: (height, width)
pub fn crop_2d(input: &ArrayView2<f32>, crop: (usize, usize)) -> Array2<f32> {
    let dim = input.dim();
    let bottom = crop.0 / 2;
    let top = crop.0 - bottom;
    let left = crop.1 / 2;
    let right = crop.1 - left;
    input.slice(s![bottom..dim.0 - top, left..dim.1 - right]).to_owned()
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

pub fn crop_3d(input: &ArrayView3<f32>, crop: (usize, usize, usize)) -> Array3<f32> {
    let dim = input.dim();
    let bottom = crop.0 / 2;
    let top = crop.0 - bottom;
    let left = crop.1 / 2;
    let right = crop.1 - left;
    let front = crop.2 / 2;
    let back = crop.2 - front;
    input.slice(s![bottom..dim.0 - top, left..dim.1 - right, front..dim.2 - back]).to_owned()
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

pub fn crop_4d(input: &ArrayView4<f32>, crop: (usize, usize, usize, usize)) -> Array4<f32> {
    let dim = input.dim();
    let bottom = crop.0 / 2;
    let top = crop.0 - bottom;
    let left = crop.1 / 2;
    let right = crop.1 - left;
    let front = crop.2 / 2;
    let back = crop.2 - front;
    let ana = crop.3 / 2;
    let kata = crop.3 - ana;
    input.slice(s![bottom..dim.0 - top, left..dim.1 - right, front..dim.2 - back, ana..dim.3 - kata]).to_owned()
}