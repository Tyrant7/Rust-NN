use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ArrayViewMut1, ArrayViewMut2};

pub fn pad_1d(input: &ArrayView1<f32>, padding: usize) -> Array1<f32> {
    assert!(padding > 0);

    let mut padded = Array1::zeros(input.dim() + padding * 2);
    padded
        .slice_mut(s![padding..input.dim() + padding])
        .assign(input);
    padded
}

pub fn convolve1d(input: ArrayView1<f32>, kernel: ArrayView1<f32>, output: &mut ArrayViewMut1<f32>, stride: usize) {
    let i_w = input.dim();
    let k_w = kernel.dim();
    
    let output_w = (i_w - k_w) / stride + 1;
    for out in 0..output_w {
        let mut acc = 0.0;
        for k in 0..k_w {
            let in_x = out * stride + k;
            acc += input[in_x] * kernel[k];
        }
        output[out] = acc;
    }
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
#[inline(always)]
pub fn convolve2d(input: &ArrayView2<f32>, kernel: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>, stride: (usize, usize)) {
    let (k_h, k_w) = kernel.dim();
    let (s_y, s_x) = stride;
    let (i_h, i_w) = input.dim();

    let o_h = (i_h - k_h) / s_y + 1;
    let o_w = (i_w - k_w) / s_x + 1;

    let input_ptr = input.as_ptr();
    let kernel_ptr = kernel.as_ptr();
    let output_ptr = output.as_mut_ptr();

    let input_stride_y = input.strides()[0];
    let input_stride_x = input.strides()[1];
    let output_stride_y = output.strides()[0];
    let output_stride_x = output.strides()[1];
    let kernel_stride_y = kernel.strides()[0];
    let kernel_stride_x = kernel.strides()[1];

    unsafe {
        for out_y in 0..o_h {
            let base_y = out_y * s_y;
            for out_x in 0..o_w {
                let base_x = out_x * s_x;
                let mut acc = 0.0;
                for ky in 0..k_h {
                    let k_offset_y = (ky as isize) * kernel_stride_y;
                    for kx in 0..k_w {
                        let in_y = (base_y + ky) as isize;
                        let in_x = (base_x + kx) as isize;

                        let i_offset = in_y * input_stride_y + in_x * input_stride_x;
                        let k_offset = k_offset_y + (kx as isize) * kernel_stride_x;

                        acc += *input_ptr.offset(i_offset) * *kernel_ptr.offset(k_offset);
                    }
                }

                let o_offset = (out_y as isize) * output_stride_y + (out_x as isize) * output_stride_x;
                *output_ptr.offset(o_offset) += acc;
            }
        }
    }
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