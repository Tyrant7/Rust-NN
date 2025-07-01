use std::ops::Range;

use ndarray::{
    Array, Array1, ArrayView, ArrayViewMut, Axis, Data, Dimension, RawDataClone, RemoveAxis, Slice,
};
use rand::{seq::SliceRandom, Rng};

/// Represents a data augmentation technique to apply to input array during training.
///
/// Each variant specifics a different augmentation strategy and its parameters.
pub enum DataAugmentation<A> {
    /// Flips the array along the specified axis with a given probability.
    ///
    /// # Parameters
    /// - `probability`: Probability of applying the flip (range: 0.0 to 1.0).
    /// - `axis`: Axis along which to flip the data.
    Flip(f32, Axis),

    /// Translates the array along the specific axis by a random offset within a given range.
    ///
    /// # Parameters
    /// - `probability`: Probability of applying the translation (range: 0.0 to 1.0).
    /// - `axis`: Axis along which to shift.
    /// - `min_offset`, `max_offset`: Bounds for randomly sampled shift amount (inclusive). A negative number denotes a shift left,
    ///   and a positive denotes a shift right. Note that if the range included zero, it is possible for no shift to occur.
    Translate(f32, Axis, i32, i32),

    /// Randomly replaces pixel with either `salt` or `pepper` values (50/50 odds for either).
    /// Typically, a low and a high value are chosen.
    ///
    /// # Parameters
    /// - `probability`: Fraction of pixels to be affected (e.g., 0.05 means 5% of pixels will be replaced with either a `salt` or `pepper` value)
    /// - `salt`, `pepper`: The values used for "salt" and "pepper" pixels respectively; typically 1.0 and 0.0.
    SaltAndPepperNoise(f32, A, A),
    /*
    GaussianNoise(f32), // Gaussian distribution
    SpeckleNoise(f32), // Random multiplicative
    PoissonNoise(f32), // Poisson distribution
    */
}

impl<A> DataAugmentation<A> {
    /// Applies this data augmentation to `data` in place.
    pub fn apply_in_place<D>(&self, data: &mut Array<A, D>)
    where
        A: Default + Clone + Copy,
        D: Dimension + RemoveAxis,
    {
        let mut rng = rand::rng();
        match *self {
            Self::Flip(probability, axis) => {
                debug_assert!(
                    (0. ..=1.).contains(&probability),
                    "Invalid probability value provided: {probability}"
                );

                if rng.random::<f32>() <= probability {
                    data.invert_axis(axis);
                }
            }
            Self::Translate(probability, axis, min_offset, max_offset) => {
                debug_assert!(
                    (0. ..=1.).contains(&probability),
                    "Invalid probability value provided: {probability}"
                );

                if rng.random::<f32>() <= probability {
                    let offset = rng.random_range(min_offset..=max_offset) as isize;
                    let mut result = Array::from_elem(data.dim(), A::default());
                    let len = data.len_of(axis) as isize;
                    if offset > 0 {
                        // Shift right: copy from [0..len+offset] to [-offset..len]
                        let copy_len = len - offset;
                        if copy_len > 0 {
                            let src = data.slice_axis(axis, Slice::new(0, Some(copy_len), 1));
                            let mut dst =
                                result.slice_axis_mut(axis, Slice::new(offset, Some(len), 1));
                            dst.assign(&src);
                        }
                    } else if offset < 0 {
                        // Shift left: copy from [offset..len] to [0..len-offset]
                        let copy_len = len + offset;
                        if copy_len > 0 {
                            let src = data.slice_axis(axis, Slice::new(-offset, Some(len), 1));
                            let mut dst =
                                result.slice_axis_mut(axis, Slice::new(0, Some(copy_len), 1));
                            dst.assign(&src);
                        }
                    } else {
                        // offset == 0 -> no change
                        result.assign(data);
                    }
                    data.assign(&result);
                }
            }
            Self::SaltAndPepperNoise(probability, salt, pepper) => {
                debug_assert!(
                    (0. ..=1.).contains(&probability),
                    "Invalid probability value provided: {probability}"
                );

                // Since iterating over all of the indices is slow, we'll instead
                // determine what fraction of the image should have added noise,
                // then iterate over that many random positions and add noise that way
                let item_count = data.len();
                let noisy_items = (item_count as f32 * probability) as usize;

                let mut indices: Vec<usize> = (0..item_count).collect();
                indices.shuffle(&mut rng);

                let mut flat_data = data.view_mut().into_shape_with_order(item_count).unwrap();
                for &ind in indices.iter().take(noisy_items) {
                    flat_data[ind] = if rng.random_bool(0.5) { salt } else { pepper };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3};

    use super::*;

    #[test]
    fn flip() {
        let augmentation = DataAugmentation::Flip(1., Axis(0));
        let mut data = Array1::from_vec(vec![0., 0., 1.]);
        augmentation.apply_in_place(&mut data);

        let target = Array1::from_vec(vec![1., 0., 0.]);
        assert_eq!(data, target);
    }

    #[test]
    fn flip_second_axis() {
        let augmentation_1 = DataAugmentation::Flip(1., Axis(1));
        let augmentation_2 = DataAugmentation::Flip(1., Axis(0));
        let mut data = Array2::from_shape_vec((2, 2), vec![0., 1., 0., 0.]).unwrap();
        augmentation_1.apply_in_place(&mut data);
        augmentation_2.apply_in_place(&mut data);

        let target = Array2::from_shape_vec((2, 2), vec![0., 0., 1., 0.]).unwrap();
        assert_eq!(data, target);
    }

    #[test]
    fn noise() {
        let augmentation = DataAugmentation::SaltAndPepperNoise(0.5, 2, 0);
        let mut data = Array3::<u32>::from_elem((1, 2, 3), 1);
        augmentation.apply_in_place(&mut data);

        let mut n_different = 0;
        for point in data {
            if point != 1 {
                n_different += 1;
            }
        }
        assert_eq!(n_different, 3);
    }

    #[test]
    fn offset_pos() {
        let augmentation = DataAugmentation::Translate(1., Axis(0), 1, 1);
        let mut data = Array1::from_vec(vec![0., 1., 1.]);
        augmentation.apply_in_place(&mut data);

        let target = Array1::from_vec(vec![0., 0., 1.]);
        assert_eq!(data, target);
    }

    #[test]
    fn offset_neg() {
        let augmentation = DataAugmentation::Translate(1., Axis(0), -2, -2);
        let mut data = Array1::from_vec(vec![0., 1., 1.]);
        augmentation.apply_in_place(&mut data);

        let target = Array1::from_vec(vec![1., 0., 0.]);
        assert_eq!(data, target);
    }

    #[test]
    fn offset_zero() {
        let augmentation = DataAugmentation::Translate(1., Axis(0), 0, 0);
        let mut data = Array1::from_vec(vec![0., 0., 1.]);
        augmentation.apply_in_place(&mut data);

        let target = Array1::from_vec(vec![0., 0., 1.]);
        assert_eq!(data, target);
    }
}
