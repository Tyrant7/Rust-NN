use std::ops::Range;

use ndarray::{Array, Array1, ArrayView, Axis, Data, Dimension, RawDataClone, RemoveAxis, Slice};
use rand::Rng;

pub enum AugmentationAction {
    Flip(f32, Axis),
    Noise(f32),
    Translate(f32, Axis, i32, i32),
}

pub struct DataAugmentation {
    actions: Vec<AugmentationAction>,
}

impl AugmentationAction {
    pub fn apply_in_place<A, D>(&self, data: &mut Array<A, D>) 
    where 
        A: Default + Clone,
        D: Dimension + RemoveAxis,
    {
        let mut rng = rand::rng();
        match *self {
            Self::Flip(temperature, axis) => {
                if rng.random::<f32>() <= temperature {
                    data.invert_axis(axis)
                }
            },
            Self::Noise(temperature) => {
                // Since iterating over all of the indices is slow, we'll instead 
                // determine what fraction of the image should have added noise,
                // then iterate over that many random indices and add noise that way
                todo!()
            },
            Self::Translate(temperature, axis, min_offset, max_offset) => {
                if rng.random::<f32>() <= temperature {
                    let offset = rng.random_range(min_offset..=max_offset) as isize;
                    let mut result = Array::from_elem(data.raw_dim(), A::default());
                    let len = data.len_of(axis) as isize;
                    if offset > 0 {
                        // Shift right: copy from [0..len+offset] to [-offset..len]
                        let copy_len = len - offset;
                        if copy_len > 0 {
                            let src = data.slice_axis(axis, Slice::new(0, Some(copy_len), 1));
                            let mut dst = result.slice_axis_mut(axis, Slice::new(offset, Some(len), 1));
                            dst.assign(&src);
                        }
                    } else if offset < 0 {
                        // Shift left: copy from [offset..len] to [0..len-offset]
                        let copy_len = len + offset;
                        if copy_len > 0 {
                            let src = data.slice_axis(axis, Slice::new(-offset, Some(len), 1));
                            let mut dst = result.slice_axis_mut(axis, Slice::new(0, Some(copy_len), 1));
                            dst.assign(&src);
                        }
                    } else {
                        // offset == 0 -> no change
                        result.assign(&data);
                    }
                    data.assign(&result);
                }
            },
        }
    }
}

impl DataAugmentation {
    pub fn new(actions: Vec<AugmentationAction>) -> Self {
        DataAugmentation { actions }
    }

    pub fn apply<A, D>(&self, data: &Array<A, D>) -> Array<A, D> 
    where 
        A: Default + Clone,
        D: Dimension + RemoveAxis,
    {
        let mut data = data.to_owned();
        for action in self.actions.iter() {
            action.apply_in_place(&mut data);
        }
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip() {
        let augmentation = DataAugmentation::new(vec![
            AugmentationAction::Flip(1., Axis(0)),
        ]);

        let data = Array1::from_vec(vec![0., 0., 1.,]);
        let augmented = augmentation.apply(&data);

        let target = Array1::from_vec(vec![1., 0., 0.,]);
        assert_eq!(augmented, target);
    }

    #[test]
    fn noise() {
        let augmentation = DataAugmentation::new(vec![
            AugmentationAction::Noise(1.),
        ]);
        todo!()
    }

    #[test]
    fn offset_pos() {
        let augmentation = DataAugmentation::new(vec![
            AugmentationAction::Translate(1., Axis(0), 1, 1),
        ]);

        let data = Array1::from_vec(vec![0., 1., 1.,]);
        let augmented = augmentation.apply(&data);

        let target = Array1::from_vec(vec![0., 0., 1.,]);
        assert_eq!(augmented, target);
    }
    
    #[test]
    fn offset_neg() {
        let augmentation = DataAugmentation::new(vec![
            AugmentationAction::Translate(1., Axis(0), -2, -2),
        ]);

        let data = Array1::from_vec(vec![0., 1., 1.,]);
        let augmented = augmentation.apply(&data);

        let target = Array1::from_vec(vec![1., 0., 0.,]);
        assert_eq!(augmented, target);
    }

    #[test]
    fn offset_zero() {
        let augmentation = DataAugmentation::new(vec![
            AugmentationAction::Translate(1., Axis(0), 0, 0),
        ]);

        let data = Array1::from_vec(vec![0., 0., 1.,]);
        let augmented = augmentation.apply(&data);

        let target = Array1::from_vec(vec![0., 0., 1.,]);
        assert_eq!(augmented, target);
    }
}