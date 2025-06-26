use ndarray::{Array, Array1, ArrayView, Axis, Data, Dimension, RawDataClone, RemoveAxis};
use rand::Rng;

pub enum AugmentationAction {
    Flip(f32, Axis),
    Noise(f32),
    Offset(f32, Axis),
}

pub struct DataAugmentation {
    actions: Vec<AugmentationAction>,
}

impl AugmentationAction {
    pub fn apply<'a, A, D>(&self, mut data: ArrayView<'a, A, D>) -> ArrayView<'a, A, D> 
    where 
        A: Clone,
        D: 'a + Dimension + RemoveAxis,
    {
        let mut rng = rand::rng();
        match *self {
            Self::Flip(temperature, axis) => {
                if rng.random::<f32>() <= temperature {
                    data.invert_axis(axis)
                }
                data
            },
            Self::Noise(temperature) => {
                data
                // Since iterating over all of the indices is slow, we'll instead 
                // determine what fraction of the image should have added noise,
                // then iterate over that many random indices and add noise that way
            },
            Self::Offset(temperature, axis) => {
                data
            },
        }
    }
}

impl DataAugmentation {
    pub fn new(actions: Vec<AugmentationAction>) -> Self {
        DataAugmentation { actions }
    }

    pub fn apply<'a, A, D>(&self, mut data: ArrayView<'a, A, D>) -> ArrayView<'a, A, D> 
    where 
        A: Clone,
        D: 'a + Dimension + RemoveAxis,
    {
        for action in self.actions.iter() {
            data = action.apply(data);
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
        let augmented = augmentation.apply(data.view());

        let target = Array1::from_vec(vec![1., 0., 0.,]);
        assert_eq!(augmented, target);
    }
}