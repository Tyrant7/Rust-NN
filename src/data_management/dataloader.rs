use std::ops::Div;

use ndarray::{stack, Array, Array1, ArrayView, Axis, Data, Dimension, RemoveAxis};
use rand::seq::SliceRandom;

use crate::data_management::data_augmentation::DataAugmentation;

/// A data loader that handles batching, shuffling, and optional augmentation of input data.
///
/// This is inspired by PyTorch's `DataLoader`. It loads data from a pre-split dataset,
/// optionally applies data augmentations, and yields batches of samples during training.
pub struct DataLoader<'a, XType, XDim, Y> {
    dataset: &'a [(ArrayView<'a, XType, XDim>, Y)],
    augmentations: Option<Vec<DataAugmentation<XType>>>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

/// The iterator for data-label pairs.
///
/// This iterator is primarily created by the [`DataLoader`]'s `iter()` method, and handles
/// the actual grouping of data into batches, returning each batch as part of its `next()` method.
#[must_use]
pub struct DataIter<XType, XDim, Y> {
    data: Vec<(Array<XType, XDim>, Y)>,
    index: usize,
    batch_size: usize,
    drop_last: bool,
}

impl<'a, XType, XDim, Y> DataLoader<'a, XType, XDim, Y>
where
    XType: Clone + Copy + Default,
    XDim: Clone + Dimension + RemoveAxis,
    Y: Clone,
{
    /// Creates a new [`DataLoader`] from a dataset and configuration options.
    ///
    /// # Arguments
    ///
    /// * `dataset` - A slice of `(input, label)` pairs, where inputs are `ArrayView`s.
    /// * `augmentations` - Optional list of augmentations to apply to each input.
    /// * `batch_size` - number of samples per batch. Cannot be zero.
    /// * `shuffle` - Whether to shuffle the dataset at the start of each call to `iter()`.
    /// * `drop_last` - Whether to drop the last incomplete batch if the dataset size isn't divisible by `batch_size`.
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - `dataset` is empty.
    /// - `batch_size` is 0.
    pub fn new(
        dataset: &'a [(ArrayView<'a, XType, XDim>, Y)],
        augmentations: Option<Vec<DataAugmentation<XType>>>,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
    ) -> Self {
        assert!(
            !dataset.is_empty(),
            "Attempted to create a dataloader with an empty dataset"
        );
        assert!(
            batch_size > 0,
            "Attempted to create a dataloader with a batch size of zero"
        );
        DataLoader {
            dataset,
            augmentations,
            batch_size,
            shuffle,
            drop_last,
        }
    }

    /// Returns a [`DataIter`] that yields batches of `(data, label)` pairs over this [`DataLoader`]'s dataset.
    pub fn iter(&self) -> DataIter<XType, XDim, Y> {
        let mut data = self
            .dataset
            .iter()
            .map(|d| (d.0.to_owned(), d.1.to_owned()))
            .collect::<Vec<_>>();
        if self.shuffle {
            data.shuffle(&mut rand::rng());
        }
        if let Some(augmentations) = &self.augmentations {
            for sample in data.iter_mut() {
                for augmentation in augmentations {
                    augmentation.apply_in_place(&mut sample.0);
                }
            }
        }
        DataIter {
            data,
            index: 0,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
        }
    }

    /// Returns the number of batches in the dataset, excluding the last incomplete batch if `drop_last` is enabled on this [`DataLoader`].
    pub fn len(&self) -> usize {
        if self.drop_last {
            self.dataset.len().div(self.batch_size)
        } else {
            self.dataset.len().div_ceil(self.batch_size)
        }
    }

    /// Returns `true` if the dataset has no elements. Typically, this can not happen.
    pub fn is_empty(&self) -> bool {
        self.dataset.len() == 0
    }
}

impl<XType, XDim, Y> Iterator for DataIter<XType, XDim, Y>
where
    XType: Clone,
    XDim: Clone + Dimension,
    Y: Clone,
{
    type Item = (Array<XType, XDim::Larger>, Array1<Y>);

    fn next(&mut self) -> Option<Self::Item> {
        let remaining = self.data.len().saturating_sub(self.index);
        if remaining == 0 || (self.drop_last && remaining < self.batch_size) {
            return None;
        }

        let end = (self.index + self.batch_size).min(self.data.len());
        let batch = &self.data[self.index..end];
        self.index = end;

        let batch_data = batch.iter().map(|b| b.0.view()).collect::<Vec<_>>();
        let batch_labels = batch.iter().map(|b| b.1.clone()).collect();

        let batch_data = stack(Axis(0), &batch_data).expect("Error creating batch data");
        let batch_labels =
            Array1::from_shape_vec(batch.len(), batch_labels).expect("Error creating batch labels");

        Some((batch_data, batch_labels))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn full_batch_shuffle() {
        let data = Array1::<f32>::zeros(3);
        let dataset = (0..4).map(|i| (data.view(), i)).collect::<Vec<_>>();
        let dataloader = DataLoader::new(dataset.as_slice(), None, 2, true, true);
        for (x, label) in dataloader.iter() {
            assert!(x.dim() == (2, 3));
            assert!(label.dim() == 2);
        }
    }

    #[test]
    fn incomplete_batch_drop() {
        let data = Array1::<f32>::zeros(3);
        let dataset = &(0..5).map(|i| (data.view(), i)).collect::<Vec<_>>();
        let dataloader = DataLoader::new(dataset.as_slice(), None, 2, false, true);
        assert!(dataloader.len() == 2);
        for (x, label) in dataloader.iter() {
            assert!(x.dim() == (2, 3));
            assert!(label.dim() == 2);
        }
    }

    #[test]
    fn incomplete_batch_use() {
        let data = Array1::<f32>::zeros(3);
        let dataset = &(0..5).map(|i| (data.view(), i)).collect::<Vec<_>>();
        let dataloader = DataLoader::new(dataset.as_slice(), None, 2, false, false);
        let (last_x, last_label) = dataloader.iter().last().unwrap();
        assert!(dataloader.len() == 3);
        assert!(last_x.dim() == (1, 3));
        assert!(last_label.dim() == 1);
    }

    #[test]
    fn smaller_than_batch() {
        let data = Array1::<f32>::zeros(3);
        let dataset = &(0..2).map(|i| (data.view(), i)).collect::<Vec<_>>();
        let dataloader = DataLoader::new(dataset.as_slice(), None, 3, false, true);
        assert!(dataloader.iter().next().is_none());
    }

    #[test]
    fn full_batch_with_augment() {
        let data = Array1::<f32>::from_vec(vec![0., 0., 1.]);
        let dataset = (0..4).map(|i| (data.view(), i)).collect::<Vec<_>>();
        let augments = vec![DataAugmentation::Flip(1., Axis(0))];
        let dataloader = DataLoader::new(dataset.as_slice(), Some(augments), 2, false, true);

        let target = Array2::<f32>::from_shape_vec((2, 3), vec![1., 0., 0., 1., 0., 0.]).unwrap();
        for (x, label) in dataloader.iter() {
            assert!(x.dim() == (2, 3));
            assert!(label.dim() == 2);
            assert!(x == target);
        }
    }
}
