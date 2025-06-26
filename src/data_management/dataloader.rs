use std::ops::Div;

use ndarray::{stack, Array, Array1, ArrayView, Axis, Data, Dimension};
use rand::seq::SliceRandom;

pub struct DataLoader<'a, XType, XDim, Y> 
{
    dataset: &'a [(ArrayView<'a, XType, XDim>, Y)],
    batch_size: usize,
    shuffle: bool,
    drop_last: bool, 
}

#[must_use]
pub struct DataIter<'a, XType, XDim, Y>
{
    data: Vec<(ArrayView<'a, XType, XDim>, Y)>,
    index: usize,
    batch_size: usize,
    drop_last: bool, 
}

impl<'a, XType, XDim, Y> DataLoader<'a, XType, XDim, Y> 
where 
    XType: Clone,
    XDim: Clone + Dimension,
    Y: Clone,
{
    pub fn new(
        dataset: &'a [(ArrayView<'a, XType, XDim>, Y)], 
        batch_size: usize, 
        shuffle: bool, 
        drop_last: bool
    ) -> Self {
        DataLoader { 
            dataset, 
            batch_size, 
            shuffle, 
            drop_last 
        }
    }

    pub fn iter(&self) -> DataIter<'a, XType, XDim, Y> {
        let mut data = self.dataset.to_vec();
        if self.shuffle {
            data.shuffle(&mut rand::rng());
        }
        DataIter {
            data, 
            index: 0,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
        }
    }

    pub fn len(&self) -> usize {
        if self.drop_last {
            self.dataset.len().div(self.batch_size)
        } else {
            self.dataset.len().div_ceil(self.batch_size)
        }
    }

    pub fn is_empty(&self) -> bool {
        self.dataset.len() == 0
    }
}

impl<'a, XType, XDim, Y> Iterator for DataIter<'a, XType, XDim, Y> 
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
        let batch_labels = Array1::from_shape_vec(batch.len(), batch_labels).expect("Error creating batch labels");
        
        Some((batch_data, batch_labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_batch_shuffle() {
        let data = Array1::<f32>::zeros(3);
        let dataset = (0..4).map(|i| (data.view(), i)).collect::<Vec<_>>();
        let dataloader = DataLoader::new(dataset.as_slice(), 2, true, true);
        for (x, label) in dataloader.iter() {
            assert!(x.dim() == (2, 3));
            assert!(label.dim() == 2);
        }
    }

    #[test]
    fn incomplete_batch_drop() {
        let data = Array1::<f32>::zeros(3);
        let dataset = &(0..5).map(|i| (data.view(), i)).collect::<Vec<_>>();
        let dataloader = DataLoader::new(dataset.as_slice(), 2, false, true);
        assert!(dataloader.iter().len() == 2);
        for (x, label) in dataloader.iter() {
            assert!(x.dim() == (2, 3));
            assert!(label.dim() == 2);
        }
    }
    
    #[test]
    fn incomplete_batch_use() {
        let data = Array1::<f32>::zeros(3);
        let dataset = &(0..5).map(|i| (data.view(), i)).collect::<Vec<_>>();
        let dataloader = DataLoader::new(dataset.as_slice(), 2, false, false);
        let (last_x, last_label) = dataloader.iter().last().unwrap();     
        assert!(dataloader.iter().len() == 3);
        assert!(last_x.dim() == (1, 3));
        assert!(last_label.dim() == 1);
    }

    #[test]
    fn smaller_than_batch() {
        let data = Array1::<f32>::zeros(3);
        let dataset = &(0..2).map(|i| (data.view(), i)).collect::<Vec<_>>();
        let dataloader = DataLoader::new(dataset.as_slice(), 3, false, true);
        assert!(dataloader.iter().next().is_none());
    }
}