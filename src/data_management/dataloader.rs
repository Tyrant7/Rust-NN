use ndarray::{stack, Array, Array1, ArrayView, Axis, Data, Dimension};
use rand::seq::SliceRandom;

pub struct DataLoader<'a, XType, XDim, Y> 
{
    dataset: Vec<(ArrayView<'a, XType, XDim>, Y)>,
    current_data: Vec<(ArrayView<'a, XType, XDim>, Y)>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool, 
}

impl<'a, XType, XDim, Y> DataLoader<'a, XType, XDim, Y> 
where 
    XType: Clone,
    XDim: Clone + Dimension,
    Y: Clone,
{
    pub fn new(
        dataset: Vec<(ArrayView<'a, XType, XDim>, Y)>, 
        batch_size: usize, 
        shuffle: bool, 
        drop_last: bool
    ) -> Self {
        let mut loader = DataLoader { 
            dataset, 
            current_data: vec![], 
            batch_size, 
            shuffle, 
            drop_last 
        };
        loader.setup();
        loader
    }

    fn setup(&mut self) {
        self.current_data = self.dataset.to_vec();
        if self.shuffle { 
            self.current_data.shuffle(&mut rand::rng())
        }
    }
}

impl<'a, XType, XDim, Y> Iterator for DataLoader<'a, XType, XDim, Y> 
where 
    XType: Clone,
    XDim: Clone + Dimension,
    Y: Clone, 
{
    type Item = (Array<XType, XDim::Larger>, Array1<Y>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_data.is_empty() || (self.drop_last && self.current_data.len() < self.batch_size) {
            // Reload the dataset for epoch
            self.setup();
            return None
        }
        
        let batch = self.current_data
            .drain(..self.batch_size.min(self.current_data.len()))
            .collect::<Vec<_>>();

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
        let dataset = (0..4).map(|i| (data.view(), i)).collect();
        let dataloader = DataLoader::new(dataset, 2, true, true);
        for (x, label) in dataloader {
            assert!(x.dim() == (2, 3));
            assert!(label.dim() == 2);
        }
    }

    #[test]
    fn incomplete_batch_drop() {
        let data = Array1::<f32>::zeros(3);
        let dataset = (0..5).map(|i| (data.view(), i)).collect();
        let dataloader = DataLoader::new(dataset, 2, false, true);
        for (x, label) in dataloader {
            assert!(x.dim() == (2, 3));
            assert!(label.dim() == 2);
        }
    }
    
    #[test]
    fn incomplete_batch_use() {
        let data = Array1::<f32>::zeros(3);
        let dataset = (0..5).map(|i| (data.view(), i)).collect();
        let dataloader = DataLoader::new(dataset, 2, false, false);
        let (last_x, last_label) = dataloader.last().unwrap();
        assert!(last_x.dim() == (1, 3));
        assert!(last_label.dim() == 1);
    }

    #[test]
    fn smaller_than_batch() {
        let data = Array1::<f32>::zeros(3);
        let dataset = (0..2).map(|i| (data.view(), i)).collect();
        let mut dataloader = DataLoader::new(dataset, 3, false, true);
        assert!(dataloader.next().is_none());
    }
}