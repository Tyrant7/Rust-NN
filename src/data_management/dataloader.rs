use ndarray::{stack, Array, Array1, Axis, Data, Dimension};
use rand::seq::SliceRandom;

pub struct DataLoader<XType, XDim, Y> 
{
    dataset: Vec<(Array<XType, XDim>, Y)>,
    current_data: Vec<(Array<XType, XDim>, Y)>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool, 
}

impl<XType, XDim, Y> DataLoader<XType, XDim, Y> 
where 
    XType: Clone,
    XDim: Clone + Dimension,
    Y: Clone,
{
    pub fn new(
        dataset: Vec<(Array<XType, XDim>, Y)>, 
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

impl<XType, XDim, Y> Iterator for DataLoader<XType, XDim, Y> 
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

// TODO: Tests
// full batch
// incomplete batch dropping
// incomplete batch usage
// when dataset is smaller than one batch -> both drop and without