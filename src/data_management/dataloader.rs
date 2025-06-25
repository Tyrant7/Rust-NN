use ndarray::Array;
use rand::seq::SliceRandom;

pub struct DataLoader<XType, XDim, Y> 
{
    dataset: Vec<(Array<XType, XDim>, Y)>,
    current_data: Vec<(Array<XType, XDim>, Y)>,
    batch_size: usize,
    shuffle: bool,
    use_incomplete_batches: bool, 
}

impl<XType, XDim, Y> DataLoader<XType, XDim, Y> 
where 
    XType: Clone,
    XDim: Clone,
    Y: Clone,
{
    pub fn new(
        dataset: Vec<(Array<XType, XDim>, Y)>, 
        batch_size: usize, 
        shuffle: bool, 
        use_incomplete_batches: bool
    ) -> Self {
        let mut current_data = dataset.to_vec();
        if shuffle { 
            current_data.shuffle(&mut rand::rng())
        }
        DataLoader { 
            dataset, 
            current_data, 
            batch_size, 
            shuffle, 
            use_incomplete_batches 
        }
    }
}