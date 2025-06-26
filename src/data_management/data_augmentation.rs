use ndarray::{Array, Array1, Data, Dimension};

// TODO: trash all of this. Bad idea

pub struct AugmentationBuilder<A> {
    stack: Vec<A>,
}

impl<A> Default for AugmentationBuilder<A> {
    fn default() -> Self {
        AugmentationBuilder { stack: vec![] }
    }
}

impl<A> AugmentationBuilder<A> 
where 
    A: Augmentation,
{
    /// Alias for `AugmentationBuilder::default()`
    pub fn new() -> Self {
        AugmentationBuilder::default()
    }

    pub fn push(mut self, augmentation: A) -> Self {
        self.stack.push(augmentation);
        self
    }
}

pub trait Augmentation {
    fn apply(data: Array1<f32>) -> Array1<f32>;
}