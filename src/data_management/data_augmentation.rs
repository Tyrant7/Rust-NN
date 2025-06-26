use ndarray::{Array, Array1, Axis, Data, Dimension, RawDataClone};

pub enum AugmentationAction {
    Flip(f32, Axis),
    Noise(f32),
    Offset(f32, Axis),
}

pub struct DataAugmentation {
    actions: Vec<AugmentationAction>,
}

impl AugmentationAction {
    pub fn apply<A, D>(&self, data: Array<A, D>) -> Array<A, D> 
    where 
        A: Clone,
        D: Dimension,
    {
        match self {
            Self::Flip(temperature, axis) => {
                data
            },
            Self::Noise(temperature) => {
                data
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

    pub fn apply<A, D>(&self, data: Array<A, D>) -> Array<A, D> 
    where 
        A: Clone,
        D: Dimension,
    {
        let mut data = data.clone();
        for action in self.actions.iter() {
            data = action.apply(data);
        }
        data
    }
}