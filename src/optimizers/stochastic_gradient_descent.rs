use crate::Model;
use super::Optimizer;

pub struct SGD {
    lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> SGD {
        SGD {
            lr,
        }
    }
}

impl Optimizer for SGD {
    fn step(&self, network: &mut Model) {
        network.apply_gradients(self.lr);
    }
}
