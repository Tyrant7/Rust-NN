use rand::{rngs::SmallRng, Rng, SeedableRng};

use super::{Layer, Tensor};

#[derive(Debug)]
pub struct Dropout {
    rate: f32,
    rng: SmallRng,
    forward_mask: Option<Tensor>,
}

impl Dropout {
    pub fn new(rate: f32, seed: u64) -> Dropout {
        Dropout {
            rate,
            rng: SmallRng::seed_from_u64(seed),
            forward_mask: None,
        }
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Tensor, train: bool) -> Tensor {
        // Dropout layers are disable outside of train mode
        if !train {
            return input.clone();
        }
        
        let mask = input.apply(|_| {
            if self.rng.random::<f32>() > self.rate {
                1.
            } else {
                0.
            }
        });
        self.forward_mask = Some(mask.clone());
        input * &mask / (1. - self.rate)
    }

    fn backward(&mut self, delta: &Tensor, _forward_input: &Tensor) -> Tensor {
        let mask = self.forward_mask.as_ref().expect("No mask created during forward pass or forward never called");
        delta * mask / (1. - self.rate)
    }
}
