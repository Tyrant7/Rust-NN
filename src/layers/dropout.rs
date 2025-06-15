use ndarray::{ArrayD, IxDyn};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use super::RawLayer;

// TODO: Serialization for this
#[derive(Debug, /* Serialize, Deserialize */)]
pub struct Dropout {
    rate: f32,
    rng: SmallRng,
    forward_mask: Option<ArrayD<f32>>,
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

impl RawLayer for Dropout {
    type Input = IxDyn;
    type Output = IxDyn;

    fn forward(&mut self, input: &ArrayD<f32>, train: bool) -> ArrayD<f32> {
        // Dropout layers are disable outside of train mode
        if !train {
            return input.clone();
        }
        
        let mask = input.map(|_| {
            if self.rng.random::<f32>() > self.rate {
                1.
            } else {
                0.
            }
        });
        self.forward_mask = Some(mask.clone());
        input * &mask / (1. - self.rate)
    }

    fn backward(&mut self, delta: &ArrayD<f32>, _forward_input: &ArrayD<f32>) -> ArrayD<f32> {
        let mask = self.forward_mask.as_ref().expect("No mask created during forward pass or forward never called");
        delta * mask / (1. - self.rate)
    }
}