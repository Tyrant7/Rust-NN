use ndarray::ArrayD;
use rand::{rngs::SmallRng, Rng, SeedableRng};

use super::RawLayer;

#[derive(Debug)]
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
    type Input = ArrayD<f32>;
    type Output = ArrayD<f32>;

    fn forward(&mut self, input: &Self::Input, train: bool) -> Self::Output {
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

    fn backward(&mut self, delta: &Self::Output, _forward_input: &Self::Input) -> Self::Input {
        let mask = self.forward_mask.as_ref().expect("No mask created during forward pass or forward never called");
        delta * mask / (1. - self.rate)
    }
}
