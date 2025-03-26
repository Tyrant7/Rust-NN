use rand::{rngs::SmallRng, Rng, SeedableRng};
use ndarray::Array2;

use super::{Layer, NoForwardError};

pub struct Dropout {
    rate: f32,
    rng: SmallRng,
    forward_input: Option<Array2<f32>>,
    forward_mask: Option<Array2<f32>>,
}

impl Dropout {
    pub fn new(rate: f32, seed: u64) -> Dropout {
        Dropout {
            rate,
            rng: SmallRng::seed_from_u64(seed),
            forward_input: None,
            forward_mask: None,
        }
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Array2<f32>, train: bool) -> Array2<f32> {
        // Dropout layers are disable outside of train mode
        if !train {
            return input.clone();
        }

        self.forward_input = Some(input.clone());
        
        let mask = input.map(|_| {
            if self.rng.random::<f32>() > self.rate {
                1.
            } else {
                0.
            }
        });
        self.forward_mask = Some(mask.clone());

        input * mask / (1. - self.rate)
    }

    fn backward(&mut self, delta: &Array2<f32>) -> Result<Array2<f32>, NoForwardError> {
        let mask = match self.forward_mask.as_ref() {
            Some(m) => m,
            None => return Err(NoForwardError),
        };
        Ok(delta * mask / (1. - self.rate))
    }
}
