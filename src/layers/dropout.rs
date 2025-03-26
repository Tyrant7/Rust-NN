use rand::{rngs::SmallRng, Rng, SeedableRng};
use ndarray::Array2;

use super::Layer;

pub struct Dropout {
    rate: f32,
    rng: SmallRng,
    forward_mask: Option<Array2<f32>>,
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
    type State = Array2<f32>;

    fn forward(&mut self, input: &Array2<f32>, train: bool) -> (Array2<f32>, Self::State) {
        // Dropout layers are disable outside of train mode
        if !train {
            return (input.clone(), Array2::from_shape_simple_fn(input.raw_dim(), || 1.))
        }
        
        let mask = input.map(|_| {
            if self.rng.random::<f32>() > self.rate {
                1.
            } else {
                0.
            }
        });
        (input * &mask / (1. - self.rate), mask.clone())
    }

    fn backward(&mut self, delta: &Array2<f32>, mask: Self::State) -> Array2<f32> {
        delta * mask / (1. - self.rate)
    }
}
