use ndarray::{Array, Dimension, IntoDimension};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// The method of seeding for random initialization of values.
///
/// - `Seeded(u64)`: Uses a fixed seeed for reproducible results.
/// - `Random`: Uses a randomly generated seed for non-deterministic behaviour.
pub enum SeedMode {
    Seeded(u64),
    Random,
}

impl SeedMode {
    /// Creates a new `StdRng` based on the [`SeedMode`].
    pub fn rng(&self) -> StdRng {
        match self {
            SeedMode::Seeded(seed) => StdRng::seed_from_u64(*seed),
            SeedMode::Random => {
                let seed = rand::random();
                StdRng::from_seed(seed)
            }
        }
    }
}

/// Initializes a new set of weights according to Kaiming Normal initialization.
///
/// Each value is drawn from a normal distribution with a mean of 0 and a standard deviation of `sqrt(2/n)`,
/// where `n` is the number of inputs to the node (i.e., the specified dimension's size).
///
/// This is often used to initialize weights for ReLU-activated neural networks.
///
/// # Parameters
/// - `dims`: Any valid shape for an `ndarray` `Array` type.
/// - `use_dim`: The index of the dimension to use for counting the number of inputs (`n` in our formula).
/// - `seed_mode`: The method of seeding the random number generation.
pub fn kaiming_normal<D: IntoDimension>(
    dims: D,
    use_dim: usize,
    seed_mode: SeedMode,
) -> Array<f32, <D as IntoDimension>::Dim> {
    let dims = dims.into_dimension();
    let std_dev = (2. / dims[use_dim] as f64).sqrt();
    let dist = Normal::new(0., std_dev).expect("Unable to create distribution");
    let mut rng = seed_mode.rng();
    Array::from_shape_fn(dims, |_| dist.sample(&mut rng) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kaiming() {
        let seed = 123;
        let a = kaiming_normal((3, 3), 1, SeedMode::Seeded(seed));
        let b = kaiming_normal((3, 3), 1, SeedMode::Seeded(seed));
        assert_eq!(a, b);
    }
}
