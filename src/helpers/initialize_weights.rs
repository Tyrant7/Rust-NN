use ndarray::{Array, Dimension, IntoDimension};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

pub enum SeedMode {
    Seeded(u64),
    Random,
}

impl SeedMode {
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

pub fn kaiming_normal<D: IntoDimension>(dims: D, use_dim: usize, seed_mode: SeedMode) -> Array<f32, <D as IntoDimension>::Dim> {
    let dims = dims.into_dimension();
    let std_dev = (2. / dims[use_dim] as f64).sqrt();
    let dist = Normal::new(0., std_dev).expect("Unable to create distribution");
    let mut rng = seed_mode.rng();
    Array::from_shape_fn(dims, |_| dist.sample(&mut rng) as f32)
}