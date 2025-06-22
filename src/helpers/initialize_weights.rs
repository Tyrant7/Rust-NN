use ndarray::{Array, IntoDimension};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

pub fn kaiming_normal<T: IntoDimension>(dims: T, use_dim: usize) -> Array<f32, <T as IntoDimension>::Dim> {
    let dims = dims.into_dimension();
    let std_dev = (2. / dims[use_dim] as f64).sqrt();
    let normal = Normal::new(0., std_dev).expect("Unable to create distribution");
    
    // TODO: Revert
    let mut rng = StdRng::seed_from_u64(0);
    Array::from_shape_fn(dims, |_| normal.sample(&mut rng) as f32)
}