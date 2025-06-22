use ndarray::{Array2, Array4};

pub enum ThreadMode {
    Single,
    Multi(usize)
}

pub fn train_model<M>(model: M, data: (Array4<f32>, Array2<usize>), thread_mode: ThreadMode) {
    
}