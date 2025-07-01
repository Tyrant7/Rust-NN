use std::{
    fs::{remove_file, File},
    io::{Read, Write},
};

use serde::{de::DeserializeOwned, ser::Serialize};

/// Serializes a model's learned state (e.g. weights and biases) and writes it to a file at the given `path`.
/// This does **not** include transient state such as cached forward input or temporary masks.
///
/// # Errors
/// Returns an `Err` if serialization or file writing fails.
pub fn save_model_state<M>(model: &M, path: &str) -> std::io::Result<()>
where
    M: Serialize,
{
    let result = serde_json::to_string_pretty(model)?;
    let mut file = File::create(path)?;
    file.write_all(result.as_bytes())?;
    Ok(())
}

/// Deserializes a model's learned state from the file at `path`.
///
/// # Returns
/// The reconstructed model of type `M`.
///
/// # Errors
/// Returns an `Err` if reading or deserialization fails.
pub fn load_model_state<M>(path: &str) -> std::io::Result<M>
where
    M: DeserializeOwned,
{
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let model = serde_json::from_str(&contents)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain;
    use crate::layers::{
        Chain, CompositeLayer, Convolutional2D, Dropout, Flatten, Linear, MaxPool2D, ReLU, Tracked,
    };
    use ndarray::Array;
    use rand::distr::{Alphanumeric, SampleString};
    use rand::random;

    #[test]
    fn save_load() {
        let mut model = chain!(
            // batch, 1, 28, 28
            Convolutional2D::new_from_rand(1, 32, (3, 3), true, (1, 1), (1, 1)),
            ReLU,
            // batch, 5, 28, 28
            MaxPool2D::new((2, 2)),
            // batch, 32, 14, 14
            Convolutional2D::new_from_rand(32, 64, (3, 3), true, (1, 1), (1, 1)),
            ReLU,
            // batch, 64, 14, 14
            MaxPool2D::new((2, 2)),
            // batch, 64, 7, 7
            Flatten::new(2),
            // batch, 64, 7*7=49
            Flatten::new(1),
            // batch, 64*7*7=3136
            Linear::new_from_rand(3136, 128),
            // batch, 3136
            Dropout::new(0.05),
            ReLU,
            // batch, 128
            Linear::new_from_rand(128, 10),
            // batch, 10
        );
        let original_state = serde_json::to_string_pretty(&model).unwrap();

        let save_path = Alphanumeric.sample_string(&mut rand::rng(), 20);
        let save_path = save_path.as_str();

        save_model_state(&model, save_path).expect("Unable to save model state");
        model = load_model_state(save_path).expect("Unable to load model state");

        let loaded_state = serde_json::to_string_pretty(&model).unwrap();

        // Delete the temporary file
        std::fs::remove_file(save_path).expect("Unable to remove temporary model file");

        assert_eq!(original_state, loaded_state);
    }

    #[test]
    fn pretrained() {
        let mut model = chain!(
            // batch, 1, 28, 28
            Convolutional2D::new_from_rand(1, 32, (3, 3), true, (1, 1), (1, 1)),
            ReLU,
            // batch, 5, 28, 28
            MaxPool2D::new((2, 2)),
            // batch, 32, 14, 14
            Convolutional2D::new_from_rand(32, 64, (3, 3), true, (1, 1), (1, 1)),
            ReLU,
            // batch, 64, 14, 14
            MaxPool2D::new((2, 2)),
            // batch, 64, 7, 7
            Flatten::new(2),
            // batch, 64, 7*7=49
            Flatten::new(1),
            // batch, 64*7*7=3136
            Linear::new_from_rand(3136, 128),
            // batch, 3136
            Dropout::new(0.05),
            ReLU,
            // batch, 128
            Linear::new_from_rand(128, 10),
            // batch, 10
        );
        let data = Array::from_shape_fn((1, 1, 28, 28), |_| random::<f32>());
        let original_output = model.forward(&data, false);

        let save_path = Alphanumeric.sample_string(&mut rand::rng(), 20);
        let save_path = save_path.as_str();

        save_model_state(&model, save_path).expect("Unable to save model state");
        model = load_model_state(save_path).expect("Unable to load model state");

        let loaded_output = model.forward(&data, false);

        // Delete the temporary file
        std::fs::remove_file(save_path).expect("Unable to remove temporary model file");

        assert_eq!(original_output, loaded_output);
    }
}
