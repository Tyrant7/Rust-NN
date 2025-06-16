use std::{fs::{remove_file, File}, io::{Read, Write}};

use serde::{ser::Serialize, de::DeserializeOwned};

use crate::layers::CompositeLayer;

pub fn save_model_state<M>(model: &M, path: &str) -> std::io::Result<()>
where 
    M: CompositeLayer + Serialize,
{
    let result = serde_json::to_string_pretty(model)?;
    let mut file = File::create(path)?;
    file.write_all(result.as_bytes())?;
    Ok(())
}

pub fn load_model_state<M>(path: &str) -> std::io::Result<M>
where 
    M: CompositeLayer + DeserializeOwned
{
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let model = serde_json::from_str(&contents)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use crate::*;
    use crate::save_load::{save_model_state, load_model_state};
    use rand::distr::{Alphanumeric, SampleString};

    #[test]
    fn save_load() {
        let mut model = chain!(
            Linear::new_from_rand(2, 16), 
            ReLU, 
            Linear::new_from_rand(16, 1),
            Dropout::new(0.5),
            Sigmoid,
            Flatten::new(0),
        );
        let original_state = serde_json::to_string_pretty(&model).unwrap();

        let save_path = Alphanumeric.sample_string(&mut rand::rng(), 20);
        let save_path = save_path.as_str();

        save_model_state(&model, save_path)
            .expect("Unable to save model state");
        model = load_model_state(save_path)
            .expect("Unable to load model state");

        let loaded_state = serde_json::to_string_pretty(&model).unwrap();

        // Delete the temporary file
        std::fs::remove_file(save_path)
            .expect("Unable to remove temporary model file");
        
        assert_eq!(original_state, loaded_state);
    }

    // TODO: Also include conv layers in above test
    // TODO: Test pretrained model weights too
}
