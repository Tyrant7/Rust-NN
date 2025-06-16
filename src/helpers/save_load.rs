use std::{fs::File, io::{Read, Write}};

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

    #[test]
    fn save_load() {
        let mut network = chain!(
            Linear::new_from_rand(2, 16), 
            ReLU, 
            Linear::new_from_rand(16, 1),
            Dropout::new(0.5),
            Sigmoid,
            Flatten::new(0),
        );
        let before = format!("{:#?}", network);

        save_model_state(&network, "model.state").unwrap();
        network = load_model_state("model.state").unwrap();

        assert_eq!(format!("{:#?}", network), before);
    }
}
