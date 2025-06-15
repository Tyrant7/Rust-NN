use crate::layers::CompositeLayer;

pub fn save_model_state<M>(model: M)
where 
    M: CompositeLayer + serde::ser::Serialize,
{
    let result = serde_json::to_string_pretty(&model);
    println!("r: {:#?}", result);
}

pub fn load_model_state() {
    todo!()
}
