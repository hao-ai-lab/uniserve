use serde_tuple::{Deserialize_tuple, Serialize_tuple};

/// Request for a LoRA adapter.
///
/// Mirrors the reference `LoRARequest`, which is a msgspec
/// `array_like=True` struct. Keep the field order aligned with Python.
///
/// `is_3d_lora_weight` is rejected loudly at the route (the
/// worker only merges 2D deltas), and `load_inplace` is honored server-side by
/// evicting the resident adapter before re-`add_lora`. `lora_int_id` is a `u64`
/// here to mirror the reference, but the engine/worker contract is `u32`; the
/// registry guards the id space at allocation.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct LoraRequest {
    pub lora_name: String,
    pub lora_int_id: u64,
    pub lora_path: String,
    #[serde(default)]
    pub load_inplace: bool,
    #[serde(default)]
    pub is_3d_lora_weight: bool,
}

impl LoraRequest {
    pub fn new(
        lora_name: String,
        lora_int_id: u64,
        lora_path: String,
        load_inplace: bool,
        is_3d_lora_weight: bool,
    ) -> Self {
        Self {
            lora_name,
            lora_int_id,
            lora_path,
            load_inplace,
            is_3d_lora_weight,
        }
    }
}
