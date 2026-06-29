use uniserve_engine_client::protocol::lora::LoraRequest;

/// Snapshot of served model names plus the resolved dynamic LoRA adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraModelResolution {
    pub model_names: Vec<String>,
    pub lora_request: Option<LoraRequest>,
}
