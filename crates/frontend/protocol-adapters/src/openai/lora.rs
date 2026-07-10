use uniserve_serving::AdapterSelection;

/// Snapshot of served model names plus the resolved dynamic LoRA adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraModelResolution {
    pub model_names: Vec<String>,
    pub adapter: AdapterSelection,
}

impl LoraModelResolution {
    pub fn response_model(&self) -> String {
        match &self.adapter {
            AdapterSelection::Base => self.model_names.first().cloned().unwrap_or_default(),
            AdapterSelection::Adapter { name, .. } => name.clone(),
        }
    }
}
