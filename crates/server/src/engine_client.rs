//! Server-owned in-process engine client.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod error;
pub mod generation;
mod in_process;
pub mod media;
pub mod metrics;

pub use error::{Error, Result};
pub use in_process::EngineClient;
pub use media::MediaSubmission;
pub use uniserve_engine::MediaEventRx;
pub use uniserve_engine::StreamCancelCause;

impl EngineClient {
    pub fn snapshot(&self) -> EngineSnapshot {
        EngineSnapshot {
            model_name: self.model_name().to_string(),
            engine_count: self.engine_count(),
            max_model_len: self.max_model_len(),
            model_dtype: self.model_dtype(),
            generation_capabilities: self.generation_capabilities(),
            sampling_controls: if self.supports_token_sampling() {
                crate::serving::ServedSamplingControl::ALL.to_vec()
            } else {
                Vec::new()
            },
        }
    }

    pub async fn cancel_request(&self, request_id: &str) -> Result<()> {
        self.cancel(std::iter::once(request_id)).await
    }

    pub async fn abort_request(&self, request_id: &str) -> Result<()> {
        self.abort(std::iter::once(request_id)).await
    }
}

/// Engine health and capability values used while binding a served model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineSnapshot {
    pub model_name: String,
    pub engine_count: usize,
    pub max_model_len: u32,
    pub model_dtype: uniserve_core::ModelDtype,
    pub generation_capabilities: uniserve_core::GenerationRuntimeCapabilities,
    pub sampling_controls: Vec<crate::serving::ServedSamplingControl>,
}
