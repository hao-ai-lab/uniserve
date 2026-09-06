//! Server-facing client for request submission and engine health control.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod error;
/// Generation request lowering and submission support.
pub mod generation;
mod in_process;
/// Media request submission values.
pub mod media;
/// Engine-statistics publication to serving metrics.
pub mod metrics;

pub use error::{Error, Result};
pub use in_process::EngineClient;
pub use media::MediaSubmission;
pub use uniserve_engine::EventRx;
pub use uniserve_engine::StreamCancelCause;

impl EngineClient {
    /// Returns an immutable snapshot of engine identity and capacity.
    pub fn snapshot(&self) -> EngineSnapshot {
        EngineSnapshot {
            model_name: self.model_name().to_string(),
            engine_count: self.engine_count(),
            max_model_len: self.max_model_len(),
            model_dtype: self.model_dtype(),
            generation_limits: self.generation_limits(),
            components: self.component_deployment(),
            sampling_controls: if self.supports_token_sampling() {
                crate::serving::ServedSamplingControl::ALL.to_vec()
            } else {
                Vec::new()
            },
        }
    }

    /// Cancels one request at the consumer's acknowledged output prefix.
    pub async fn cancel_request(&self, request_id: &str) -> Result<()> {
        self.cancel(std::iter::once(request_id)).await
    }

    /// Aborts one request immediately.
    pub async fn abort_request(&self, request_id: &str) -> Result<()> {
        self.abort(std::iter::once(request_id)).await
    }
}

/// Engine health and limits used while binding a served model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineSnapshot {
    /// Model identifier loaded by the engine.
    pub model_name: String,
    /// Number of engine instances represented by the client.
    pub engine_count: usize,
    /// Maximum supported model context length in tokens.
    pub max_model_len: u32,
    /// Numeric data type used by model execution.
    pub model_dtype: uniserve_core::ModelDtype,
    /// Runtime generation features and resource limits.
    pub generation_limits: uniserve_core::GenerationLimits,
    /// Finalized component geometry reported by the running workers.
    pub components: std::collections::BTreeMap<String, uniserve_core::ComponentDeployConfig>,
    /// Sampling controls supported by this engine.
    pub sampling_controls: Vec<crate::serving::ServedSamplingControl>,
}
