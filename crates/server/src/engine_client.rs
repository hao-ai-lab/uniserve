//! Server-facing client for request submission and engine health control.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod error;
/// Generation request lowering and submission support.
pub mod generation;
mod in_process;
/// Engine-statistics publication to serving metrics.
pub mod metrics;
pub(crate) mod requests;

pub use error::{Error, Result};
pub use in_process::EngineClient;
pub use uniserve_engine::EventRx;
pub use uniserve_engine::StreamCancelCause;

impl EngineClient {
    /// Returns the sampling controls this engine exposes to served requests.
    pub fn served_sampling_controls(&self) -> Vec<crate::serving::ServedSamplingControl> {
        if self.supports_token_sampling() {
            crate::serving::ServedSamplingControl::ALL.to_vec()
        } else {
            Vec::new()
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
