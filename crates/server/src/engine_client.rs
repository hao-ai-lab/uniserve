//! Server-facing client for request submission and engine health control.
//!
//! `EngineClient` owns the in-process `uniserve_engine::EngineCore`s, one per
//! data-parallel replica, routes every request to one of them, and owns the
//! `requests::RequestRegistry` that maps external request identifiers to
//! engine `RequestId`s and tracks each request's lifecycle state. The serving
//! runtime (`serving::ServingRuntime`) registers, submits, cancels, and aborts
//! requests through this client and never talks to the engine core directly;
//! it reads lifecycle state and waits for control commands on the client's
//! `requests` registry itself.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod error;
/// Generation request lowering and submission support.
pub mod generation;
mod in_process;
/// Export engine statistics as serving metrics.
pub mod metrics;
pub(crate) mod requests;

pub use error::{Error, Result};
pub use in_process::EngineClient;
pub use uniserve_engine::EventRx;
pub use uniserve_engine::StreamCancelCause;

impl EngineClient {
    /// Returns the sampling controls this engine exposes to served requests:
    /// every control when the worker supports token sampling, none otherwise.
    pub fn served_sampling_controls(&self) -> Vec<crate::serving::ServedSamplingControl> {
        if self.supports_token_sampling() {
            crate::serving::ServedSamplingControl::ALL.to_vec()
        } else {
            Vec::new()
        }
    }

    /// Cancels one request at the consumer's acknowledged output prefix.
    ///
    /// See `EngineClient::cancel` for how an unsubmitted or unknown request is
    /// handled.
    pub async fn cancel_request(&self, request_id: &str) -> Result<()> {
        self.cancel(std::iter::once(request_id)).await
    }

    /// Aborts one request immediately.
    ///
    /// See `EngineClient::abort` for how an unsubmitted or unknown request is
    /// handled.
    pub async fn abort_request(&self, request_id: &str) -> Result<()> {
        self.abort(std::iter::once(request_id)).await
    }
}
