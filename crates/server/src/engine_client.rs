//! Server-owned in-process and ZMQ engine clients.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::sync::Arc;

mod client;
mod error;
pub mod generation;
mod in_process;
pub mod media;
pub mod metrics;
pub mod zmq;

pub use client::{
    EngineClient, EngineStatus, InProcessEngineClient, StreamCancelCause, StreamControl,
    StreamControlRequest,
};
pub use error::{Error, Result};
pub use generation::{GenerationEventStream, GenerationSubmission};
pub(crate) use in_process::RuntimeEngineClient;
pub use media::{MediaEventStream, MediaSubmission};
pub use zmq::{EngineId, TransportMode, ZmqClientConfig, ZmqEngineClient};

impl EngineClient {
    pub fn snapshot(&self) -> EngineSnapshot {
        EngineSnapshot {
            model_name: self.model_name().to_string(),
            engine_count: self.engine_count(),
            max_model_len: self.max_model_len(),
            model_dtype: self.model_dtype(),
            generation_capabilities: self.generation_capabilities(),
        }
    }

    pub fn status(self: &Arc<Self>) -> EngineStatus {
        EngineStatus::new(self)
    }

    pub async fn cancel_request(&self, request_id: &str) -> Result<()> {
        self.cancel(&[request_id.to_string()]).await
    }

    pub async fn abort_request(&self, request_id: &str) -> Result<()> {
        self.abort(&[request_id.to_string()]).await
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
}
