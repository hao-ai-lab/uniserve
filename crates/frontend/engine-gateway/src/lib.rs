//! Engine transport and generation submission gateway.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::sync::Arc;

mod client;
mod error;
pub mod generation;
pub mod media;
pub mod metrics;
pub mod zmq;

pub use uniserve_engine_wire as protocol;

pub use client::{
    EngineCoreClient, EngineStatus, InProcessEngineClient, StreamCancelCause, StreamControl,
    StreamControlRequest,
};
pub use error::{Error, Result};
pub use generation::{
    EngineSamplingParams, GenEvent, GenerationConstraint, GenerationEventStream,
    GenerationFinishReason, GenerationPositionLogprobs, GenerationSubmission,
    GenerationTokenLogprob, ImageParams, PublicCommit, PublicModality, SemanticRoot,
};
pub use media::{MediaEvent, MediaEventStream, MediaSubmission};
pub use zmq::{EngineId, TransportMode, ZmqClientConfig, ZmqEngineCoreClient};

/// Runtime transport seam for generation submission and request control.
#[derive(Clone)]
pub struct EngineGateway {
    client: Arc<EngineCoreClient>,
    stats_logger: Option<Arc<generation::log_stats::StatsLogger>>,
}

impl EngineGateway {
    pub fn new(client: EngineCoreClient) -> Self {
        Self {
            client: Arc::new(client),
            stats_logger: None,
        }
    }

    pub fn with_log_stats(mut self, enabled: bool) -> Self {
        self.stats_logger = enabled.then(|| {
            Arc::new(generation::log_stats::StatsLogger::start(
                self.client.model_name().to_string(),
                self.client.engine_count(),
            ))
        });
        self
    }

    pub fn snapshot(&self) -> EngineGatewaySnapshot {
        EngineGatewaySnapshot {
            model_name: self.client().model_name().to_string(),
            engine_count: self.client().engine_count(),
            max_model_len: self.client().max_model_len(),
            model_dtype: self.client().model_dtype(),
            generation_capabilities: self.client().generation_capabilities(),
        }
    }

    fn client(&self) -> &EngineCoreClient {
        &self.client
    }

    /// Create the application-owned health and build-provenance view.
    pub fn status(&self) -> EngineStatus {
        EngineStatus::new(&self.client)
    }

    pub async fn submit_generation(
        &self,
        submission: GenerationSubmission,
    ) -> Result<GenerationEventStream> {
        self.client().submit_generation(submission).await
    }

    pub async fn submit_media(&self, submission: MediaSubmission) -> Result<MediaEventStream> {
        self.client().submit_media(submission).await
    }

    /// Cancel an in-flight request because its caller no longer needs output.
    pub async fn cancel(&self, request_id: &str) -> Result<()> {
        self.client().cancel(&[request_id.to_string()]).await
    }

    /// Abort an in-flight request because serving cannot continue it.
    pub async fn abort(&self, request_id: &str) -> Result<()> {
        self.client().abort(&[request_id.to_string()]).await
    }

    pub async fn shutdown(self) -> Result<()> {
        drop(self.stats_logger);
        let client = Arc::try_unwrap(self.client).map_err(|_| Error::ClientClosed {
            message: "execution gateway still has a strong client owner during shutdown"
                .to_string(),
        })?;
        client.shutdown().await
    }
}

/// Namespaced transport interface for callers that do not use generation lowering.
pub mod transport {
    pub use super::protocol;
    pub use super::{
        EngineCoreClient, EngineId, EngineSamplingParams, EngineStatus, Error, GenEvent,
        GenerationConstraint, GenerationEventStream, GenerationFinishReason, GenerationSubmission,
        ImageParams, InProcessEngineClient, MediaEvent, MediaEventStream, MediaSubmission,
        PublicCommit, PublicModality, Result, SemanticRoot, StreamCancelCause, StreamControl,
        StreamControlRequest, TransportMode, ZmqClientConfig, ZmqEngineCoreClient,
    };
    pub use super::{generation, metrics, zmq};
}

/// Gateway health and capability snapshot carried into execution plans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineGatewaySnapshot {
    pub model_name: String,
    pub engine_count: usize,
    pub max_model_len: u32,
    pub model_dtype: uniserve_core::ModelDtype,
    pub generation_capabilities: uniserve_core::GenerationRuntimeCapabilities,
}
