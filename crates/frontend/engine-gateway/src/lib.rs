//! Engine transport and generation submission gateway.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::sync::Arc;

mod client;
mod error;
pub mod generation;
pub mod metrics;
pub mod mock;
#[doc(hidden)]
pub mod test_utils;
pub mod zmq;

pub use uniserve_engine_wire as protocol;

pub use client::{
    EngineAppControl, EngineCoreClient, EngineCoreOutputStream, EngineCoreStreamOutput,
    InProcessEngineClient, StreamCancelCause, StreamControl, StreamControlRequest,
};
pub use error::{Error, Result};
pub use generation::{
    EngineSamplingParams, GenEvent, GenerationConstraint, GenerationEventStream,
    GenerationFinishReason, GenerationPositionLogprobs, GenerationSubmission,
    GenerationTokenLogprob, ImageParams,
};
pub use mock::{MockClientMessage, MockEngine};
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

    /// Create the application-owned lifecycle and administration capability.
    pub fn app_control(&self) -> EngineAppControl {
        EngineAppControl::new(&self.client)
    }

    pub async fn submit_generation(
        &self,
        submission: GenerationSubmission,
    ) -> Result<GenerationEventStream> {
        self.client().submit_generation(submission).await
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
    pub use super::test_utils;
    pub use super::{
        EngineAppControl, EngineCoreClient, EngineCoreOutputStream, EngineCoreStreamOutput,
        EngineId, EngineSamplingParams, Error, GenEvent, GenerationConstraint,
        GenerationEventStream, GenerationFinishReason, GenerationSubmission, ImageParams,
        InProcessEngineClient, MockClientMessage, MockEngine, Result, StreamCancelCause,
        StreamControl, StreamControlRequest, TransportMode, ZmqClientConfig, ZmqEngineCoreClient,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn application_control_does_not_keep_execution_gateway_alive() {
        let (client, _mock) = EngineCoreClient::connect_mock("test-model");
        let gateway = EngineGateway::new(client);
        let control = gateway.app_control();
        assert!(control.is_healthy());
        assert_eq!(control.version().unwrap(), "mock");

        gateway.shutdown().await.unwrap();

        assert!(!control.is_healthy());
        assert!(matches!(
            control.version(),
            Err(Error::ApplicationControlUnavailable)
        ));
    }
}
