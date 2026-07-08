//! Engine gateway boundary used by serving runtimes.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub use uniserve_engine_client::protocol;
#[cfg(any(test, feature = "test-util"))]
pub use uniserve_engine_client::test_utils;
pub use uniserve_engine_client::{
    AbortCause, AbortRequest, EngineCoreClient, EngineCoreOutputStream, EngineCoreStreamOutput,
    InProcessEngineClient, NativeEventStream, TransportMode, ZmqClientConfig,
};
pub use uniserve_engine_client::{EngineSamplingParams, GenEvent, GenerationConstraint};
pub use uniserve_engine_client::{ImageParams, MmItem, NativeFinishReason, NativeGenerateRequest};
pub use uniserve_engine_client::{MockClientMessage, MockEngine};
pub use uniserve_llm::{
    CollectedGenerateOutput, FinishReason, GenerateOutput, GenerateOutputStream,
    GenerateOutputStreamExt, GeneratePromptInfo, GenerateRequest, Llm,
};

/// Gateway health and capability snapshot carried into execution plans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineGatewaySnapshot {
    pub model_name: String,
    pub engine_count: usize,
    pub max_model_len: u32,
}

impl EngineGatewaySnapshot {
    pub fn from_client(client: &EngineCoreClient) -> Self {
        Self {
            model_name: client.model_name().to_string(),
            engine_count: client.engine_count(),
            max_model_len: client.max_model_len(),
        }
    }
}
