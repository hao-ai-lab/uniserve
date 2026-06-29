#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod client;
mod error;
pub mod metrics;
pub mod mock;
pub mod native;
#[cfg(any(test, feature = "test-util"))]
pub mod test_utils;
pub mod zmq;

/// The wire protocol DTOs live in the shared engine-wire crate; re-export them
/// under the client-facing `protocol` path.
pub use uniserve_engine_wire as protocol;

pub use client::{
    AbortCause, AbortRequest, EngineCoreClient, EngineCoreOutputStream, EngineCoreStreamOutput,
    InProcessEngineClient,
};
pub use error::{Error, Result};
pub use mock::{MockClientMessage, MockEngine};
pub use native::{
    EngineSamplingParams, GenEvent, GenMode, ImageParams, MmItem, NativeEventStream,
    NativeFinishReason, NativeGenerateRequest,
};
pub use zmq::{EngineId, TransportMode, ZmqClientConfig, ZmqEngineCoreClient};
