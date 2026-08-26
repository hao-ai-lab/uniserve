use std::sync::{Arc, Weak};

use crate::engine_client::GenerationEventStream;
use crate::engine_client::error::{Error, Result};
use uniserve_core::codec::handshake::EngineCoreReadyResponse;
use uniserve_core::{GenerationRuntimeCapabilities, ModelDtype};

pub(crate) mod state;

/// The reason a request stream is being cancelled when its output stream is
/// dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamCancelCause {
    /// The consumer dropped the stream before the request reached a terminal
    /// engine output.
    #[default]
    DroppedStream,
    /// The frontend matched a stop string locally and intentionally stopped
    /// consuming the stream.
    StopStringMatched,
}

task_local::task_local! {
    static STREAM_CANCEL_CAUSE: StreamCancelCause;
}

impl StreamCancelCause {
    /// Return the cancellation cause currently associated with this task, or
    /// [`StreamCancelCause::DroppedStream`] by default.
    pub fn current() -> Self {
        STREAM_CANCEL_CAUSE.try_get().unwrap_or_default()
    }

    /// Drop one value while marking the drop as happening for this cancellation cause.
    pub fn drop_as<T>(self, value: T) {
        STREAM_CANCEL_CAUSE.sync_scope(self, move || drop(value));
    }
}

/// Request-scoped semantic control emitted by a frontend output stream.
#[derive(Debug, Clone)]
pub enum StreamControl {
    Cancel {
        cause: StreamCancelCause,
        output_token_count: usize,
    },
    Acknowledge {
        output_token_count: usize,
    },
}

/// Stream control work item sent to the active backend.
#[derive(Debug, Clone)]
pub struct StreamControlRequest {
    pub request_id: String,
    pub control: StreamControl,
}

/// Backend contract for a runtime hosted in the same process as the server.
///
/// This keeps engine construction below the server/runtime boundary while
/// preserving one frontend-facing client API for in-process, socket, and mock
/// modes.
pub trait InProcessEngineClient: Send + Sync {
    fn submit_generation(
        &self,
        submission: crate::engine_client::generation::GenerationSubmission,
    ) -> Result<GenerationEventStream>;
    fn submit_media(
        &self,
        submission: crate::engine_client::media::MediaSubmission,
    ) -> Result<crate::engine_client::media::MediaEventStream>;
    fn cancel(&self, ids: &[String]) -> Result<()>;
    fn abort(&self, ids: &[String]) -> Result<()>;
    fn engine_count(&self) -> usize;
    fn model_dtype(&self) -> ModelDtype;
    fn uniserve_version(&self) -> &str;
    fn total_num_gpu_blocks(&self) -> u64;
    fn max_model_len(&self) -> u32;
    fn generation_capabilities(&self) -> GenerationRuntimeCapabilities;
    fn model_name(&self) -> &str;
    fn is_healthy(&self) -> bool;
    fn health_error(&self) -> Option<Arc<Error>> {
        None
    }
    fn shutdown(self: Box<Self>) -> Result<()>;
}

/// The engine client used by the frontend.
///
/// It dispatches behind a single API to one of:
/// - [`Self::InProcess`]: a server-owned runtime backend hosted in this
///   process.
/// - [`Self::Zmq`]: the socket transport to one or more headless
///   `uniserve engine` processes.
pub enum EngineClient {
    InProcess(Box<dyn InProcessEngineClient>),
    Zmq(Box<crate::engine_client::zmq::ZmqEngineClient>),
}

/// Owned, typed health and build-provenance capability for an engine connection.
///
/// The weak reference keeps this capability independent of execution
/// ownership and deliberately exposes no generation submission capability.
#[derive(Clone)]
pub struct EngineStatus {
    client: Weak<EngineClient>,
}

impl EngineStatus {
    pub(crate) fn new(client: &Arc<EngineClient>) -> Self {
        Self {
            client: Arc::downgrade(client),
        }
    }

    fn client(&self) -> Result<Arc<EngineClient>> {
        self.client.upgrade().ok_or(Error::StatusUnavailable)
    }

    pub fn version(&self) -> Result<String> {
        Ok(self.client()?.uniserve_version().to_string())
    }

    pub fn is_healthy(&self) -> bool {
        self.client
            .upgrade()
            .is_some_and(|client| client.is_healthy())
    }

    pub fn health_error(&self) -> Option<Arc<Error>> {
        self.client
            .upgrade()
            .and_then(|client| client.health_error())
    }
}

impl EngineClient {
    /// Wrap a server-owned in-process runtime backend.
    pub fn from_in_process(client: impl InProcessEngineClient + 'static) -> Self {
        Self::InProcess(Box::new(client))
    }

    /// Connect to out-of-process engine cores over the ZMQ transport.
    pub async fn connect_zmq(config: crate::engine_client::zmq::ZmqClientConfig) -> Result<Self> {
        Ok(Self::Zmq(Box::new(
            crate::engine_client::zmq::ZmqEngineClient::connect(config).await?,
        )))
    }

    /// Submit a canonical generation request and return its typed event stream.
    pub async fn submit_generation(
        &self,
        submission: crate::engine_client::generation::GenerationSubmission,
    ) -> Result<GenerationEventStream> {
        match self {
            Self::InProcess(c) => c.submit_generation(submission),
            Self::Zmq(c) => c.submit_generation(submission).await,
        }
    }

    pub async fn submit_media(
        &self,
        submission: crate::engine_client::media::MediaSubmission,
    ) -> Result<crate::engine_client::media::MediaEventStream> {
        match self {
            Self::InProcess(client) => client.submit_media(submission),
            Self::Zmq(client) => client.submit_media(submission).await,
        }
    }

    /// Cancel in-flight requests because their owners no longer need output.
    pub async fn cancel(&self, ids: &[String]) -> Result<()> {
        match self {
            Self::InProcess(c) => c.cancel(ids),
            Self::Zmq(c) => c.cancel(ids).await,
        }
    }

    /// Abort in-flight requests because serving cannot continue them.
    pub async fn abort(&self, ids: &[String]) -> Result<()> {
        match self {
            Self::InProcess(c) => c.abort(ids),
            Self::Zmq(c) => c.abort(ids).await,
        }
    }

    /// Number of engines backing this client.
    pub fn engine_count(&self) -> usize {
        match self {
            Self::InProcess(c) => c.engine_count(),
            Self::Zmq(c) => c.engine_count(),
        }
    }

    /// Engine routing identities (empty for the in-process engines).
    pub fn engine_identities(&self) -> Vec<&[u8]> {
        match self {
            Self::Zmq(c) => c.engine_identities(),
            Self::InProcess(_) => Vec::new(),
        }
    }

    /// Engine ready responses (empty for the in-process engines).
    pub fn ready_responses(&self) -> Vec<&EngineCoreReadyResponse> {
        match self {
            Self::Zmq(c) => c.ready_responses(),
            Self::InProcess(_) => Vec::new(),
        }
    }

    /// Effective model dtype.
    pub fn model_dtype(&self) -> ModelDtype {
        match self {
            Self::InProcess(c) => c.model_dtype(),
            Self::Zmq(c) => c.model_dtype(),
        }
    }

    /// Engine version string for `/version`.
    pub fn uniserve_version(&self) -> &str {
        match self {
            Self::InProcess(c) => c.uniserve_version(),
            Self::Zmq(c) => c.engine_version(),
        }
    }

    /// Total number of GPU KV blocks across engines.
    pub fn total_num_gpu_blocks(&self) -> u64 {
        match self {
            Self::InProcess(c) => c.total_num_gpu_blocks(),
            Self::Zmq(c) => c.total_num_gpu_blocks(),
        }
    }

    /// Minimum effective `max_model_len`.
    pub fn max_model_len(&self) -> u32 {
        match self {
            Self::InProcess(c) => c.max_model_len(),
            Self::Zmq(c) => c.max_model_len(),
        }
    }

    /// Lowest common generation limits across every engine that may receive a
    /// request from this client.
    pub fn generation_capabilities(&self) -> GenerationRuntimeCapabilities {
        match self {
            Self::InProcess(c) => c.generation_capabilities(),
            Self::Zmq(c) => c.generation_capabilities(),
        }
    }

    /// Model name used for metrics labeling.
    pub fn model_name(&self) -> &str {
        match self {
            Self::InProcess(c) => c.model_name(),
            Self::Zmq(c) => c.model_name(),
        }
    }

    /// Whether the client still considers the engine healthy.
    pub fn is_healthy(&self) -> bool {
        match self {
            Self::InProcess(c) => c.is_healthy(),
            Self::Zmq(c) => c.is_healthy(),
        }
    }

    /// First persistent health error, if any.
    pub fn health_error(&self) -> Option<Arc<Error>> {
        match self {
            Self::InProcess(c) => c.health_error(),
            Self::Zmq(c) => c.health_error(),
        }
    }

    /// Shut down the client and its engine.
    pub async fn shutdown(self) -> Result<()> {
        match self {
            Self::InProcess(c) => c.shutdown(),
            Self::Zmq(c) => c.shutdown().await,
        }
    }
}
