use std::sync::Arc;

use crate::error::{Error, Result};
use crate::protocol::handshake::EngineCoreReadyResponse;
use crate::protocol::lora::LoraRequest;
use crate::protocol::{EngineCoreRequest, ModelDtype};
use crate::{NativeEventStream, NativeGenerateRequest};

pub(crate) mod state;
pub(crate) mod stream;

pub use stream::{EngineCoreOutputStream, EngineCoreStreamOutput};

/// The reason a request stream is being aborted when its output stream is
/// dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AbortCause {
    /// The consumer dropped the stream before the request reached a terminal
    /// engine output.
    #[default]
    DroppedStream,
    /// The frontend matched a stop string locally and intentionally stopped
    /// consuming the stream.
    StopStringMatched,
}

task_local::task_local! {
    static ABORT_CAUSE: AbortCause;
}

impl AbortCause {
    /// Return the abort cause currently associated with this task, or
    /// [`AbortCause::DroppedStream`] by default.
    pub fn current() -> Self {
        ABORT_CAUSE.try_get().unwrap_or_default()
    }

    /// Drop one value while marking the drop as happening for this abort cause.
    pub fn drop_as<T>(self, value: T) {
        ABORT_CAUSE.sync_scope(self, move || drop(value));
    }
}

/// Auto-abort work item sent from stream `Drop` handlers to the active backend.
#[derive(Debug, Clone)]
pub struct AbortRequest {
    pub request_id: String,
    pub cause: AbortCause,
}

/// Backend contract for a runtime hosted in the same process as the server.

/// This keeps engine construction below the server/runtime boundary while
/// preserving one frontend-facing client API for in-process, socket, and mock
/// modes.
pub trait InProcessEngineClient: Send + Sync {
    fn call(&self, req: EngineCoreRequest) -> Result<EngineCoreOutputStream>;
    fn generate_native(&self, req: NativeGenerateRequest) -> Result<NativeEventStream>;
    fn abort(&self, ids: &[String]) -> Result<()>;
    fn engine_count(&self) -> usize;
    fn model_dtype(&self) -> ModelDtype;
    fn uniserve_version(&self) -> &str;
    fn total_num_gpu_blocks(&self) -> u64;
    fn max_model_len(&self) -> u32;
    fn model_name(&self) -> &str;
    fn is_healthy(&self) -> bool;
    fn health_error(&self) -> Option<Arc<Error>> {
        None
    }
    fn collective_rpc(&self, method: &str) -> Result<Vec<rmpv::Value>>;
    fn is_sleeping(&self) -> Result<bool>;
    fn reset_mm_cache(&self) -> Result<()>;
    fn reset_encoder_cache(&self) -> Result<()>;
    fn reset_prefix_cache(
        &self,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> Result<bool>;
    fn add_lora(&self, lora_request: &LoraRequest) -> Result<bool>;
    fn remove_lora(&self, lora_id: u64) -> Result<bool>;
    fn sleep(&self, level: u32, mode: &str) -> Result<()>;
    fn wake_up(&self, tags: Option<Vec<String>>) -> Result<()>;
    fn shutdown(self: Box<Self>) -> Result<()>;
}

/// The engine client used by the frontend.

/// It dispatches behind a single API to one of:
/// - [`Self::InProcess`]: a server-owned runtime backend hosted in this
/// process.
/// - [`Self::Zmq`]: the socket transport to one or more headless
/// `uniserve engine` processes.
/// - [`Self::Mock`]: an in-process scriptable engine used by the test suite
/// (see [`crate::mock`]).
pub enum EngineCoreClient {
    InProcess(Box<dyn InProcessEngineClient>),
    Zmq(Box<crate::zmq::ZmqEngineCoreClient>),
    Mock(crate::mock::MockEngineClient),
}

impl EngineCoreClient {
    /// Wrap a server-owned in-process runtime backend.
    pub fn from_in_process(client: impl InProcessEngineClient + 'static) -> Self {
        Self::InProcess(Box::new(client))
    }

    /// Connect to out-of-process engine cores over the ZMQ transport.
    pub async fn connect_zmq(config: crate::zmq::ZmqClientConfig) -> Result<Self> {
        Ok(Self::Zmq(Box::new(
            crate::zmq::ZmqEngineCoreClient::connect(config).await?,
        )))
    }

    /// Build a mock-backed client plus its scripting handle (test infrastructure).
    pub fn connect_mock(model_name: impl Into<String>) -> (Self, crate::mock::MockEngine) {
        let (client, engine) = crate::mock::connect_mock(model_name);
        (Self::Mock(client), engine)
    }

    /// Add a request and return its per-request raw output stream.
    pub async fn call(&self, req: EngineCoreRequest) -> Result<EngineCoreOutputStream> {
        match self {
            Self::InProcess(c) => c.call(req),
            Self::Zmq(c) => c.call(req).await,
            Self::Mock(c) => c.call(req),
        }
    }

    /// Submit a native generation request and return its typed
    /// text+image event stream.
    pub async fn generate_native(&self, req: NativeGenerateRequest) -> Result<NativeEventStream> {
        match self {
            Self::InProcess(c) => c.generate_native(req),
            Self::Zmq(c) => c.generate_native(req).await,
            Self::Mock(c) => {
                let request_id = format!("native-mock-{}", native_mock_seq());
                let wire = crate::native::native_request_to_wire(req, request_id);
                let stream = c.call(wire)?;
                Ok(crate::native::native_stream_from_wire_stream(stream))
            }
        }
    }

    /// Abort in-flight requests by request id.
    pub async fn abort(&self, ids: &[String]) -> Result<()> {
        match self {
            Self::InProcess(c) => c.abort(ids),
            Self::Zmq(c) => c.abort(ids).await,
            Self::Mock(c) => {
                c.abort(ids);
                Ok(())
            }
        }
    }

    /// Number of engines backing this client.
    pub fn engine_count(&self) -> usize {
        match self {
            Self::InProcess(c) => c.engine_count(),
            Self::Zmq(c) => c.engine_count(),
            Self::Mock(_) => 1,
        }
    }

    /// Engine routing identities (empty for the in-process engines).
    pub fn engine_identities(&self) -> Vec<&[u8]> {
        match self {
            Self::Zmq(c) => c.engine_identities(),
            Self::InProcess(_) | Self::Mock(_) => Vec::new(),
        }
    }

    /// Engine ready responses (empty for the in-process engines).
    pub fn ready_responses(&self) -> Vec<&EngineCoreReadyResponse> {
        match self {
            Self::Zmq(c) => c.ready_responses(),
            Self::InProcess(_) | Self::Mock(_) => Vec::new(),
        }
    }

    /// Effective model dtype.
    pub fn model_dtype(&self) -> ModelDtype {
        match self {
            Self::InProcess(c) => c.model_dtype(),
            Self::Zmq(c) => c.model_dtype(),
            Self::Mock(_) => ModelDtype::BFloat16,
        }
    }

    /// Engine version string for `/version`.
    pub fn uniserve_version(&self) -> &str {
        match self {
            Self::InProcess(c) => c.uniserve_version(),
            Self::Zmq(c) => c.engine_version(),
            Self::Mock(_) => "mock",
        }
    }

    /// Total number of GPU KV blocks across engines.
    pub fn total_num_gpu_blocks(&self) -> u64 {
        match self {
            Self::InProcess(c) => c.total_num_gpu_blocks(),
            Self::Zmq(c) => c.total_num_gpu_blocks(),
            Self::Mock(_) => 0,
        }
    }

    /// Minimum effective `max_model_len`.
    pub fn max_model_len(&self) -> u32 {
        match self {
            Self::InProcess(c) => c.max_model_len(),
            Self::Zmq(c) => c.max_model_len(),
            Self::Mock(_) => 32768,
        }
    }

    /// Model name used for metrics labeling.
    pub fn model_name(&self) -> &str {
        match self {
            Self::InProcess(c) => c.model_name(),
            Self::Zmq(c) => c.model_name(),
            Self::Mock(c) => c.model_name(),
        }
    }

    /// Whether the client still considers the engine healthy.
    pub fn is_healthy(&self) -> bool {
        match self {
            Self::InProcess(c) => c.is_healthy(),
            Self::Zmq(c) => c.is_healthy(),
            Self::Mock(_) => true,
        }
    }

    /// First persistent health error, if any.
    pub fn health_error(&self) -> Option<Arc<Error>> {
        match self {
            Self::InProcess(c) => c.health_error(),
            Self::Zmq(c) => c.health_error(),
            Self::Mock(_) => None,
        }
    }

    /// Run `collective_rpc` across engines and their worker ranks.
    pub async fn collective_rpc<A, K>(
        &self,
        method: &str,
        timeout: Option<f64>,
        args: A,
        kwargs: K,
    ) -> Result<Vec<rmpv::Value>>
    where
        A: serde::Serialize + std::fmt::Debug,
        K: serde::Serialize + std::fmt::Debug,
    {
        match self {
            Self::InProcess(c) => c.collective_rpc(method),
            Self::Zmq(c) => c.collective_rpc(method, timeout, args, kwargs).await,
            Self::Mock(_) => Ok(Vec::new()),
        }
    }

    /// Whether the engine is currently sleeping.
    pub async fn is_sleeping(&self) -> Result<bool> {
        match self {
            Self::InProcess(c) => c.is_sleeping(),
            Self::Zmq(c) => c.is_sleeping().await,
            Self::Mock(_) => Ok(false),
        }
    }

    /// Reset the multimodal cache.
    pub async fn reset_mm_cache(&self) -> Result<()> {
        match self {
            Self::InProcess(c) => c.reset_mm_cache(),
            Self::Zmq(c) => c.reset_mm_cache().await,
            Self::Mock(_) => Ok(()),
        }
    }

    /// Reset the encoder cache.
    pub async fn reset_encoder_cache(&self) -> Result<()> {
        match self {
            Self::InProcess(c) => c.reset_encoder_cache(),
            Self::Zmq(c) => c.reset_encoder_cache().await,
            Self::Mock(_) => Ok(()),
        }
    }

    /// Reset the prefix cache.
    pub async fn reset_prefix_cache(
        &self,
        _reset_running_requests: bool,
        _reset_connector: bool,
    ) -> Result<bool> {
        match self {
            Self::InProcess(c) => c.reset_prefix_cache(_reset_running_requests, _reset_connector),
            Self::Zmq(c) => c.reset_prefix_cache().await,
            Self::Mock(_) => Ok(true),
        }
    }

    /// Load or refresh one LoRA adapter.
    pub async fn add_lora(&self, lora_request: &LoraRequest) -> Result<bool> {
        match self {
            Self::InProcess(c) => c.add_lora(lora_request),
            Self::Zmq(c) => c.add_lora(lora_request).await,
            Self::Mock(_) => Ok(true),
        }
    }

    /// Remove one LoRA adapter.
    pub async fn remove_lora(&self, lora_id: u64) -> Result<bool> {
        match self {
            Self::InProcess(c) => c.remove_lora(lora_id),
            Self::Zmq(c) => c.remove_lora(lora_id).await,
            Self::Mock(_) => Ok(true),
        }
    }

    /// Put the engine to sleep.
    pub async fn sleep(&self, level: u32, mode: &str) -> Result<()> {
        match self {
            Self::InProcess(c) => c.sleep(level, mode),
            Self::Zmq(c) => c.sleep(level, mode).await,
            Self::Mock(_) => Ok(()),
        }
    }

    /// Wake the engine from sleep.
    pub async fn wake_up(&self, tags: Option<Vec<String>>) -> Result<()> {
        match self {
            Self::InProcess(c) => c.wake_up(tags),
            Self::Zmq(c) => c.wake_up(tags).await,
            Self::Mock(_) => Ok(()),
        }
    }

    /// Shut down the client and its engine.
    pub async fn shutdown(self) -> Result<()> {
        match self {
            Self::InProcess(c) => c.shutdown(),
            Self::Zmq(c) => c.shutdown().await,
            Self::Mock(_) => Ok(()),
        }
    }
}

fn native_mock_seq() -> u64 {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
