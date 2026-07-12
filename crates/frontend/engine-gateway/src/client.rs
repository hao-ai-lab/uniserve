use std::sync::{Arc, Weak};

use crate::GenerationEventStream;
use crate::error::{Error, Result};
use crate::protocol::EngineCoreRequest;
use crate::protocol::handshake::EngineCoreReadyResponse;
use crate::protocol::lora::LoraRequest;
use uniserve_core::{GenerationRuntimeCapabilities, ModelDtype, OpKind};

pub(crate) mod state;
pub(crate) mod stream;

pub use stream::{EngineCoreOutputStream, EngineCoreStreamOutput};

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

/// Cancellation work item sent from stream `Drop` handlers to the active backend.
#[derive(Debug, Clone)]
pub struct StreamCancelRequest {
    pub request_id: String,
    pub cause: StreamCancelCause,
}

/// Backend contract for a runtime hosted in the same process as the server.
///
/// This keeps engine construction below the server/runtime boundary while
/// preserving one frontend-facing client API for in-process, socket, and mock
/// modes.
pub trait InProcessEngineClient: Send + Sync {
    fn call(&self, req: EngineCoreRequest) -> Result<EngineCoreOutputStream>;
    fn submit_generation(
        &self,
        submission: crate::generation::GenerationSubmission,
    ) -> Result<GenerationEventStream>;
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
///
/// It dispatches behind a single API to one of:
/// - [`Self::InProcess`]: a server-owned runtime backend hosted in this
///   process.
/// - [`Self::Zmq`]: the socket transport to one or more headless
///   `uniserve engine` processes.
/// - [`Self::Mock`]: an in-process scriptable engine used by the test suite
///   (see [`crate::mock`]).
pub enum EngineCoreClient {
    InProcess(Box<dyn InProcessEngineClient>),
    Zmq(Box<crate::zmq::ZmqEngineCoreClient>),
    Mock(crate::mock::MockEngineClient),
}

/// Owned, typed application control capability for an engine connection.
///
/// The weak reference keeps this capability independent of execution
/// ownership and deliberately exposes neither generation submission nor raw
/// transport calls.
#[derive(Clone)]
pub struct EngineAppControl {
    client: Weak<EngineCoreClient>,
}

impl EngineAppControl {
    pub(crate) fn new(client: &Arc<EngineCoreClient>) -> Self {
        Self {
            client: Arc::downgrade(client),
        }
    }

    fn client(&self) -> Result<Arc<EngineCoreClient>> {
        self.client
            .upgrade()
            .ok_or(Error::ApplicationControlUnavailable)
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
        self.client()?
            .collective_rpc(method, timeout, args, kwargs)
            .await
    }

    pub async fn is_sleeping(&self) -> Result<bool> {
        self.client()?.is_sleeping().await
    }

    pub async fn reset_mm_cache(&self) -> Result<()> {
        self.client()?.reset_mm_cache().await
    }

    pub async fn reset_encoder_cache(&self) -> Result<()> {
        self.client()?.reset_encoder_cache().await
    }

    pub async fn reset_prefix_cache(
        &self,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> Result<bool> {
        self.client()?
            .reset_prefix_cache(reset_running_requests, reset_connector)
            .await
    }

    pub async fn add_lora(&self, request: &LoraRequest) -> Result<bool> {
        self.client()?.add_lora(request).await
    }

    pub async fn remove_lora(&self, lora_id: u64) -> Result<bool> {
        self.client()?.remove_lora(lora_id).await
    }

    pub async fn sleep(&self, level: u32, mode: &str) -> Result<()> {
        self.client()?.sleep(level, mode).await
    }

    pub async fn wake_up(&self, tags: Option<Vec<String>>) -> Result<()> {
        self.client()?.wake_up(tags).await
    }
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

    /// Submit a canonical generation request and return its typed event stream.
    pub async fn submit_generation(
        &self,
        submission: crate::generation::GenerationSubmission,
    ) -> Result<GenerationEventStream> {
        match self {
            Self::InProcess(c) => c.submit_generation(submission),
            Self::Zmq(c) => c.submit_generation(submission).await,
            Self::Mock(c) => {
                let wire = crate::generation::generation_request_to_wire(submission);
                let stream = c.call(wire)?;
                Ok(crate::generation::generation_event_stream_from_wire(stream))
            }
        }
    }

    /// Cancel in-flight requests because their owners no longer need output.
    pub async fn cancel(&self, ids: &[String]) -> Result<()> {
        match self {
            Self::InProcess(c) => c.cancel(ids),
            Self::Zmq(c) => c.cancel(ids).await,
            Self::Mock(c) => {
                c.cancel(ids);
                Ok(())
            }
        }
    }

    /// Abort in-flight requests because serving cannot continue them.
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

    /// Lowest common generation limits across every engine that may receive a
    /// request from this client.
    pub fn generation_capabilities(&self) -> GenerationRuntimeCapabilities {
        match self {
            Self::InProcess(c) => c.generation_capabilities(),
            Self::Zmq(c) => c.generation_capabilities(),
            Self::Mock(_) => GenerationRuntimeCapabilities {
                supported_ops: vec![
                    OpKind::PrefillUnd,
                    OpKind::DecodeUnd,
                    OpKind::VaeEncode,
                    OpKind::VitEncode,
                    OpKind::DenoiseGen,
                    OpKind::CommitGen,
                    OpKind::CommitWriteback,
                ],
                max_latent_units: 1 << 20,
                latent_downsample: 16,
                max_vae_grid_tokens: 4096,
                max_vit_grid_tokens: 4096,
                commit_marker_tokens: 2,
                max_cfg_branches: 3,
                scratch_capacity_tokens: 1 << 20,
                scratch_block_size: 64,
                encoder_cache_entries: 256,
                generated_image_commit: uniserve_core::GeneratedImageCommitCapabilities {
                    inline: true,
                    separate_writeback: true,
                },
            },
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
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> Result<bool> {
        match self {
            Self::InProcess(c) => c.reset_prefix_cache(reset_running_requests, reset_connector),
            Self::Zmq(c) => {
                c.reset_prefix_cache(reset_running_requests, reset_connector)
                    .await
            }
            Self::Mock(_) if reset_connector => Err(Error::UnsupportedControl {
                control: "external prefix-cache reset".to_string(),
            }),
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
