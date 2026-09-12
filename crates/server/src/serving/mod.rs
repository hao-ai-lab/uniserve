//! Model-aware admission, engine submission, and output streaming.
//!
//! Requests follow one ownership chain:
//! `GenerateReqInput -> ResolvedModel::tokenize -> TokenizedGenerateReqInput ->
//! EngineClient::submit_generation -> ServeEvent stream`.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod assembly;
/// Chat request rendering and structured output processing.
pub mod chat;
mod input;
mod model;
mod omni;
#[cfg(test)]
mod test_support;
/// Text tokenization, decoding, and sampling utilities.
pub mod text;

use std::borrow::Borrow;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};
use std::time::Instant;

use crate::engine_client::{EngineClient, EventRx, MediaSubmission, StreamCancelCause};
use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Notify;
use uniserve_core::Event;

pub use input::{
    CacheBounds, DecodeControls, GenerateReqInput, ImageGenControls, ImageInput, ModalitySelection,
    ModelEventIdentity, OutputDetail, OutputProcessorPolicy, PromptInput, SamplingConfig,
    SchedulingBounds, StopConfig, TokenizedGenerateReqInput,
};
pub use model::{
    ModelSupport, ResolvedAssets, ResolvedModel, ServedEndpoint, ServedFeature, ServedModality,
    ServedSamplingControl,
};

use crate::serving::chat::{
    AssistantBlockKind, AssistantContentBlock, ChatEvent, Qwen3ChatOutputProcessor,
};
use crate::serving::omni::{SenseNovaOutputProcessor, SenseNovaTextDelta};
use crate::serving::text::output::stop_string_holdback_bytes;
use crate::serving::text::{
    DecodedLogprobs, DecodedPromptLogprobs, FinishReason, StopReason, TextDecodeOptions,
};

use assembly::{assemble_chat_event_stream, assemble_event_stream};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
/// Validated external request identifier used throughout serving.
pub struct ServeRequestId(String);

impl ServeRequestId {
    /// Constructs a serving request identifier.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Consumes the identifier and returns its string representation.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for ServeRequestId {
    /// Formats the value for diagnostic output.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Deref for ServeRequestId {
    type Target = str;

    /// Returns shared access to the wrapped value.
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<str> for ServeRequestId {
    /// Returns shared access to the wrapped value.
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for ServeRequestId {
    /// Borrows the wrapped value.
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl From<String> for ServeRequestId {
    /// Converts the source value into this type.
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ServeRequestId {
    /// Converts the source value into this type.
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<ServeRequestId> for String {
    /// Converts the source value into this type.
    fn from(value: ServeRequestId) -> Self {
        value.0
    }
}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
/// Stable identifier for one output candidate within a request.
pub struct CandidateId(u32);

impl CandidateId {
    /// Identifier of the primary output candidate.
    pub const PRIMARY: Self = Self(0);

    /// Returns the numeric candidate identifier.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for CandidateId {
    /// Converts the source value into this type.
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<CandidateId> for u32 {
    /// Converts the source value into this type.
    fn from(value: CandidateId) -> Self {
        value.0
    }
}

/// Boxed asynchronous stream of serving events.
pub type ServeEventStream = Pin<Box<dyn Stream<Item = Result<ServeEvent>> + Send>>;
/// Result type returned by serving-runtime operations.
pub type Result<T> = std::result::Result<T, ServeError>;

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Error)]
/// Failure while resolving and tokenizing a model input.
pub enum TokenizeError {
    /// The model tokenizer cannot encode request content.
    #[error(transparent)]
    Tokenizer(#[from] crate::profile::tokenizer::TokenizerError),
    /// Chat rendering or semantic request validation fails.
    #[error(transparent)]
    Chat(#[from] crate::serving::chat::Error),
    /// Text sampling or decoding options cannot be resolved.
    #[error(transparent)]
    Text(#[from] crate::serving::text::Error),
    /// Sampling controls violate the engine contract.
    #[error(transparent)]
    Sampling(#[from] uniserve_core::SamplingParamsError),
    /// The lowered request violates the canonical generation contract.
    #[error(transparent)]
    Generation(#[from] uniserve_core::GenerationRequestError),
    /// Multimodal request controls or geometry are invalid.
    #[error(transparent)]
    Omni(#[from] crate::serving::omni::OmniError),
    /// A requested log-probability count falls outside the accepted range.
    #[error("{field} must be non-negative or -1, got {value}")]
    InvalidLogprobCount {
        /// Invalid request field.
        field: &'static str,
        /// Rejected field value.
        value: i32,
    },
    /// The minimum generation length exceeds the maximum generation length.
    #[error("min_tokens ({min_tokens}) exceeds max_tokens ({max_tokens})")]
    MinTokensExceedsMaximum {
        /// Requested minimum generated-token count.
        min_tokens: u32,
        /// Effective maximum generated-token count.
        max_tokens: u32,
    },
    /// The selected model path cannot report token log probabilities.
    #[error("this model does not support logprobs")]
    UnsupportedLogprobs,
    /// The request contains an invalid combination of serving controls.
    #[error("{0}")]
    Invalid(String),
    /// The blocking tokenization task cannot complete.
    #[error("tokenization task failed")]
    Task(#[source] tokio::task::JoinError),
}

#[derive(Debug, Error)]
/// Failure while interpreting or assembling model output.
pub enum OutputProcessingError {
    /// Generated token identifiers cannot be decoded.
    #[error(transparent)]
    Tokenizer(#[from] crate::profile::tokenizer::TokenizerError),
    /// Structured chat output cannot be parsed or assembled.
    #[error(transparent)]
    Chat(#[from] crate::serving::chat::Error),
    /// Text output cannot be decoded or collected.
    #[error(transparent)]
    Text(#[from] crate::serving::text::Error),
    /// Reasoning delimiters cannot be parsed.
    #[error(transparent)]
    Reasoning(#[from] crate::profile::reasoning::ReasoningError),
    /// Engine output violates a serving-layer invariant.
    #[error("{0}")]
    Malformed(String),
}

/// Runtime-local serving errors.
#[derive(Debug, Error)]
pub enum ServeError {
    /// A request asks for more output candidates than the runtime supports.
    #[error("request `{request_id}` asks for {requested} outputs; only one output is supported")]
    UnsupportedOutputCount {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
        /// Requested number of candidates.
        requested: u32,
    },
    /// A request requires a feature unavailable on the selected model path.
    #[error("request `{request_id}` requires unsupported feature `{feature}`")]
    UnsupportedFeature {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
        /// Stable feature name.
        feature: &'static str,
    },
    /// The prompt exceeds the profile context limit.
    #[error(
        "request `{request_id}` has {prompt_tokens} prompt tokens, exceeding the {max_tokens}-token profile limit"
    )]
    ContextLengthExceeded {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
        /// Tokenized prompt length.
        prompt_tokens: usize,
        /// Maximum token count declared by the profile.
        max_tokens: u32,
    },
    /// The request's full KV requirement exceeds runtime capacity.
    #[error(
        "request `{request_id}` requires {required_tokens} KV tokens, exceeding the {max_tokens}-token runtime limit"
    )]
    ContextCapacityExceeded {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
        /// Total KV tokens required by the request.
        required_tokens: u64,
        /// Maximum KV tokens available to one request.
        max_tokens: u32,
    },
    /// Another live request already owns the same identifier.
    #[error("request `{request_id}` is already active")]
    DuplicateRequestId {
        /// Conflicting request identifier.
        request_id: ServeRequestId,
    },
    /// Model assets or capabilities cannot be resolved.
    #[error(transparent)]
    ModelResolution(#[from] crate::serving::model::ModelResolutionError),
    /// Engine submission or streaming fails.
    #[error(transparent)]
    Engine(#[from] crate::engine_client::Error),
    /// Request tokenization fails before engine submission.
    #[error("request `{request_id}` cannot be tokenized")]
    Tokenize {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
        /// Underlying tokenization failure.
        #[source]
        source: TokenizeError,
    },
    /// Engine output cannot be converted into serving events.
    #[error("request `{request_id}` output processing failed")]
    OutputProcessing {
        /// Identifier of the affected request.
        request_id: ServeRequestId,
        /// Underlying output-processing failure.
        #[source]
        source: OutputProcessingError,
    },
}

/// Builds an output-processing error for malformed engine output.
fn malformed_output(
    request_id: impl Into<ServeRequestId>,
    message: impl Into<String>,
) -> ServeError {
    ServeError::OutputProcessing {
        request_id: request_id.into(),
        source: OutputProcessingError::Malformed(message.into()),
    }
}

/// Folds cache namespace/salt into a stable isolation key.
pub(crate) fn cache_isolation_key(namespace: Option<&str>, salt: Option<&str>) -> Option<u64> {
    (namespace.is_some() || salt.is_some()).then(|| {
        let namespace = namespace.unwrap_or_default();
        let salt = salt.unwrap_or_default();
        let material = format!("{}:{namespace}{}:{salt}", namespace.len(), salt.len());
        material
            .as_bytes()
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
    })
}

/// Single-model serving runtime.
pub struct ServingRuntime {
    runtime_id: u64,
    model: Arc<ResolvedModel>,
    engine: Arc<EngineClient>,
    _stats_logger: Option<Arc<crate::engine_client::generation::log_stats::StatsLogger>>,
    metrics: Arc<RuntimeLifecycleMetrics>,
    requests: Arc<RuntimeRequestRegistry>,
}

#[derive(Debug, Clone, PartialEq)]
/// Tokenized prompt and geometry for terminal video generation.
pub struct VideoGenerationInput {
    /// Caller-visible request identifier.
    pub request_id: ServeRequestId,
    /// Text prompt for video generation.
    pub prompt: String,
    /// Deterministic generation seed.
    pub seed: u64,
    /// Requested video duration in seconds.
    pub seconds: f64,
    /// Optional scheduler grid-point count; the checkpoint owns the default.
    pub steps: Option<u32>,
}

impl ServingRuntime {
    /// Creates a serving runtime for one resolved model and engine client.
    pub fn new(model: ResolvedModel, engine: Arc<EngineClient>, log_stats: bool) -> Self {
        let stats_logger = log_stats.then(|| {
            Arc::new(
                crate::engine_client::generation::log_stats::StatsLogger::start(
                    engine.model_name().to_string(),
                    engine.engine_count(),
                ),
            )
        });
        Self {
            runtime_id: NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed),
            model: Arc::new(model),
            engine,
            _stats_logger: stats_logger,
            metrics: Arc::new(RuntimeLifecycleMetrics::default()),
            requests: Arc::new(RuntimeRequestRegistry::default()),
        }
    }

    /// Returns the resolved model owned by this runtime.
    pub fn model(&self) -> &ResolvedModel {
        &self.model
    }

    /// Returns the public served-model name.
    pub fn served_model_name(&self) -> &str {
        self.model.served_model_name()
    }

    /// Returns the engine client backing this runtime.
    pub fn engine(&self) -> &EngineClient {
        &self.engine
    }

    /// Submits a terminal video request to the engine.
    pub async fn generate_video(&self, request: VideoGenerationInput) -> Result<EventRx> {
        let submission = self.prepare_video(request)?;
        self.engine
            .submit_media(submission)
            .await
            .map_err(ServeError::Engine)
    }

    /// Validate and tokenize a video before transferring submission ownership.
    pub fn prepare_video(&self, request: VideoGenerationInput) -> Result<MediaSubmission> {
        if !self
            .model
            .support()
            .endpoints
            .contains(&ServedEndpoint::VideoGenerations)
        {
            return Err(ServeError::UnsupportedFeature {
                request_id: request.request_id,
                feature: "video_generation",
            });
        }
        if request.prompt.trim().is_empty() {
            return Err(ServeError::Tokenize {
                request_id: request.request_id,
                source: TokenizeError::Invalid("video prompt must not be empty".to_string()),
            });
        }
        let (geometry, prompt_token_ids) = self.model.resolve_video_request_geometry(
            &request.request_id,
            &request.prompt,
            request.seconds,
            request.steps,
        )?;
        let submission = MediaSubmission::new(
            request.request_id.to_string(),
            prompt_token_ids,
            request.seed,
            geometry,
        );
        Ok(submission)
    }

    /// Returns the process-local identity of this runtime instance.
    pub fn runtime_id(&self) -> u64 {
        self.runtime_id
    }

    /// Returns an aggregate point-in-time metrics snapshot.
    pub fn metrics_snapshot(&self) -> RuntimeMetricsSnapshot {
        let mut snapshot = self.metrics.snapshot();
        snapshot.active = self.requests.active_count() as u64;
        snapshot
    }

    /// Returns current lifecycle statistics for one request.
    pub fn request_stats(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        self.requests.stats(request_id)
    }

    /// Waits for one request to leave the active registry and returns its final snapshot.
    pub async fn drain_request(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        self.requests.drain_request(request_id).await
    }

    /// Waits until every active request leaves the runtime.
    pub async fn drain(&self) {
        self.requests.drain().await;
    }

    /// Generates a serving event stream for a validated request.
    pub async fn generate(&self, request: GenerateReqInput) -> Result<ServeEventStream> {
        let request_id = request.request_id.clone();
        let compile_started = Instant::now();
        let identity = self.model.event_identity();

        if let Err(error) = self.model.validate_request(&request) {
            self.metrics.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(error);
        }

        if !self.requests.register(
            request_id.clone(),
            identity.served_name.clone(),
            identity.description.clone(),
            0,
            RequestLifecycleState::Compiling,
        ) {
            self.metrics.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(ServeError::DuplicateRequestId { request_id });
        }

        let model = Arc::clone(&self.model);
        let tokenize_request_id = request_id.clone();
        let tokenize_result = tokio::select! {
            terminal = self.requests.wait_for_control(&request_id) => {
                return Ok(self.control_event_stream(request_id, terminal));
            }
            tokenized = tokio::task::spawn_blocking(move || model.tokenize(request)) => {
                tokenized.unwrap_or_else(|error| Err(ServeError::Tokenize {
                    request_id: tokenize_request_id,
                    source: TokenizeError::Task(error),
                }))
            }
        };
        let tokenized = match tokenize_result {
            Ok(tokenized) => tokenized,
            Err(error) => {
                self.metrics.rejected.fetch_add(1, Ordering::Relaxed);
                self.requests.complete(
                    &request_id,
                    RequestLifecycleState::Rejected,
                    compile_started.elapsed().as_micros() as u64,
                );
                return Err(error);
            }
        };
        let compile_duration_us = compile_started.elapsed().as_micros() as u64;
        self.requests
            .mark_submitting(&request_id, compile_duration_us);

        self.submit_and_stream(tokenized, compile_duration_us).await
    }

    /// Submits a tokenized request and wraps its output with model and lifecycle processing.
    async fn submit_and_stream(
        &self,
        tokenized: TokenizedGenerateReqInput,
        compile_duration_us: u64,
    ) -> Result<ServeEventStream> {
        let request_id = tokenized.request_id.clone();
        if let Some(terminal) = self.requests.control_terminal(&request_id) {
            return Ok(self.control_event_stream(request_id, terminal));
        }

        let engine_stream = self.engine.submit_generation(&tokenized).await;

        let TokenizedGenerateReqInput {
            request_id: _,
            request: _,
            tokenizer,
            prompt_token_ids,
            decode,
            emit_token_ids,
            prompt_logprobs_requested,
            generated_logprobs_requested,
            skip_special_tokens,
            output_processor,
            identity,
            cache,
            resources,
        } = tokenized;

        let event_context = EventContext {
            served_name: identity.served_name.clone(),
            description: identity.description.clone(),
            compile_duration_us,
            cache,
            resources,
            skip_special_tokens,
            metrics: Arc::clone(&self.metrics),
        };

        let stream_result: Result<ServeEventStream> = match engine_stream {
            Ok(stream) => {
                let assembly = StreamInput {
                    request_id: request_id.clone(),
                    event_context,
                    prompt_token_ids,
                    tokenizer,
                    prompt_logprobs_requested,
                    generated_logprobs_requested,
                    emit_token_ids,
                    decode_options: decode,
                    stream,
                };
                let output: ServeEventStream = match output_processor {
                    OutputProcessorPolicy::Qwen3(processor) => {
                        Box::pin(assemble_chat_event_stream(assembly, processor))
                    }
                    output_processor => Box::pin(assemble_event_stream(assembly, output_processor)),
                };
                Ok(output)
            }
            Err(error) => Err(ServeError::Engine(error)),
        };

        if let Some(terminal) = self.requests.control_terminal(&request_id) {
            if let Ok(stream) = stream_result {
                drop(stream);
            }
            self.apply_engine_control(&request_id, terminal).await;
            return Ok(self.control_event_stream(request_id, terminal));
        }
        let stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                self.metrics.failed.fetch_add(1, Ordering::Relaxed);
                self.requests.complete(
                    &request_id,
                    RequestLifecycleState::Failed,
                    compile_duration_us,
                );
                return Err(error);
            }
        };
        if !self.requests.accept(&request_id) {
            let terminal = self
                .requests
                .control_terminal(&request_id)
                .unwrap_or(LifecycleTerminal::Cancelled);
            drop(stream);
            self.apply_engine_control(&request_id, terminal).await;
            return Ok(self.control_event_stream(request_id, terminal));
        }
        self.metrics.accepted.fetch_add(1, Ordering::Relaxed);
        let stream = Box::pin(control_aware_event_stream(
            request_id.clone(),
            Arc::clone(&self.requests),
            stream,
        )) as ServeEventStream;
        Ok(Box::pin(LifecycleTrackedStream::new(
            request_id,
            stream,
            Arc::clone(&self.metrics),
            Arc::clone(&self.requests),
        )))
    }

    /// Builds an immediately terminal stream for a request controlled before engine ownership.
    fn control_event_stream(
        &self,
        request_id: ServeRequestId,
        terminal: LifecycleTerminal,
    ) -> ServeEventStream {
        let event = match terminal {
            LifecycleTerminal::Aborted => ServeEvent::Aborted {
                request_id: request_id.clone(),
            },
            LifecycleTerminal::Cancelled
            | LifecycleTerminal::Finished
            | LifecycleTerminal::Rejected
            | LifecycleTerminal::Failed => ServeEvent::Cancelled {
                request_id: request_id.clone(),
            },
        };
        let inner = Box::pin(futures::stream::iter([Ok(event)])) as ServeEventStream;
        Box::pin(LifecycleTrackedStream::new(
            request_id,
            inner,
            Arc::clone(&self.metrics),
            Arc::clone(&self.requests),
        ))
    }

    /// Cancels a live request at its acknowledged output prefix.
    pub async fn cancel(
        &self,
        request_id: impl Into<ServeRequestId>,
    ) -> std::result::Result<(), ServeControlError> {
        let request_id = request_id.into();
        let engine_may_own_request = self
            .requests
            .mark_control(&request_id, RequestLifecycleState::Cancelling);
        if engine_may_own_request {
            self.engine.cancel_request(&request_id).await?;
        }
        Ok(())
    }

    /// Aborts a live request immediately.
    pub async fn abort(
        &self,
        request_id: impl Into<ServeRequestId>,
        reason: AbortReason,
    ) -> std::result::Result<(), ServeControlError> {
        let request_id = request_id.into();
        let engine_may_own_request = self
            .requests
            .mark_control(&request_id, RequestLifecycleState::Aborting);
        if engine_may_own_request {
            self.engine.abort_request(&request_id).await?;
        }
        let _ = reason;
        Ok(())
    }

    /// Applies the engine control.
    async fn apply_engine_control(&self, request_id: &str, terminal: LifecycleTerminal) {
        let result = match terminal {
            LifecycleTerminal::Aborted => self.engine.abort_request(request_id).await,
            LifecycleTerminal::Cancelled
            | LifecycleTerminal::Finished
            | LifecycleTerminal::Rejected
            | LifecycleTerminal::Failed => self.engine.cancel_request(request_id).await,
        };
        if let Err(error) = result {
            tracing::warn!(%request_id, %error, "failed to apply pending request control after submission");
        }
    }

    /// Drains requests and shuts down the backing engine.
    pub async fn shutdown(self) -> std::result::Result<(), crate::engine_client::Error> {
        self.drain().await;
        self.engine.shutdown().await
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Caller-visible reason for aborting a live request.
pub enum AbortReason {
    /// The request owner cancelled its operation.
    OwnerCancelled,
    /// An administrative control aborted the request.
    Admin,
    /// Runtime shutdown aborted outstanding work.
    RuntimeShutdown,
}

#[derive(Debug, Error)]
/// Failure while applying an external serving control command.
pub enum ServeControlError {
    /// The backing engine rejects or fails the control command.
    #[error(transparent)]
    Engine(#[from] crate::engine_client::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Observable lifecycle phase of one serving request.
pub enum RequestLifecycleState {
    /// The public request is being validated and tokenized.
    Compiling,
    /// The tokenized request is being submitted to the engine.
    Submitting,
    /// The engine has accepted the request.
    Accepted,
    /// The engine has scheduled model execution.
    Scheduled,
    /// Output is being streamed to the caller.
    Streaming,
    /// Cancellation has been requested.
    Cancelling,
    /// Forced abortion has been requested.
    Aborting,
    /// The request completed normally.
    Finished,
    /// The request was rejected before execution.
    Rejected,
    /// The request was cancelled.
    Cancelled,
    /// The request was forcibly aborted.
    Aborted,
    /// The request failed.
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Point-in-time request counts grouped by lifecycle state.
pub struct RequestStatsSnapshot {
    /// Caller-visible request identifier.
    pub request_id: ServeRequestId,
    /// Model name used for the request.
    pub served_name: String,
    /// Stable model profile description.
    pub description: String,
    /// Latest observed lifecycle phase.
    pub state: RequestLifecycleState,
    /// Number of prompt tokens submitted to the engine.
    pub prompt_tokens: u32,
    /// Number of generated tokens included in user-visible output.
    pub visible_output_tokens: u32,
    /// Number of generated tokens consumed by internal protocol sections.
    pub internal_tokens: u32,
    /// Number of generated images.
    pub image_count: u32,
    /// Total image-generation steps completed.
    pub image_steps: u32,
    /// Cache policy and pin counts.
    pub cache: CacheAccounting,
    /// KV and media resources reserved for the request.
    pub resources: ResourceAccounting,
    /// Request phase durations.
    pub timings: RuntimeTimings,
}

impl RequestStatsSnapshot {
    /// Creates a zeroed snapshot for a request entering engine submission.
    fn submitting(
        request_id: ServeRequestId,
        served_name: String,
        description: String,
        compile_us: u64,
    ) -> Self {
        Self {
            request_id,
            served_name,
            description,
            state: RequestLifecycleState::Submitting,
            prompt_tokens: 0,
            visible_output_tokens: 0,
            internal_tokens: 0,
            image_count: 0,
            image_steps: 0,
            cache: CacheAccounting::default(),
            resources: ResourceAccounting::default(),
            timings: RuntimeTimings {
                compile_us,
                ..RuntimeTimings::default()
            },
        }
    }
}

#[derive(Default)]
struct RequestRegistryState {
    active: HashMap<ServeRequestId, RequestStatsSnapshot>,
    completed: HashMap<ServeRequestId, RequestStatsSnapshot>,
    completed_order: VecDeque<ServeRequestId>,
}

struct RuntimeRequestRegistry {
    state: Mutex<RequestRegistryState>,
    changed: Notify,
    completed_retention: usize,
}

impl Default for RuntimeRequestRegistry {
    /// Returns the default value.
    fn default() -> Self {
        Self {
            state: Mutex::new(RequestRegistryState::default()),
            changed: Notify::new(),
            completed_retention: 1024,
        }
    }
}

impl RuntimeRequestRegistry {
    /// Locks the shared state and recovers it after poisoning.
    fn lock(&self) -> std::sync::MutexGuard<'_, RequestRegistryState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Registers a unique active request and clears any retained snapshot with the same identity.
    fn register(
        &self,
        request_id: ServeRequestId,
        served_name: String,
        description: String,
        compile_us: u64,
        initial_state: RequestLifecycleState,
    ) -> bool {
        let mut state = self.lock();
        if state.active.contains_key(&request_id) {
            return false;
        }
        state.completed.remove(&request_id);
        state.completed_order.retain(|id| id != &request_id);
        let mut stats = RequestStatsSnapshot::submitting(
            request_id.clone(),
            served_name,
            description,
            compile_us,
        );
        stats.state = initial_state;
        state.active.insert(request_id.clone(), stats);
        true
    }

    /// Marks the request as submitting.
    fn mark_submitting(&self, request_id: &str, compile_us: u64) {
        if let Some(stats) = self.lock().active.get_mut(request_id) {
            if !matches!(
                stats.state,
                RequestLifecycleState::Cancelling | RequestLifecycleState::Aborting
            ) {
                stats.state = RequestLifecycleState::Submitting;
            }
            stats.timings.compile_us = compile_us;
        }
    }

    /// Accepts an engine submission into lifecycle tracking.
    fn accept(&self, request_id: &str) -> bool {
        let mut registry = self.lock();
        let Some(stats) = registry.active.get_mut(request_id) else {
            return false;
        };
        if matches!(
            stats.state,
            RequestLifecycleState::Cancelling | RequestLifecycleState::Aborting
        ) {
            return false;
        }
        stats.state = RequestLifecycleState::Accepted;
        true
    }

    /// Applies monotonic cancellation or abort state and reports whether the engine may own work.
    fn mark_control(&self, request_id: &str, state: RequestLifecycleState) -> bool {
        let (engine_may_own_request, changed) = {
            let mut registry = self.lock();
            let Some(stats) = registry.active.get_mut(request_id) else {
                return false;
            };
            let previous = stats.state;
            let next = match (previous, state) {
                (RequestLifecycleState::Aborting, _) => RequestLifecycleState::Aborting,
                (RequestLifecycleState::Cancelling, RequestLifecycleState::Aborting) => {
                    RequestLifecycleState::Aborting
                }
                (RequestLifecycleState::Cancelling, _) => RequestLifecycleState::Cancelling,
                (_, requested) => requested,
            };
            let changed = next != previous;
            stats.state = next;
            (
                previous != RequestLifecycleState::Compiling && changed,
                changed,
            )
        };
        if changed {
            self.changed.notify_waiters();
        }
        engine_may_own_request
    }

    /// Returns the accepted control command terminal event.
    fn control_terminal(&self, request_id: &str) -> Option<LifecycleTerminal> {
        match self.lock().active.get(request_id).map(|stats| stats.state) {
            Some(RequestLifecycleState::Cancelling) => Some(LifecycleTerminal::Cancelled),
            Some(RequestLifecycleState::Aborting) => Some(LifecycleTerminal::Aborted),
            _ => None,
        }
    }

    /// Waits the for control.
    async fn wait_for_control(&self, request_id: &str) -> LifecycleTerminal {
        loop {
            let changed = self.changed.notified();
            if let Some(terminal) = self.control_terminal(request_id) {
                return terminal;
            }
            changed.await;
        }
    }

    /// Returns the terminal event emitted when the guard is dropped.
    fn drop_terminal(&self, request_id: &str) -> LifecycleTerminal {
        match self.lock().active.get(request_id).map(|stats| stats.state) {
            Some(RequestLifecycleState::Aborting) => LifecycleTerminal::Aborted,
            _ => LifecycleTerminal::Cancelled,
        }
    }

    /// Incorporates one serving event into request statistics and surfaces pending control.
    fn observe(
        &self,
        request_id: &str,
        event: &ServeEvent,
        elapsed_us: u64,
    ) -> Option<LifecycleTerminal> {
        let mut state = self.lock();
        let stats = state.active.get_mut(request_id)?;

        // External cancellation and abort state dominates every later stream
        // observation and asks the caller to terminate the producer.
        match stats.state {
            RequestLifecycleState::Cancelling => return Some(LifecycleTerminal::Cancelled),
            RequestLifecycleState::Aborting => return Some(LifecycleTerminal::Aborted),
            _ => {}
        }

        // Each event updates only the lifecycle dimensions it authoritatively
        // carries; the final usage event reconciles all cumulative counters.
        match event {
            ServeEvent::Accepted {
                compile_duration_us,
                prompt_token_count,
                ..
            } => {
                stats.state = RequestLifecycleState::Accepted;
                stats.prompt_tokens = (*prompt_token_count).min(u32::MAX as usize) as u32;
                stats.timings.compile_us = *compile_duration_us;
            }
            ServeEvent::Scheduled {
                queued_at,
                scheduled_at,
                cache,
                resources,
                ..
            } => {
                stats.state = RequestLifecycleState::Scheduled;
                stats.cache = cache.clone();
                stats.resources = resources.clone();
                stats.timings.queue_us =
                    (*queued_at).zip(*scheduled_at).map(|(queued, scheduled)| {
                        ((scheduled - queued).max(0.0) * 1_000_000.0) as u64
                    });
            }
            ServeEvent::TextDelta { token_ids, .. } => {
                stats.state = RequestLifecycleState::Streaming;
                stats.visible_output_tokens = stats
                    .visible_output_tokens
                    .saturating_add(token_ids.len().min(u32::MAX as usize) as u32);
                stats
                    .timings
                    .first_visible_output_us
                    .get_or_insert(elapsed_us);
            }
            // Internal tokens affect usage but never establish visible-output latency.
            ServeEvent::InternalTextDelta { .. } => {
                stats.state = RequestLifecycleState::Streaming;
                stats.internal_tokens = stats.internal_tokens.saturating_add(1);
            }
            ServeEvent::ReasoningDelta { .. }
            | ServeEvent::OutputBlockStart { .. }
            | ServeEvent::OutputBlockEnd { .. }
            | ServeEvent::ToolCallStart { .. }
            | ServeEvent::ToolCallArgumentsDelta { .. }
            | ServeEvent::ToolCallEnd { .. } => {
                stats.state = RequestLifecycleState::Streaming;
            }
            // Image begin and commit are public media progress boundaries.
            ServeEvent::ImageBegin {
                elapsed_us: event_elapsed,
                ..
            }
            | ServeEvent::ImageCommit {
                elapsed_us: event_elapsed,
                ..
            } => {
                stats.state = RequestLifecycleState::Streaming;
                stats
                    .timings
                    .first_visible_output_us
                    .get_or_insert(*event_elapsed);
            }
            ServeEvent::ImageStep { .. } => {
                stats.state = RequestLifecycleState::Streaming;
                stats.image_steps = stats.image_steps.saturating_add(1);
            }
            ServeEvent::ImageDone { .. } => {
                stats.state = RequestLifecycleState::Streaming;
                stats.image_count = stats.image_count.saturating_add(1);
            }
            // Terminal usage is the canonical aggregate produced by the assembler.
            ServeEvent::Usage {
                prompt_tokens,
                visible_output_tokens,
                internal_tokens,
                image_count,
                image_steps,
                cache,
                resources,
                timings,
            } => {
                stats.prompt_tokens = *prompt_tokens;
                stats.visible_output_tokens = *visible_output_tokens;
                stats.internal_tokens = *internal_tokens;
                stats.image_count = *image_count;
                stats.image_steps = *image_steps;
                stats.cache = cache.clone();
                stats.resources = resources.clone();
                stats.timings = timings.clone();
            }
            // Completion transitions are applied by `complete` after observation.
            ServeEvent::Finished { .. }
            | ServeEvent::Rejected { .. }
            | ServeEvent::Cancelled { .. }
            | ServeEvent::Aborted { .. }
            | ServeEvent::Failed { .. } => {}
        }
        None
    }

    /// Moves an active request into retained terminal history with final timing.
    fn complete(
        &self,
        request_id: &str,
        terminal: RequestLifecycleState,
        elapsed_us: u64,
    ) -> Option<RequestLifecycleState> {
        let mut state = self.lock();
        let mut stats = state.active.remove(request_id)?;
        let actual_terminal = match stats.state {
            RequestLifecycleState::Cancelling => RequestLifecycleState::Cancelled,
            RequestLifecycleState::Aborting => RequestLifecycleState::Aborted,
            _ => terminal,
        };
        stats.state = actual_terminal;
        stats.timings.total_us = stats.timings.total_us.max(elapsed_us);
        Self::insert_completed(&mut state, stats, self.completed_retention);
        drop(state);
        self.changed.notify_waiters();
        Some(actual_terminal)
    }

    /// Inserts the completed.
    fn insert_completed(
        state: &mut RequestRegistryState,
        stats: RequestStatsSnapshot,
        retention: usize,
    ) {
        let request_id = stats.request_id.clone();
        state.completed_order.retain(|id| id != &request_id);
        state.completed.insert(request_id.clone(), stats);
        state.completed_order.push_back(request_id);
        while state.completed_order.len() > retention {
            if let Some(expired) = state.completed_order.pop_front() {
                state.completed.remove(&expired);
            }
        }
    }

    /// Returns a snapshot of the current statistics.
    fn stats(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        let state = self.lock();
        state
            .active
            .get(request_id)
            .or_else(|| state.completed.get(request_id))
            .cloned()
    }

    /// Returns the number of active requests.
    fn active_count(&self) -> usize {
        self.lock().active.len()
    }

    /// Waits for a request to leave the active registry and returns its final statistics.
    async fn drain_request(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        loop {
            let changed = self.changed.notified();
            if !self.lock().active.contains_key(request_id) {
                return self.stats(request_id);
            }
            changed.await;
        }
    }

    /// Drains completed entries from the tracker.
    async fn drain(&self) {
        loop {
            let changed = self.changed.notified();
            if self.active_count() == 0 {
                return;
            }
            changed.await;
        }
    }
}

#[derive(Debug, Clone)]
struct EventContext {
    served_name: String,
    description: String,
    compile_duration_us: u64,
    cache: CacheAccounting,
    resources: ResourceAccounting,
    skip_special_tokens: bool,
    metrics: Arc<RuntimeLifecycleMetrics>,
}

struct StreamInput {
    request_id: ServeRequestId,
    event_context: EventContext,
    prompt_token_ids: Vec<u32>,
    tokenizer: crate::serving::text::tokenizer::DynTokenizer,
    prompt_logprobs_requested: bool,
    generated_logprobs_requested: bool,
    emit_token_ids: bool,
    decode_options: TextDecodeOptions,
    stream: EventRx,
}

#[derive(Debug, Default)]
struct RuntimeLifecycleMetrics {
    accepted: AtomicU64,
    scheduled: AtomicU64,
    finished: AtomicU64,
    rejected: AtomicU64,
    cancelled: AtomicU64,
    aborted: AtomicU64,
    failed: AtomicU64,
}

impl RuntimeLifecycleMetrics {
    /// Captures the current state as an immutable snapshot.
    fn snapshot(&self) -> RuntimeMetricsSnapshot {
        RuntimeMetricsSnapshot {
            active: 0,
            accepted: self.accepted.load(Ordering::Relaxed),
            scheduled: self.scheduled.load(Ordering::Relaxed),
            finished: self.finished.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Relaxed),
            aborted: self.aborted.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Aggregate request, cache, resource, and timing metrics.
pub struct RuntimeMetricsSnapshot {
    /// Number of requests presently tracked by the runtime.
    pub active: u64,
    /// Cumulative number of engine-accepted requests.
    pub accepted: u64,
    /// Cumulative number of scheduled requests.
    pub scheduled: u64,
    /// Cumulative number of normally completed requests.
    pub finished: u64,
    /// Cumulative number of requests rejected before execution.
    pub rejected: u64,
    /// Cumulative number of cancelled requests.
    pub cancelled: u64,
    /// Cumulative number of forcibly aborted requests.
    pub aborted: u64,
    /// Cumulative number of failed requests.
    pub failed: u64,
}

impl RuntimeMetricsSnapshot {
    /// Returns stable lifecycle state names and values exported by the public metrics
    /// route.
    pub const fn state_counts(self) -> [(&'static str, u64); 8] {
        [
            ("active", self.active),
            ("accepted", self.accepted),
            ("scheduled", self.scheduled),
            ("finished", self.finished),
            ("rejected", self.rejected),
            ("cancelled", self.cancelled),
            ("aborted", self.aborted),
            ("failed", self.failed),
        ]
    }
}

struct LifecycleGuard {
    request_id: ServeRequestId,
    metrics: Arc<RuntimeLifecycleMetrics>,
    requests: Arc<RuntimeRequestRegistry>,
    started: Instant,
    terminal: bool,
}

struct LifecycleTrackedStream {
    inner: ServeEventStream,
    lifecycle: LifecycleGuard,
}

impl LifecycleTrackedStream {
    /// Creates an initialized serving runtime component.
    fn new(
        request_id: ServeRequestId,
        inner: ServeEventStream,
        metrics: Arc<RuntimeLifecycleMetrics>,
        requests: Arc<RuntimeRequestRegistry>,
    ) -> Self {
        Self {
            inner,
            lifecycle: LifecycleGuard::new(request_id, metrics, requests),
        }
    }
}

impl Stream for LifecycleTrackedStream {
    type Item = Result<ServeEvent>;

    /// Polls the inner stream while enforcing control precedence and terminal accounting.
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(mut event))) => {
                let elapsed_us = self.lifecycle.started.elapsed().as_micros() as u64;
                if let Some(control) =
                    self.lifecycle
                        .requests
                        .observe(&self.lifecycle.request_id, &event, elapsed_us)
                {
                    let terminal = self.lifecycle.terminal(control, elapsed_us);
                    event = control_terminal_event(&self.lifecycle.request_id, terminal);
                    return Poll::Ready(Some(Ok(event)));
                }
                let terminal = match &event {
                    ServeEvent::Finished { .. } => Some(LifecycleTerminal::Finished),
                    ServeEvent::Rejected { .. } => Some(LifecycleTerminal::Rejected),
                    ServeEvent::Cancelled { .. } => Some(LifecycleTerminal::Cancelled),
                    ServeEvent::Aborted { .. } => Some(LifecycleTerminal::Aborted),
                    ServeEvent::Failed { .. } => Some(LifecycleTerminal::Failed),
                    _ => None,
                };
                if let Some(terminal) = terminal {
                    let actual = self.lifecycle.terminal(terminal, elapsed_us);
                    if actual != terminal {
                        event = control_terminal_event(&self.lifecycle.request_id, actual);
                    }
                }
                Poll::Ready(Some(Ok(event)))
            }
            Poll::Ready(Some(Err(error))) => {
                let elapsed_us = self.lifecycle.started.elapsed().as_micros() as u64;
                let actual = self
                    .lifecycle
                    .terminal(LifecycleTerminal::Failed, elapsed_us);
                if actual == LifecycleTerminal::Failed {
                    Poll::Ready(Some(Err(error)))
                } else {
                    Poll::Ready(Some(Ok(control_terminal_event(
                        &self.lifecycle.request_id,
                        actual,
                    ))))
                }
            }
            Poll::Ready(None) => {
                if !self.lifecycle.terminal {
                    let elapsed_us = self.lifecycle.started.elapsed().as_micros() as u64;
                    let actual = self
                        .lifecycle
                        .terminal(LifecycleTerminal::Failed, elapsed_us);
                    if actual != LifecycleTerminal::Failed {
                        return Poll::Ready(Some(Ok(control_terminal_event(
                            &self.lifecycle.request_id,
                            actual,
                        ))));
                    }
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl LifecycleGuard {
    /// Creates an initialized serving runtime component.
    fn new(
        request_id: ServeRequestId,
        metrics: Arc<RuntimeLifecycleMetrics>,
        requests: Arc<RuntimeRequestRegistry>,
    ) -> Self {
        Self {
            request_id,
            metrics,
            requests,
            started: Instant::now(),
            terminal: false,
        }
    }

    /// Records the first terminal transition and increments its aggregate metric exactly once.
    fn terminal(&mut self, kind: LifecycleTerminal, elapsed_us: u64) -> LifecycleTerminal {
        if self.terminal {
            return kind;
        }
        let actual = self
            .requests
            .complete(&self.request_id, kind.into(), elapsed_us)
            .and_then(lifecycle_terminal_for_state)
            .unwrap_or(kind);
        let counter = match actual {
            LifecycleTerminal::Finished => &self.metrics.finished,
            LifecycleTerminal::Rejected => &self.metrics.rejected,
            LifecycleTerminal::Cancelled => &self.metrics.cancelled,
            LifecycleTerminal::Aborted => &self.metrics.aborted,
            LifecycleTerminal::Failed => &self.metrics.failed,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        self.terminal = true;
        actual
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LifecycleTerminal {
    Finished,
    Rejected,
    Cancelled,
    Aborted,
    Failed,
}

/// Returns the terminal lifecycle event represented by a state.
fn lifecycle_terminal_for_state(state: RequestLifecycleState) -> Option<LifecycleTerminal> {
    match state {
        RequestLifecycleState::Finished => Some(LifecycleTerminal::Finished),
        RequestLifecycleState::Rejected => Some(LifecycleTerminal::Rejected),
        RequestLifecycleState::Cancelled | RequestLifecycleState::Cancelling => {
            Some(LifecycleTerminal::Cancelled)
        }
        RequestLifecycleState::Aborted | RequestLifecycleState::Aborting => {
            Some(LifecycleTerminal::Aborted)
        }
        RequestLifecycleState::Failed => Some(LifecycleTerminal::Failed),
        RequestLifecycleState::Compiling
        | RequestLifecycleState::Submitting
        | RequestLifecycleState::Accepted
        | RequestLifecycleState::Scheduled
        | RequestLifecycleState::Streaming => None,
    }
}

/// Builds the terminal event for a control outcome.
fn control_terminal_event(request_id: &ServeRequestId, terminal: LifecycleTerminal) -> ServeEvent {
    match terminal {
        LifecycleTerminal::Aborted => ServeEvent::Aborted {
            request_id: request_id.clone(),
        },
        LifecycleTerminal::Cancelled => ServeEvent::Cancelled {
            request_id: request_id.clone(),
        },
        LifecycleTerminal::Finished | LifecycleTerminal::Rejected | LifecycleTerminal::Failed => {
            unreachable!("only cancellation and abort can replace an engine terminal")
        }
    }
}

impl From<LifecycleTerminal> for RequestLifecycleState {
    /// Converts the source value into this type.
    fn from(value: LifecycleTerminal) -> Self {
        match value {
            LifecycleTerminal::Finished => Self::Finished,
            LifecycleTerminal::Rejected => Self::Rejected,
            LifecycleTerminal::Cancelled => Self::Cancelled,
            LifecycleTerminal::Aborted => Self::Aborted,
            LifecycleTerminal::Failed => Self::Failed,
        }
    }
}

impl Drop for LifecycleGuard {
    /// Releases resources owned by this value.
    fn drop(&mut self) {
        if !self.terminal {
            let elapsed_us = self.started.elapsed().as_micros() as u64;
            let terminal = self.requests.drop_terminal(&self.request_id);
            let _ = self.terminal(terminal, elapsed_us);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Prefix and encoder cache activity for a request or interval.
pub struct CacheAccounting {
    /// Whether the request may reuse existing cache entries.
    pub read_enabled: bool,
    /// Whether the request may publish reusable cache entries.
    pub write_enabled: bool,
    /// Number of encoder-cache entries pinned for the request lifetime.
    pub encoder_pin_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Allocated KV, latent, and buffer capacity.
pub struct ResourceAccounting {
    /// KV-token capacity required for the request.
    pub expected_kv_tokens: u64,
    /// Latent image units reserved for generation.
    pub image_latent_units: u64,
    /// Number of encoder-cache entries retained by the request.
    pub encoder_cache_pins: usize,
    /// Whether the engine can reconstruct the request after worker recovery.
    pub replayable: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Queue, execution, and end-to-end duration totals.
pub struct RuntimeTimings {
    /// Request validation and tokenization duration in microseconds.
    pub compile_us: u64,
    /// Queue duration in microseconds, when scheduling timestamps are available.
    pub queue_us: Option<u64>,
    /// Time from admission to the first visible output in microseconds.
    pub first_visible_output_us: Option<u64>,
    /// End-to-end request duration in microseconds.
    pub total_us: u64,
}

/// Runtime event consumed by the HTTP response layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ServeEvent {
    /// The engine accepted a compiled request.
    Accepted {
        /// Caller-visible request identifier.
        request_id: ServeRequestId,
        /// Model name used for the request.
        served_name: String,
        /// Stable model profile description.
        description: String,
        /// Validation and tokenization duration in microseconds.
        compile_duration_us: u64,
        /// Number of prompt tokens submitted to the engine.
        prompt_token_count: usize,
        /// Prompt token identifiers submitted to the engine.
        prompt_token_ids: Vec<u32>,
        /// Per-position prompt log probabilities, when requested.
        prompt_logprobs: Option<DecodedPromptLogprobs>,
    },
    /// The engine scheduled the request for model execution.
    Scheduled {
        /// Caller-visible request identifier.
        request_id: ServeRequestId,
        /// Monotonic timestamp at which the request entered the serving queue.
        queued_at: Option<f64>,
        /// Monotonic timestamp at which model execution began.
        scheduled_at: Option<f64>,
        /// Cache policy and pin counts.
        cache: CacheAccounting,
        /// KV and media resources reserved for the request.
        resources: ResourceAccounting,
    },
    /// Newly decoded user-visible text.
    TextDelta {
        /// Output candidate receiving the text.
        candidate_id: CandidateId,
        /// Newly visible decoded text.
        text: String,
        /// Token identifiers represented by this update.
        token_ids: Vec<u32>,
        /// Per-position candidate log probabilities, when requested.
        logprobs: Option<DecodedLogprobs>,
    },
    /// Newly decoded text consumed by a model protocol rather than exposed as output.
    InternalTextDelta {
        /// Output candidate receiving the text.
        candidate_id: CandidateId,
        /// Newly decoded internal text.
        text: String,
    },
    /// Newly decoded structured reasoning text.
    ReasoningDelta {
        /// Output candidate receiving the reasoning.
        candidate_id: CandidateId,
        /// Newly decoded reasoning text.
        text: String,
    },
    /// A structured assistant output block has opened.
    OutputBlockStart {
        /// Output candidate containing the block.
        candidate_id: CandidateId,
        /// Zero-based content-block index.
        index: usize,
        /// Semantic kind of the opened block.
        kind: AssistantBlockKind,
    },
    /// A structured assistant output block has closed.
    OutputBlockEnd {
        /// Output candidate containing the block.
        candidate_id: CandidateId,
        /// Zero-based content-block index.
        index: usize,
        /// Complete normalized block content.
        block: AssistantContentBlock,
    },
    /// A structured function-tool call has opened.
    ToolCallStart {
        /// Output candidate containing the tool call.
        candidate_id: CandidateId,
        /// Zero-based tool-call index.
        index: usize,
        /// Request-local tool-call identifier.
        id: String,
        /// Selected function name.
        name: String,
    },
    /// Newly decoded serialized tool arguments.
    ToolCallArgumentsDelta {
        /// Output candidate containing the tool call.
        candidate_id: CandidateId,
        /// Zero-based tool-call index.
        index: usize,
        /// Newly decoded argument text.
        delta: String,
    },
    /// A structured function-tool call has closed.
    ToolCallEnd {
        /// Output candidate containing the tool call.
        candidate_id: CandidateId,
        /// Zero-based tool-call index.
        index: usize,
        /// Request-local tool-call identifier.
        id: String,
        /// Selected function name.
        name: String,
        /// Complete serialized function arguments.
        arguments: String,
    },
    /// Generation of one image has begun.
    ImageBegin {
        /// Output candidate containing the image.
        candidate_id: CandidateId,
        /// Request-local image identifier.
        image_id: String,
        /// Output width in pixels, when known.
        width: Option<u32>,
        /// Output height in pixels, when known.
        height: Option<u32>,
        /// Planned diffusion step count, when known.
        steps: Option<u32>,
        /// Elapsed request duration in microseconds.
        elapsed_us: u64,
    },
    /// One image-generation step has completed.
    ImageStep {
        /// Output candidate containing the image.
        candidate_id: CandidateId,
        /// Request-local image identifier.
        image_id: String,
        /// Completed step number.
        step: u32,
        /// Elapsed request duration in microseconds.
        elapsed_us: u64,
    },
    /// Generated image state has been committed to the model context.
    ImageCommit {
        /// Output candidate containing the image.
        candidate_id: CandidateId,
        /// Request-local image identifier.
        image_id: String,
        /// Elapsed request duration in microseconds.
        elapsed_us: u64,
    },
    /// One generated image is complete and available to the response layer.
    ImageDone {
        /// Output candidate containing the image.
        candidate_id: CandidateId,
        /// Request-local image identifier.
        image_id: String,
        /// Output width in pixels, when reported.
        width: Option<u32>,
        /// Output height in pixels, when reported.
        height: Option<u32>,
        /// Encoded image size in bytes, when reported.
        bytes: Option<u64>,
        /// SHA-256 digest of the encoded image, when reported.
        sha256: Option<String>,
        /// Base64-encoded PNG payload, when retained for transport.
        pixels_png_b64: Option<String>,
        /// Elapsed request duration in microseconds.
        elapsed_us: u64,
    },
    /// Final request usage and timing totals.
    Usage {
        /// Number of prompt tokens submitted to the engine.
        prompt_tokens: u32,
        /// Number of generated tokens included in user-visible output.
        visible_output_tokens: u32,
        /// Number of generated tokens consumed by internal protocol sections.
        internal_tokens: u32,
        /// Number of generated images.
        image_count: u32,
        /// Total image-generation steps completed.
        image_steps: u32,
        /// Cache policy and pin counts.
        cache: CacheAccounting,
        /// KV and media resources reserved for the request.
        resources: ResourceAccounting,
        /// Request phase durations.
        timings: RuntimeTimings,
    },
    /// The output candidate reached a terminal condition.
    Finished {
        /// Completed output candidate.
        candidate_id: CandidateId,
        /// Semantic terminal status.
        reason: FinishStatus,
        /// Optional model- or engine-provided terminal detail.
        finish_detail: Option<String>,
    },
    /// The request was rejected before execution.
    Rejected {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
        /// Caller-visible rejection description.
        message: String,
    },
    /// The request was cancelled.
    Cancelled {
        /// Identifier of the cancelled request.
        request_id: ServeRequestId,
    },
    /// The request was forcibly aborted.
    Aborted {
        /// Identifier of the aborted request.
        request_id: ServeRequestId,
    },
    /// The request failed.
    Failed {
        /// Identifier of the failed request.
        request_id: ServeRequestId,
        /// Caller-visible failure description.
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Semantic terminal status of a completed serving stream.
pub enum FinishStatus {
    /// Generation reached a normal stop condition.
    Stop {
        /// Concrete stop cause, when one was observed.
        cause: Option<StopCause>,
    },
    /// The configured generation-length limit was reached.
    Length,
    /// The request was cancelled or aborted.
    Abort,
    /// Generation failed.
    Error,
    /// Repetition detection terminated generation.
    Repetition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
/// Concrete token or string that triggered a stop.
pub enum StopCause {
    /// The model emitted an end-of-sequence token.
    Eos,
    /// The sampler emitted a configured stop-token identifier.
    TokenId(u32),
    /// Decoded output matched a configured stop string.
    Text(String),
}

impl From<&FinishReason> for FinishStatus {
    /// Converts the source value into this type.
    fn from(reason: &FinishReason) -> Self {
        match reason.reason() {
            uniserve_core::FinishReason::Eos
            | uniserve_core::FinishReason::Completed
            | uniserve_core::FinishReason::Stop
            | uniserve_core::FinishReason::ImageDone => Self::Stop {
                cause: reason.as_stop_reason().map(|value| match value {
                    StopReason::TokenId(id) => StopCause::TokenId(*id),
                    StopReason::Text(text) => StopCause::Text(text.clone()),
                }),
            },
            uniserve_core::FinishReason::MaxTokens => Self::Length,
            uniserve_core::FinishReason::Cancelled | uniserve_core::FinishReason::Aborted => {
                Self::Abort
            }
            uniserve_core::FinishReason::Error => Self::Error,
            uniserve_core::FinishReason::Repetition => Self::Repetition,
        }
    }
}

#[try_stream]
/// Forwards events until the stream ends or an external control reaches terminal precedence.
async fn control_aware_event_stream(
    request_id: ServeRequestId,
    requests: Arc<RuntimeRequestRegistry>,
    mut stream: ServeEventStream,
    mut y: TryYielder<ServeEvent, ServeError>,
) -> Result<()> {
    loop {
        if let Some(terminal) = requests.control_terminal(&request_id) {
            y.yield_ok(control_terminal_event(&request_id, terminal))
                .await;
            return Ok(());
        }
        tokio::select! {
            next = stream.next() => match next {
                Some(Ok(event)) => {
                    if let Some(terminal) = requests.control_terminal(&request_id) {
                        y.yield_ok(control_terminal_event(&request_id, terminal)).await;
                        return Ok(());
                    }
                    let terminal = matches!(
                        event,
                        ServeEvent::Finished { .. }
                            | ServeEvent::Rejected { .. }
                            | ServeEvent::Cancelled { .. }
                            | ServeEvent::Aborted { .. }
                            | ServeEvent::Failed { .. }
                    );
                    y.yield_ok(event).await;
                    if terminal {
                        return Ok(());
                    }
                }
                Some(Err(error)) => return Err(error),
                None => return Ok(()),
            },
            terminal = requests.wait_for_control(&request_id) => {
                y.yield_ok(control_terminal_event(&request_id, terminal)).await;
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::TokenLogprob;
    use uniserve_engine::EventRx;

    fn event_context() -> EventContext {
        EventContext {
            served_name: "profile".to_string(),
            description: "bagel".to_string(),
            compile_duration_us: 7,
            cache: CacheAccounting {
                read_enabled: true,
                write_enabled: true,
                encoder_pin_count: 0,
            },
            resources: ResourceAccounting {
                expected_kv_tokens: 2,
                replayable: true,
                ..ResourceAccounting::default()
            },
            skip_special_tokens: false,
            metrics: Arc::new(RuntimeLifecycleMetrics::default()),
        }
    }

    fn bagel_output_policy() -> OutputProcessorPolicy {
        OutputProcessorPolicy::Bagel
    }

    #[tokio::test]
    async fn assembler_attaches_ranked_logprobs_to_text_delta() {
        let tokenizer = crate::serving::test_support::configured_tokenizer();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(Event::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        tx.try_send(Event::TextToken {
            id: b'a' as u32,
            logprob: Some(-0.25),
        })
        .unwrap();
        tx.try_send(Event::TokenLogprobs {
            id: b'a' as u32,
            candidates: vec![TokenLogprob {
                token_id: b'a' as u32,
                logprob: -0.25,
                rank: 3,
            }],
        })
        .unwrap();
        tx.try_send(Event::Finished {
            reason: uniserve_core::FinishReason::MaxTokens,
            stop_reason: None,
            prompt_tokens: 1,
            completion_tokens: 1,
            images: 0,
        })
        .unwrap();
        drop(tx);

        let events = assemble_event_stream(
            StreamInput {
                request_id: "req".into(),
                event_context: event_context(),
                prompt_token_ids: vec![b'p' as u32],
                tokenizer,
                prompt_logprobs_requested: false,
                generated_logprobs_requested: true,
                emit_token_ids: true,
                decode_options: TextDecodeOptions::default(),
                stream: EventRx::from_receiver(rx),
            },
            bagel_output_policy(),
        )
        .collect::<Vec<_>>()
        .await;
        let delta = events
            .iter()
            .find_map(|event| match event {
                Ok(ServeEvent::TextDelta {
                    text,
                    token_ids,
                    logprobs,
                    ..
                }) => Some((text, token_ids, logprobs.as_ref())),
                _ => None,
            })
            .expect("text delta");
        assert_eq!(delta.0, "a");
        assert_eq!(delta.1, &[b'a' as u32]);
        let candidate = &delta.2.expect("decoded logprobs").positions[0].entries[0];
        assert_eq!(candidate.token_id, b'a' as u32);
        assert_eq!(candidate.rank, 3);
        assert!(events.iter().all(std::result::Result::is_ok));
        assert!(matches!(
            events.last(),
            Some(Ok(ServeEvent::Finished {
                reason: FinishStatus::Length,
                ..
            }))
        ));
    }

    #[tokio::test]
    async fn assembler_publishes_an_image_with_its_next_text_token() {
        let tokenizer = crate::serving::test_support::configured_tokenizer();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(Event::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();

        let events = assemble_event_stream(
            StreamInput {
                request_id: "feedback-output".into(),
                event_context: event_context(),
                prompt_token_ids: vec![b'p' as u32],
                tokenizer,
                prompt_logprobs_requested: false,
                generated_logprobs_requested: false,
                emit_token_ids: false,
                decode_options: TextDecodeOptions::default(),
                stream: EventRx::from_receiver(rx),
            },
            bagel_output_policy(),
        );
        tokio::pin!(events);
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::Accepted { .. }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::Scheduled { .. }))
        ));

        tx.send(Event::TextToken {
            id: b'a' as u32,
            logprob: None,
        })
        .await
        .unwrap();
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::TextDelta { text, .. })) if text == "a"
        ));

        tx.send(Event::ImageCommit { image_id: 0 }).await.unwrap();
        tx.send(Event::ImageDone {
            image_id: 0,
            height: 1,
            width: 1,
            bytes: 3,
            sha256: "0".repeat(64),
            pixels_png_b64: "cG5n".to_string(),
        })
        .await
        .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), events.next())
                .await
                .is_err(),
            "the image became public before a continuation token arrived"
        );

        tx.send(Event::TextToken {
            id: b'b' as u32,
            logprob: None,
        })
        .await
        .unwrap();
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::ImageCommit { .. }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::ImageDone { .. }))
        ));
    }
}
