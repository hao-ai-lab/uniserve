//! Canonical serving runtime.
//!
//! The public funnel is a single ownership chain:
//! `GenerateReqInput -> ResolvedModel::tokenize -> TokenizedGenerateReqInput ->
//! EngineClient::submit_generation -> ServeEvent stream`. There is one
//! internal admission value, one model-owned tokenize arrow, one engine
//! submission per request, and one stream assembler.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod assembly;
pub mod chat;
mod input;
mod model;
mod omni;
#[cfg(test)]
mod test_support;
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

use crate::engine_client::{EngineClient, MediaEventRx, MediaSubmission, StreamCancelCause};
use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Notify;
use uniserve_core::{GenerationEvent, PublicCommit};
use uniserve_engine::EventRx;

pub use input::{
    CacheBounds, DecodeControls, GenerateReqInput, ImageGenControls, ImageInput, ModalitySelection,
    ModelEventIdentity, OutputDetail, OutputProcessorPolicy, PromptInput, SamplingConfig,
    SchedulingBounds, StopConfig, TokenizedGenerateReqInput,
};
pub use model::{
    ResolvedAssets, ResolvedModel, ServedEndpoint, ServedFeature, ServedModality,
    ServedModelCapabilities, ServedSamplingControl,
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
pub struct ServeRequestId(String);

impl ServeRequestId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for ServeRequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Deref for ServeRequestId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<str> for ServeRequestId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for ServeRequestId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl From<String> for ServeRequestId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ServeRequestId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<ServeRequestId> for String {
    fn from(value: ServeRequestId) -> Self {
        value.0
    }
}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct CandidateId(u32);

impl CandidateId {
    pub const PRIMARY: Self = Self(0);

    pub const fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for CandidateId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<CandidateId> for u32 {
    fn from(value: CandidateId) -> Self {
        value.0
    }
}

pub type ServeEventStream = Pin<Box<dyn Stream<Item = Result<ServeEvent>> + Send>>;
pub type Result<T> = std::result::Result<T, ServeError>;

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Error)]
pub enum TokenizeError {
    #[error(transparent)]
    Tokenizer(#[from] crate::profile::tokenizer::TokenizerError),
    #[error(transparent)]
    Chat(#[from] crate::serving::chat::Error),
    #[error(transparent)]
    Text(#[from] crate::serving::text::Error),
    #[error(transparent)]
    Sampling(#[from] uniserve_core::SamplingParamsError),
    #[error(transparent)]
    Generation(#[from] uniserve_core::GenerationRequestError),
    #[error(transparent)]
    Omni(#[from] crate::serving::omni::OmniError),
    #[error("{field} must be non-negative or -1, got {value}")]
    InvalidLogprobCount { field: &'static str, value: i32 },
    #[error("min_tokens ({min_tokens}) exceeds max_tokens ({max_tokens})")]
    MinTokensExceedsMaximum { min_tokens: u32, max_tokens: u32 },
    #[error("this model does not support logprobs")]
    UnsupportedLogprobs,
    #[error("{0}")]
    Invalid(String),
    #[error("tokenization task failed")]
    Task(#[source] tokio::task::JoinError),
}

#[derive(Debug, Error)]
pub enum OutputProcessingError {
    #[error(transparent)]
    Tokenizer(#[from] crate::profile::tokenizer::TokenizerError),
    #[error(transparent)]
    Chat(#[from] crate::serving::chat::Error),
    #[error(transparent)]
    Text(#[from] crate::serving::text::Error),
    #[error(transparent)]
    Reasoning(#[from] crate::profile::reasoning::ReasoningError),
    #[error("{0}")]
    Malformed(String),
}

/// Runtime-local serving errors.
#[derive(Debug, Error)]
pub enum ServeError {
    #[error("request `{request_id}` asks for {requested} outputs; only one output is supported")]
    UnsupportedOutputCount {
        request_id: ServeRequestId,
        requested: u32,
    },
    #[error("request `{request_id}` requires unsupported capability `{capability}`")]
    UnsupportedCapability {
        request_id: ServeRequestId,
        capability: &'static str,
    },
    #[error(
        "request `{request_id}` has {prompt_tokens} prompt tokens, exceeding the {max_tokens}-token profile limit"
    )]
    ContextLengthExceeded {
        request_id: ServeRequestId,
        prompt_tokens: usize,
        max_tokens: u32,
    },
    #[error(
        "request `{request_id}` requires {required_tokens} KV tokens, exceeding the {max_tokens}-token runtime limit"
    )]
    ContextCapacityExceeded {
        request_id: ServeRequestId,
        required_tokens: u64,
        max_tokens: u32,
    },
    #[error("request `{request_id}` is already active")]
    DuplicateRequestId { request_id: ServeRequestId },
    #[error(transparent)]
    ModelResolution(#[from] crate::serving::model::ModelResolutionError),
    #[error(transparent)]
    Engine(#[from] crate::engine_client::Error),
    #[error("request `{request_id}` cannot be tokenized")]
    Tokenize {
        request_id: ServeRequestId,
        #[source]
        source: TokenizeError,
    },
    #[error("request `{request_id}` output processing failed")]
    OutputProcessing {
        request_id: ServeRequestId,
        #[source]
        source: OutputProcessingError,
    },
}

fn malformed_output(
    request_id: impl Into<ServeRequestId>,
    message: impl Into<String>,
) -> ServeError {
    ServeError::OutputProcessing {
        request_id: request_id.into(),
        source: OutputProcessingError::Malformed(message.into()),
    }
}

/// Fold cache namespace/salt into a stable isolation key.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoGenerationInput {
    pub request_id: ServeRequestId,
    pub prompt: String,
    pub seed: u64,
    pub output_path: String,
}

impl ServingRuntime {
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

    /// The resolved model owned by this runtime.
    pub fn model(&self) -> &ResolvedModel {
        &self.model
    }

    /// The friendly served-model name.
    pub fn served_model_name(&self) -> &str {
        self.model.served_model_name()
    }

    pub fn engine(&self) -> &EngineClient {
        &self.engine
    }

    pub async fn generate_video(&self, request: VideoGenerationInput) -> Result<MediaEventRx> {
        if !self
            .model
            .served_capabilities()
            .endpoints
            .contains(&ServedEndpoint::VideoGenerations)
        {
            return Err(ServeError::UnsupportedCapability {
                request_id: request.request_id,
                capability: "video_generation",
            });
        }
        if request.prompt.trim().is_empty() {
            return Err(ServeError::Tokenize {
                request_id: request.request_id,
                source: TokenizeError::Invalid("video prompt must not be empty".to_string()),
            });
        }
        let submission = MediaSubmission::new(
            request.request_id.to_string(),
            request.prompt,
            request.seed,
            request.output_path,
        );
        self.engine
            .submit_media(submission)
            .await
            .map_err(ServeError::Engine)
    }

    pub fn runtime_id(&self) -> u64 {
        self.runtime_id
    }

    pub fn metrics_snapshot(&self) -> RuntimeMetricsSnapshot {
        let mut snapshot = self.metrics.snapshot();
        snapshot.active = self.requests.active_count() as u64;
        snapshot
    }

    pub fn request_stats(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        self.requests.stats(request_id)
    }

    pub async fn drain_request(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        self.requests.drain_request(request_id).await
    }

    pub async fn drain(&self) {
        self.requests.drain().await;
    }

    /// The one public generation arrow.
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

    pub async fn shutdown(self) -> std::result::Result<(), crate::engine_client::Error> {
        self.drain().await;
        self.engine.shutdown().await
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbortReason {
    OwnerCancelled,
    Admin,
    RuntimeShutdown,
}

#[derive(Debug, Error)]
pub enum ServeControlError {
    #[error(transparent)]
    Engine(#[from] crate::engine_client::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestLifecycleState {
    Compiling,
    Submitting,
    Accepted,
    Scheduled,
    Streaming,
    Cancelling,
    Aborting,
    Finished,
    Rejected,
    Cancelled,
    Aborted,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestStatsSnapshot {
    pub request_id: ServeRequestId,
    pub served_name: String,
    pub description: String,
    pub state: RequestLifecycleState,
    pub prompt_tokens: u32,
    pub visible_output_tokens: u32,
    pub internal_tokens: u32,
    pub image_count: u32,
    pub image_steps: u32,
    pub cache: CacheAccounting,
    pub resources: ResourceAccounting,
    pub timings: RuntimeTimings,
}

impl RequestStatsSnapshot {
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
    fn default() -> Self {
        Self {
            state: Mutex::new(RequestRegistryState::default()),
            changed: Notify::new(),
            completed_retention: 1024,
        }
    }
}

impl RuntimeRequestRegistry {
    fn lock(&self) -> std::sync::MutexGuard<'_, RequestRegistryState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

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

    fn control_terminal(&self, request_id: &str) -> Option<LifecycleTerminal> {
        match self.lock().active.get(request_id).map(|stats| stats.state) {
            Some(RequestLifecycleState::Cancelling) => Some(LifecycleTerminal::Cancelled),
            Some(RequestLifecycleState::Aborting) => Some(LifecycleTerminal::Aborted),
            _ => None,
        }
    }

    async fn wait_for_control(&self, request_id: &str) -> LifecycleTerminal {
        loop {
            let changed = self.changed.notified();
            if let Some(terminal) = self.control_terminal(request_id) {
                return terminal;
            }
            changed.await;
        }
    }

    fn drop_terminal(&self, request_id: &str) -> LifecycleTerminal {
        match self.lock().active.get(request_id).map(|stats| stats.state) {
            Some(RequestLifecycleState::Aborting) => LifecycleTerminal::Aborted,
            _ => LifecycleTerminal::Cancelled,
        }
    }

    fn observe(
        &self,
        request_id: &str,
        event: &ServeEvent,
        elapsed_us: u64,
    ) -> Option<LifecycleTerminal> {
        let mut state = self.lock();
        let stats = state.active.get_mut(request_id)?;
        match stats.state {
            RequestLifecycleState::Cancelling => return Some(LifecycleTerminal::Cancelled),
            RequestLifecycleState::Aborting => return Some(LifecycleTerminal::Aborted),
            _ => {}
        }
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
            ServeEvent::PublicCommit { .. } => {}
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
            ServeEvent::Finished { .. }
            | ServeEvent::Rejected { .. }
            | ServeEvent::Cancelled { .. }
            | ServeEvent::Aborted { .. }
            | ServeEvent::Failed { .. } => {}
        }
        None
    }

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

    fn stats(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        let state = self.lock();
        state
            .active
            .get(request_id)
            .or_else(|| state.completed.get(request_id))
            .cloned()
    }

    fn active_count(&self) -> usize {
        self.lock().active.len()
    }

    async fn drain_request(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        loop {
            let changed = self.changed.notified();
            if !self.lock().active.contains_key(request_id) {
                return self.stats(request_id);
            }
            changed.await;
        }
    }

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
pub struct RuntimeMetricsSnapshot {
    pub active: u64,
    pub accepted: u64,
    pub scheduled: u64,
    pub finished: u64,
    pub rejected: u64,
    pub cancelled: u64,
    pub aborted: u64,
    pub failed: u64,
}

impl RuntimeMetricsSnapshot {
    /// Stable lifecycle state names and values exported by the public metrics
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
    fn drop(&mut self) {
        if !self.terminal {
            let elapsed_us = self.started.elapsed().as_micros() as u64;
            let terminal = self.requests.drop_terminal(&self.request_id);
            let _ = self.terminal(terminal, elapsed_us);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheAccounting {
    pub read_enabled: bool,
    pub write_enabled: bool,
    pub encoder_pin_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceAccounting {
    pub expected_kv_tokens: u64,
    pub image_latent_units: u64,
    pub encoder_cache_pins: usize,
    pub replayable: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTimings {
    pub compile_us: u64,
    pub queue_us: Option<u64>,
    pub first_visible_output_us: Option<u64>,
    pub total_us: u64,
}

/// Protocol-neutral runtime event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ServeEvent {
    Accepted {
        request_id: ServeRequestId,
        served_name: String,
        description: String,
        compile_duration_us: u64,
        prompt_token_count: usize,
        prompt_token_ids: Vec<u32>,
        prompt_logprobs: Option<DecodedPromptLogprobs>,
    },
    Scheduled {
        request_id: ServeRequestId,
        queued_at: Option<f64>,
        scheduled_at: Option<f64>,
        cache: CacheAccounting,
        resources: ResourceAccounting,
    },
    PublicCommit {
        commit: PublicCommit,
    },
    TextDelta {
        candidate_id: CandidateId,
        text: String,
        token_ids: Vec<u32>,
        logprobs: Option<DecodedLogprobs>,
    },
    InternalTextDelta {
        candidate_id: CandidateId,
        text: String,
    },
    ReasoningDelta {
        candidate_id: CandidateId,
        text: String,
    },
    OutputBlockStart {
        candidate_id: CandidateId,
        index: usize,
        kind: AssistantBlockKind,
    },
    OutputBlockEnd {
        candidate_id: CandidateId,
        index: usize,
        block: AssistantContentBlock,
    },
    ToolCallStart {
        candidate_id: CandidateId,
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        candidate_id: CandidateId,
        index: usize,
        delta: String,
    },
    ToolCallEnd {
        candidate_id: CandidateId,
        index: usize,
        id: String,
        name: String,
        arguments: String,
    },
    ImageBegin {
        candidate_id: CandidateId,
        image_id: String,
        width: Option<u32>,
        height: Option<u32>,
        steps: Option<u32>,
        elapsed_us: u64,
    },
    ImageStep {
        candidate_id: CandidateId,
        image_id: String,
        step: u32,
        elapsed_us: u64,
    },
    ImageCommit {
        candidate_id: CandidateId,
        image_id: String,
        elapsed_us: u64,
    },
    ImageDone {
        candidate_id: CandidateId,
        image_id: String,
        width: Option<u32>,
        height: Option<u32>,
        bytes: Option<u64>,
        sha256: Option<String>,
        pixels_png_b64: Option<String>,
        elapsed_us: u64,
    },
    Usage {
        prompt_tokens: u32,
        visible_output_tokens: u32,
        internal_tokens: u32,
        image_count: u32,
        image_steps: u32,
        cache: CacheAccounting,
        resources: ResourceAccounting,
        timings: RuntimeTimings,
    },
    Finished {
        candidate_id: CandidateId,
        reason: FinishStatus,
        finish_detail: Option<String>,
    },
    Rejected {
        request_id: ServeRequestId,
        message: String,
    },
    Cancelled {
        request_id: ServeRequestId,
    },
    Aborted {
        request_id: ServeRequestId,
    },
    Failed {
        request_id: ServeRequestId,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishStatus {
    Stop { cause: Option<StopCause> },
    Length,
    Abort,
    Error,
    Repetition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum StopCause {
    Eos,
    TokenId(u32),
    Text(String),
}

impl From<&FinishReason> for FinishStatus {
    fn from(reason: &FinishReason) -> Self {
        match reason.reason() {
            uniserve_core::FinishReason::Eos
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
    use uniserve_core::{PublicModality, SemanticRoot, TokenLogprob};
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

    fn public_commit(event_seq: u64, modality: PublicModality) -> PublicCommit {
        PublicCommit {
            event_seq,
            modality,
            committed_at: event_seq as f64,
            semantic_root: SemanticRoot {
                producer_op_id: uniserve_core::OpId(event_seq),
                point_index: event_seq as u32,
            },
        }
    }

    fn bagel_output_policy() -> OutputProcessorPolicy {
        OutputProcessorPolicy::Bagel
    }

    #[tokio::test]
    async fn assembler_attaches_ranked_logprobs_to_text_delta() {
        let tokenizer = crate::serving::test_support::configured_tokenizer();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(GenerationEvent::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        tx.try_send(GenerationEvent::TextToken {
            id: b'a' as u32,
            logprob: Some(-0.25),
            public_commit: None,
        })
        .unwrap();
        tx.try_send(GenerationEvent::TokenLogprobs {
            id: b'a' as u32,
            candidates: vec![TokenLogprob {
                token_id: b'a' as u32,
                logprob: -0.25,
                rank: 3,
            }],
        })
        .unwrap();
        tx.try_send(GenerationEvent::Finished {
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
        tx.try_send(GenerationEvent::Scheduled {
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

        tx.send(GenerationEvent::TextToken {
            id: b'a' as u32,
            logprob: None,
            public_commit: Some(public_commit(1, PublicModality::Text)),
        })
        .await
        .unwrap();
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::PublicCommit {
                commit: PublicCommit { event_seq: 1, .. }
            }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::TextDelta { text, .. })) if text == "a"
        ));

        tx.send(GenerationEvent::ImageCommit { image_id: 0 })
            .await
            .unwrap();
        tx.send(GenerationEvent::ImageDone {
            image_id: 0,
            height: 1,
            width: 1,
            bytes: 3,
            sha256: uniserve_core::Digest::zero(),
            pixels_png_b64: "cG5n".to_string(),
            public_commit: Some(public_commit(3, PublicModality::Image)),
        })
        .await
        .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), events.next())
                .await
                .is_err(),
            "the image became public before a continuation token arrived"
        );

        tx.send(GenerationEvent::TextToken {
            id: b'b' as u32,
            logprob: None,
            public_commit: Some(public_commit(4, PublicModality::Text)),
        })
        .await
        .unwrap();
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::ImageCommit { .. }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::PublicCommit {
                commit: PublicCommit { event_seq: 3, .. }
            }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::ImageDone { .. }))
        ));
    }
}
