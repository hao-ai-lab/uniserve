//! Canonical serving runtime.
//!
//! The public funnel is a single ownership chain:
//! `GenerateReqInput -> ResolvedModel::tokenize -> TokenizedGenerateReqInput ->
//! EngineGateway::submit_generation -> ServeEvent stream`. There is one
//! internal admission value, one model-owned tokenize arrow, one engine
//! submission per request, and one stream assembler.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};
use std::time::Instant;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{Notify, mpsc};
use uniserve_engine_gateway::transport::{
    GenEvent, GenerationEventStream, GenerationFinishReason, StreamCancelCause,
};
use uniserve_engine_gateway::{EngineGateway, GenerationSubmission};
pub use uniserve_engine_gateway::{PublicCommit, PublicModality, SemanticRoot};

pub use input::{
    CacheBounds, DecodeControls, GenerateReqInput, ImageGenControls, ImageInput, ModalitySelection,
    ModelEventIdentity, OutputContract, OutputProcessorPolicy, PromptInput, SamplingConfig,
    SchedulingBounds, StopConfig, SubmissionMetadata, TokenizedGenerateReqInput,
};
pub use model::{BagelDesc, Qwen3Desc, ResolvedModel, SenseNovaDesc};

use crate::chat::{AssistantBlockKind, AssistantContentBlock, ChatEvent, Qwen3ChatOutputProcessor};
use crate::omni::output::{DialectOutputProcessor, DialectTextDelta};
use crate::text::output::stop_string_holdback_bytes;
use crate::text::{
    DecodedLogprobs, DecodedPromptLogprobs, DecodedTextEvent, FinishReason, StopReason,
    TextDecodeOptions,
};

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
    #[error("model resolution failed: {0}")]
    ModelResolution(String),
    #[error("engine runtime error: {0}")]
    Engine(String),
    #[error("request `{request_id}` cannot be tokenized: {message}")]
    Tokenize {
        request_id: ServeRequestId,
        message: String,
    },
    #[error("request `{request_id}` output processing failed: {message}")]
    OutputProcessing {
        request_id: ServeRequestId,
        message: String,
    },
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
    gateway: EngineGateway,
    metrics: Arc<RuntimeLifecycleMetrics>,
    requests: Arc<RuntimeRequestRegistry>,
}

impl ServingRuntime {
    pub fn new(model: ResolvedModel, gateway: EngineGateway) -> Self {
        Self {
            runtime_id: NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed),
            model: Arc::new(model),
            gateway,
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

    pub fn tokenizer(&self) -> crate::text::tokenizer::DynTokenizer {
        self.model.tokenizer()
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
            identity.profile_id.clone(),
            identity.dialect_id.clone(),
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
                    message: format!("tokenize worker failed: {error}"),
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

        let TokenizedGenerateReqInput {
            request_id: _,
            request,
            prompt_token_ids,
            decode,
            emit_token_ids,
            prompt_logprobs_requested,
            generated_logprobs_requested,
            skip_special_tokens,
            output_processor,
            submission,
            identity,
            cache,
            resources,
        } = tokenized;

        let event_context = EventContext {
            profile_id: identity.profile_id.clone(),
            dialect_id: identity.dialect_id.clone(),
            compile_duration_us,
            cache,
            resources,
            skip_special_tokens,
            metrics: Arc::clone(&self.metrics),
        };

        let mut gateway_submission = GenerationSubmission::new(request_id.to_string(), request);
        gateway_submission.trace_headers = submission.trace_headers;

        let stream_result: Result<ServeEventStream> =
            match self.gateway.submit_generation(gateway_submission).await {
                Ok(stream) => Ok(Box::pin(assemble_event_stream(
                    request_id.clone(),
                    event_context,
                    prompt_token_ids,
                    self.model.tokenizer(),
                    prompt_logprobs_requested,
                    generated_logprobs_requested,
                    emit_token_ids,
                    decode,
                    output_processor,
                    stream,
                )) as ServeEventStream),
                Err(error) => Err(ServeError::Engine(error.to_string())),
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
            self.gateway
                .cancel(&request_id)
                .await
                .map_err(|message| ServeControlError::Engine(message.to_string()))?;
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
            self.gateway
                .abort(&request_id)
                .await
                .map_err(|message| ServeControlError::Engine(message.to_string()))?;
        }
        let _ = reason;
        Ok(())
    }

    async fn apply_engine_control(&self, request_id: &str, terminal: LifecycleTerminal) {
        let result = match terminal {
            LifecycleTerminal::Aborted => self.gateway.abort(request_id).await,
            LifecycleTerminal::Cancelled
            | LifecycleTerminal::Finished
            | LifecycleTerminal::Rejected
            | LifecycleTerminal::Failed => self.gateway.cancel(request_id).await,
        };
        if let Err(error) = result {
            tracing::warn!(%request_id, %error, "failed to apply pending request control after submission");
        }
    }

    pub async fn shutdown(self) -> std::result::Result<(), uniserve_engine_gateway::Error> {
        self.drain().await;
        self.gateway.shutdown().await
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
    #[error("engine control failed: {0}")]
    Engine(String),
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
    pub profile_id: String,
    pub dialect_id: String,
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
        profile_id: String,
        dialect_id: String,
        compile_us: u64,
    ) -> Self {
        Self {
            request_id,
            profile_id,
            dialect_id,
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
        profile_id: String,
        dialect_id: String,
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
            profile_id,
            dialect_id,
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
    profile_id: String,
    dialect_id: String,
    compile_duration_us: u64,
    cache: CacheAccounting,
    resources: ResourceAccounting,
    skip_special_tokens: bool,
    metrics: Arc<RuntimeLifecycleMetrics>,
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
    pub scratch_units: u64,
    pub host_scratch_tokens: u64,
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
        profile_id: String,
        dialect_id: String,
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
        match reason {
            FinishReason::Stop(stop_reason) => Self::Stop {
                cause: stop_reason.as_ref().map(|value| match value {
                    StopReason::TokenId(id) => StopCause::TokenId(*id),
                    StopReason::Text(text) => StopCause::Text(text.clone()),
                }),
            },
            FinishReason::Length => Self::Length,
            FinishReason::Abort => Self::Abort,
            FinishReason::Cancelled => Self::Abort,
            FinishReason::Aborted => Self::Abort,
            FinishReason::Error => Self::Error,
            FinishReason::Repetition => Self::Repetition,
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

// ==========================================================================
// Single engine -> ServeEvent stream assembler.
// ==========================================================================

/// Runtime output sink built from the model-supplied [`OutputProcessorPolicy`].
enum OutputSink {
    /// Raw visible text.
    Raw,
    /// Qwen3 chat reasoning/tool parsing.
    Chat(ChatOutputBridge),
    /// SenseNova/Bagel output filter over committed text.
    Dialect(DialectOutputProcessor),
}

fn build_output_sink(
    request_id: &ServeRequestId,
    policy: OutputProcessorPolicy,
    tokenizer: &crate::text::tokenizer::DynTokenizer,
    prompt_token_ids: &[u32],
) -> Result<OutputSink> {
    match policy {
        OutputProcessorPolicy::None => Ok(OutputSink::Raw),
        OutputProcessorPolicy::Qwen3(request) => {
            let mut request = *request;
            let processor =
                Qwen3ChatOutputProcessor::new(&mut request, std::sync::Arc::clone(tokenizer))
                    .map_err(|error| ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: error.to_string(),
                    })?;
            let bridge =
                ChatOutputBridge::new(processor).map_err(|error| ServeError::OutputProcessing {
                    request_id: request_id.clone(),
                    message: error.to_string(),
                })?;
            Ok(OutputSink::Chat(bridge))
        }
        OutputProcessorPolicy::Dialect(output_filter) => {
            let processor = DialectOutputProcessor::new(
                output_filter,
                std::sync::Arc::clone(tokenizer),
                prompt_token_ids,
            )
            .map_err(|error| ServeError::OutputProcessing {
                request_id: request_id.clone(),
                message: error.to_string(),
            })?;
            Ok(OutputSink::Dialect(processor))
        }
    }
}

enum ChatDecodedInput {
    Event(Box<DecodedTextEvent>),
    Barrier(Arc<AtomicBool>),
}

struct ChatDecodedInputStream {
    receiver: mpsc::Receiver<ChatDecodedInput>,
}

impl Stream for ChatDecodedInputStream {
    type Item = crate::chat::output::Result<DecodedTextEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match self.receiver.poll_recv(cx) {
                Poll::Ready(Some(ChatDecodedInput::Event(event))) => {
                    return Poll::Ready(Some(Ok(*event)));
                }
                Poll::Ready(Some(ChatDecodedInput::Barrier(reached))) => {
                    reached.store(true, Ordering::Release);
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

struct ChatOutputBridge {
    sender: mpsc::Sender<ChatDecodedInput>,
    output: crate::chat::output::DynChatEventStream,
}

impl ChatOutputBridge {
    fn new(processor: Qwen3ChatOutputProcessor) -> crate::chat::output::Result<Self> {
        let (sender, receiver) = mpsc::channel(2);
        let decoded = Box::pin(ChatDecodedInputStream { receiver });
        let output = processor.process(decoded)?;
        Ok(Self { sender, output })
    }

    async fn push(
        &mut self,
        request_id: &ServeRequestId,
        event: DecodedTextEvent,
    ) -> Result<Vec<ChatEvent>> {
        self.sender
            .try_send(ChatDecodedInput::Event(Box::new(event)))
            .map_err(|_| ServeError::OutputProcessing {
                request_id: request_id.clone(),
                message: "chat output processor closed before receiving decoded text".to_string(),
            })?;
        let reached = Arc::new(AtomicBool::new(false));
        self.sender
            .try_send(ChatDecodedInput::Barrier(Arc::clone(&reached)))
            .map_err(|_| ServeError::OutputProcessing {
                request_id: request_id.clone(),
                message: "chat output processor closed before its synchronization barrier"
                    .to_string(),
            })?;

        let mut events = Vec::new();
        futures::future::poll_fn(|cx| {
            loop {
                match self.output.as_mut().poll_next(cx) {
                    Poll::Ready(Some(Ok(event))) => events.push(event),
                    Poll::Ready(Some(Err(error))) => {
                        return Poll::Ready(Err(ServeError::OutputProcessing {
                            request_id: request_id.clone(),
                            message: error.to_string(),
                        }));
                    }
                    Poll::Ready(None) => {
                        return Poll::Ready(Err(ServeError::OutputProcessing {
                            request_id: request_id.clone(),
                            message: "chat output processor closed before terminal output"
                                .to_string(),
                        }));
                    }
                    Poll::Pending if reached.load(Ordering::Acquire) => {
                        return Poll::Ready(Ok(std::mem::take(&mut events)));
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
        })
        .await
    }
}

struct ChatDone {
    prompt_token_count: usize,
    output_token_count: usize,
    visible_output_token_count: usize,
    internal_token_count: usize,
    finish_reason: FinishReason,
}

enum MappedChatEvent {
    Ignore,
    Event(ServeEvent),
    Done(ChatDone),
}

fn map_chat_event(event: ChatEvent) -> MappedChatEvent {
    match event {
        ChatEvent::Start { .. } => MappedChatEvent::Ignore,
        ChatEvent::BlockStart { index, kind } => {
            MappedChatEvent::Event(ServeEvent::OutputBlockStart {
                candidate_id: CandidateId::PRIMARY,
                index,
                kind,
            })
        }
        ChatEvent::BlockDelta { kind, delta, .. } => match kind {
            AssistantBlockKind::Text => MappedChatEvent::Event(ServeEvent::TextDelta {
                candidate_id: CandidateId::PRIMARY,
                text: delta,
                token_ids: Vec::new(),
                logprobs: None,
            }),
            AssistantBlockKind::Reasoning => MappedChatEvent::Event(ServeEvent::ReasoningDelta {
                candidate_id: CandidateId::PRIMARY,
                text: delta,
            }),
            AssistantBlockKind::ToolCall => MappedChatEvent::Ignore,
        },
        ChatEvent::LogprobsDelta {
            token_ids,
            logprobs,
        } => MappedChatEvent::Event(ServeEvent::TextDelta {
            candidate_id: CandidateId::PRIMARY,
            text: String::new(),
            token_ids,
            logprobs,
        }),
        ChatEvent::BlockEnd { index, block } => {
            MappedChatEvent::Event(ServeEvent::OutputBlockEnd {
                candidate_id: CandidateId::PRIMARY,
                index,
                block,
            })
        }
        ChatEvent::ToolCallStart { index, id, name } => {
            MappedChatEvent::Event(ServeEvent::ToolCallStart {
                candidate_id: CandidateId::PRIMARY,
                index,
                id,
                name,
            })
        }
        ChatEvent::ToolCallArgumentsDelta { index, delta } => {
            MappedChatEvent::Event(ServeEvent::ToolCallArgumentsDelta {
                candidate_id: CandidateId::PRIMARY,
                index,
                delta,
            })
        }
        ChatEvent::ToolCallEnd { index, call } => MappedChatEvent::Event(ServeEvent::ToolCallEnd {
            candidate_id: CandidateId::PRIMARY,
            index,
            id: call.id,
            name: call.name,
            arguments: call.arguments,
        }),
        ChatEvent::Done {
            prompt_token_count,
            output_token_count,
            visible_output_token_count,
            internal_token_count,
            finish_reason,
            ..
        } => MappedChatEvent::Done(ChatDone {
            prompt_token_count,
            output_token_count,
            visible_output_token_count,
            internal_token_count,
            finish_reason,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
async fn emit_text_update(
    request_id: &ServeRequestId,
    sink: &mut OutputSink,
    text: String,
    token_ids: Vec<u32>,
    logprobs: Option<DecodedLogprobs>,
    mut public_commit: Option<PublicCommit>,
    finished: Option<crate::text::Finished>,
    started: &Instant,
    first_visible_output_us: &mut Option<u64>,
    y: &mut TryYielder<ServeEvent, ServeError>,
) -> Result<Option<ChatDone>> {
    if let OutputSink::Chat(bridge) = sink {
        let events = bridge
            .push(
                request_id,
                DecodedTextEvent::TextDelta {
                    delta: text,
                    token_ids,
                    logprobs,
                    public_commit: public_commit.clone(),
                    finished,
                },
            )
            .await?;
        let mut done = None;
        for event in events {
            match map_chat_event(event) {
                MappedChatEvent::Ignore => {}
                MappedChatEvent::Event(event) => {
                    let visible = matches!(&event, ServeEvent::TextDelta { text, .. } if !text.is_empty())
                        || matches!(&event, ServeEvent::ReasoningDelta { text, .. } if !text.is_empty())
                        || matches!(&event, ServeEvent::ToolCallStart { .. });
                    if visible {
                        first_visible_output_us
                            .get_or_insert_with(|| started.elapsed().as_micros() as u64);
                        if let Some(commit) = public_commit.take() {
                            y.yield_ok(ServeEvent::PublicCommit { commit }).await;
                        }
                    }
                    y.yield_ok(event).await;
                }
                MappedChatEvent::Done(next) => {
                    if done.replace(next).is_some() {
                        return Err(ServeError::OutputProcessing {
                            request_id: request_id.clone(),
                            message: "chat output processor emitted multiple terminal events"
                                .to_string(),
                        });
                    }
                }
            }
        }
        return Ok(done);
    }

    let delta: DialectTextDelta = match sink {
        OutputSink::Dialect(processor) => processor.push(&text),
        OutputSink::Raw => DialectTextDelta {
            visible: text,
            reasoning: String::new(),
        },
        OutputSink::Chat(_) => unreachable!("chat sink handled above"),
    };

    if !delta.reasoning.is_empty() {
        first_visible_output_us.get_or_insert_with(|| started.elapsed().as_micros() as u64);
        if let Some(commit) = public_commit.take() {
            y.yield_ok(ServeEvent::PublicCommit { commit }).await;
        }
        y.yield_ok(ServeEvent::ReasoningDelta {
            candidate_id: CandidateId::PRIMARY,
            text: delta.reasoning,
        })
        .await;
    }
    if !delta.visible.is_empty() {
        first_visible_output_us.get_or_insert_with(|| started.elapsed().as_micros() as u64);
        if let Some(commit) = public_commit.take() {
            y.yield_ok(ServeEvent::PublicCommit { commit }).await;
        }
    }
    if !delta.visible.is_empty()
        || !token_ids.is_empty()
        || logprobs.is_some()
        || finished.is_some()
    {
        y.yield_ok(ServeEvent::TextDelta {
            candidate_id: CandidateId::PRIMARY,
            text: delta.visible,
            token_ids,
            logprobs,
        })
        .await;
    }
    Ok(finished.map(|finished| ChatDone {
        prompt_token_count: finished.prompt_token_count,
        output_token_count: finished.output_token_count,
        visible_output_token_count: finished
            .output_token_count
            .saturating_sub(finished.internal_token_count),
        internal_token_count: finished.internal_token_count,
        finish_reason: finished.finish_reason,
    }))
}

#[allow(clippy::too_many_arguments)]
async fn emit_terminal(
    request_id: &ServeRequestId,
    event_context: &EventContext,
    started: &Instant,
    queue_us: Option<u64>,
    first_visible_output_us: Option<u64>,
    image_count: u32,
    image_steps: u32,
    finish_detail: Option<String>,
    done: ChatDone,
    y: &mut TryYielder<ServeEvent, ServeError>,
) {
    debug_assert_eq!(
        done.output_token_count,
        done.visible_output_token_count
            .saturating_add(done.internal_token_count)
    );
    y.yield_ok(ServeEvent::Usage {
        prompt_tokens: done.prompt_token_count.min(u32::MAX as usize) as u32,
        visible_output_tokens: done.visible_output_token_count.min(u32::MAX as usize) as u32,
        internal_tokens: done.internal_token_count.min(u32::MAX as usize) as u32,
        image_count,
        image_steps,
        cache: event_context.cache.clone(),
        resources: event_context.resources.clone(),
        timings: RuntimeTimings {
            compile_us: event_context.compile_duration_us,
            queue_us,
            first_visible_output_us,
            total_us: started.elapsed().as_micros() as u64,
        },
    })
    .await;
    match done.finish_reason {
        FinishReason::Cancelled => {
            y.yield_ok(ServeEvent::Cancelled {
                request_id: request_id.clone(),
            })
            .await;
        }
        FinishReason::Abort | FinishReason::Aborted => {
            y.yield_ok(ServeEvent::Aborted {
                request_id: request_id.clone(),
            })
            .await;
        }
        FinishReason::Error => {
            y.yield_ok(ServeEvent::Failed {
                request_id: request_id.clone(),
                message: finish_detail.unwrap_or_else(|| "engine execution failed".to_string()),
            })
            .await;
        }
        reason => {
            y.yield_ok(ServeEvent::Finished {
                candidate_id: CandidateId::PRIMARY,
                reason: FinishStatus::from(&reason),
                finish_detail,
            })
            .await;
        }
    }
}

#[try_stream]
#[allow(clippy::too_many_arguments)]
async fn assemble_event_stream(
    request_id: ServeRequestId,
    event_context: EventContext,
    prompt_token_ids: Vec<u32>,
    tokenizer: crate::text::tokenizer::DynTokenizer,
    prompt_logprobs_requested: bool,
    generated_logprobs_requested: bool,
    emit_token_ids: bool,
    mut decode_options: TextDecodeOptions,
    output_processor: OutputProcessorPolicy,
    mut stream: GenerationEventStream,
    mut y: TryYielder<ServeEvent, ServeError>,
) -> Result<()> {
    let started = Instant::now();
    let mut first_visible_output_us = None;
    let mut queue_us = None;
    let mut emitted_output_tokens = 0_u32;
    let mut image_count = 0_u32;
    let mut image_steps = 0_u32;
    let expected_prompt_positions = prompt_token_ids.len().saturating_sub(1);
    let mut prompt_positions = Vec::new();
    let mut accepted = false;
    let mut pending_scheduled = None;
    let mut pending_token = None;
    let mut last_public_commit = None;
    let mut pending_image_events = Vec::new();
    let mut sink = build_output_sink(&request_id, output_processor, &tokenizer, &prompt_token_ids)?;
    let mut decoder = tokenizer.create_decode_stream(
        &prompt_token_ids,
        decode_options.skip_special_tokens,
        stop_string_holdback_bytes(&decode_options),
    );
    macro_rules! emit_accepted {
        ($prompt_logprobs:expr) => {{
            let prompt_logprobs: Option<DecodedPromptLogprobs> = $prompt_logprobs;
            y.yield_ok(ServeEvent::Accepted {
                request_id: request_id.clone(),
                profile_id: event_context.profile_id.clone(),
                dialect_id: event_context.dialect_id.clone(),
                compile_duration_us: event_context.compile_duration_us,
                prompt_token_count: prompt_token_ids.len(),
                prompt_token_ids: prompt_token_ids.clone(),
                prompt_logprobs: prompt_logprobs.clone(),
            })
            .await;
            if let OutputSink::Chat(bridge) = &mut sink {
                let events = bridge
                    .push(
                        &request_id,
                        DecodedTextEvent::Start {
                            prompt_token_ids: Arc::from(prompt_token_ids.clone()),
                            prompt_logprobs,
                            queued_at: None,
                            scheduled_at: None,
                        },
                    )
                    .await?;
                for event in events {
                    if !matches!(map_chat_event(event), MappedChatEvent::Ignore) {
                        return Err(ServeError::OutputProcessing {
                            request_id: request_id.clone(),
                            message: "chat output processor emitted output while initializing"
                                .to_string(),
                        });
                    }
                }
            }
            accepted = true;
            if let Some((queued_at, scheduled_at)) = pending_scheduled.take() {
                y.yield_ok(ServeEvent::Scheduled {
                    request_id: request_id.clone(),
                    queued_at: Some(queued_at),
                    scheduled_at: Some(scheduled_at),
                    cache: event_context.cache.clone(),
                    resources: event_context.resources.clone(),
                })
                .await;
            }
        }};
    }
    macro_rules! flush_pending_images {
        () => {{
            for event in pending_image_events.drain(..) {
                y.yield_ok(event).await;
            }
        }};
    }
    macro_rules! consume_token {
        ($id:expr, $logprobs:expr, $public_commit:expr) => {{
            let id = $id;
            let logprobs: Option<DecodedLogprobs> = $logprobs;
            let public_commit: Option<PublicCommit> = $public_commit;
            if public_commit.is_some() {
                last_public_commit = public_commit.clone();
            }
            flush_pending_images!();
            emitted_output_tokens = emitted_output_tokens.saturating_add(1);
            let new_bytes =
                decoder
                    .push_token(id)
                    .map_err(|error| ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: error.to_string(),
                    })?;
            let matched_stop = if emitted_output_tokens > decode_options.min_tokens {
                decode_options.stop_strings.as_ref().and_then(|stops| {
                    crate::text::output::matches_stop_string(stops, decoder.output(), new_bytes)
                })
            } else {
                None
            };
            let (text, stop_string) = if let Some((index, offset)) = matched_stop {
                let stop_string = decode_options
                    .stop_strings
                    .as_mut()
                    .ok_or_else(|| ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: "stop-string match lost its configured stop set".to_string(),
                    })?
                    .swap_remove(index);
                let truncate_to = if decode_options.include_stop_str_in_output {
                    offset + stop_string.len()
                } else {
                    offset
                };
                let (last_chunk, _) = decoder.flush(Some(truncate_to)).map_err(|error| {
                    ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: error.to_string(),
                    }
                })?;
                (last_chunk.unwrap_or_default(), Some(stop_string))
            } else {
                (decoder.next_chunk().unwrap_or_default(), None)
            };
            let finished = stop_string
                .as_ref()
                .map(|stop_string| crate::text::Finished {
                    prompt_token_count: prompt_token_ids.len(),
                    output_token_count: emitted_output_tokens as usize,
                    internal_token_count: 0,
                    finish_reason: FinishReason::Stop(Some(StopReason::Text(stop_string.clone()))),
                });
            if stop_string.is_none() {
                stream.acknowledge_text_prefix();
            } else {
                stream.cancel_at_consumed_prefix(StreamCancelCause::StopStringMatched);
            }
            let emitted_ids = if emit_token_ids { vec![id] } else { Vec::new() };
            let done = emit_text_update(
                &request_id,
                &mut sink,
                text,
                emitted_ids,
                logprobs,
                public_commit,
                finished,
                &started,
                &mut first_visible_output_us,
                &mut y,
            )
            .await?;
            if stop_string.is_some() {
                let done = done.ok_or_else(|| ServeError::OutputProcessing {
                    request_id: request_id.clone(),
                    message: "output processor omitted the stop-string terminal event".to_string(),
                })?;
                emit_terminal(
                    &request_id,
                    &event_context,
                    &started,
                    queue_us,
                    first_visible_output_us,
                    image_count,
                    image_steps,
                    Some("stop".to_string()),
                    done,
                    &mut y,
                )
                .await;
                return Ok(());
            }
        }};
    }
    macro_rules! ensure_output_ready {
        ($kind:literal) => {{
            if !accepted || pending_token.is_some() {
                return Err(ServeError::OutputProcessing {
                    request_id: request_id.clone(),
                    message: concat!($kind, " arrived before output metadata was complete")
                        .to_string(),
                });
            }
        }};
    }
    if !prompt_logprobs_requested || expected_prompt_positions == 0 {
        let prompt_logprobs =
            if prompt_logprobs_requested {
                let first_token_id = prompt_token_ids.first().copied().ok_or_else(|| {
                    ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: "prompt logprobs require a non-empty tokenized prompt".to_string(),
                    }
                })?;
                let first_token = tokenizer
                    .decode(&[first_token_id], event_context.skip_special_tokens)
                    .map_err(|error| ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: error.to_string(),
                    })?;
                Some(DecodedPromptLogprobs {
                    first_token_id,
                    first_token,
                    scored_positions: Vec::new(),
                })
            } else {
                None
            };
        emit_accepted!(prompt_logprobs);
    }
    while let Some(event) = stream.next().await {
        match event {
            GenEvent::Scheduled {
                queued_at,
                scheduled_at,
            } => {
                queue_us = Some(((scheduled_at - queued_at).max(0.0) * 1_000_000.0) as u64);
                if accepted {
                    y.yield_ok(ServeEvent::Scheduled {
                        request_id: request_id.clone(),
                        queued_at: Some(queued_at),
                        scheduled_at: Some(scheduled_at),
                        cache: event_context.cache.clone(),
                        resources: event_context.resources.clone(),
                    })
                    .await;
                } else {
                    pending_scheduled = Some((queued_at, scheduled_at));
                }
                event_context
                    .metrics
                    .scheduled
                    .fetch_add(1, Ordering::Relaxed);
            }
            GenEvent::PromptLogprobs { positions } => {
                prompt_positions.extend(positions);
                if prompt_positions.len() > expected_prompt_positions {
                    return Err(ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: "engine returned more prompt logprob positions than requested"
                            .to_string(),
                    });
                }
                if !accepted && prompt_positions.len() == expected_prompt_positions {
                    let positions = std::mem::take(&mut prompt_positions);
                    let decoded = crate::text::output::decode_prompt_logprobs(
                        &request_id,
                        tokenizer.as_ref(),
                        &prompt_token_ids,
                        &positions,
                        event_context.skip_special_tokens,
                    )
                    .map_err(|error| ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: error.to_string(),
                    })?;
                    emit_accepted!(Some(decoded));
                }
            }
            GenEvent::TextToken {
                id, public_commit, ..
            } => {
                if !accepted {
                    return Err(ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: "engine began generation before prompt logprobs were complete"
                            .to_string(),
                    });
                }
                if pending_token.is_some() {
                    return Err(ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: "engine emitted a new token before resolving prior logprobs"
                            .to_string(),
                    });
                }
                if generated_logprobs_requested {
                    pending_token = Some((id, public_commit));
                } else {
                    consume_token!(id, None, public_commit);
                }
            }
            GenEvent::TokenLogprobs { id, candidates } => {
                let (pending, public_commit) =
                    pending_token
                        .take()
                        .ok_or_else(|| ServeError::OutputProcessing {
                            request_id: request_id.clone(),
                            message: "engine returned token logprobs without a pending token"
                                .to_string(),
                        })?;
                if pending != id || candidates.first().is_none_or(|entry| entry.token_id != id) {
                    return Err(ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: "token logprobs do not match the emitted token".to_string(),
                    });
                }
                let logprobs = crate::text::output::decode_logprobs(
                    tokenizer.as_ref(),
                    &[
                        uniserve_engine_gateway::generation::GenerationPositionLogprobs {
                            entries: candidates,
                        },
                    ],
                    event_context.skip_special_tokens,
                )
                .map_err(|error| ServeError::OutputProcessing {
                    request_id: request_id.clone(),
                    message: error.to_string(),
                })?;
                consume_token!(id, Some(logprobs), public_commit);
            }
            GenEvent::ImageBegin {
                image_id,
                height,
                width,
                steps,
            } => {
                ensure_output_ready!("image-begin event");
                first_visible_output_us.get_or_insert_with(|| started.elapsed().as_micros() as u64);
                y.yield_ok(ServeEvent::ImageBegin {
                    candidate_id: CandidateId::PRIMARY,
                    image_id: image_id.to_string(),
                    width: Some(width),
                    height: Some(height),
                    steps: Some(steps as u32),
                    elapsed_us: started.elapsed().as_micros() as u64,
                })
                .await;
            }
            GenEvent::ImageStep { image_id, step } => {
                ensure_output_ready!("image-step event");
                image_steps = image_steps.saturating_add(1);
                y.yield_ok(ServeEvent::ImageStep {
                    candidate_id: CandidateId::PRIMARY,
                    image_id: image_id.to_string(),
                    step: step as u32,
                    elapsed_us: started.elapsed().as_micros() as u64,
                })
                .await;
            }
            GenEvent::ImageCommit { image_id } => {
                ensure_output_ready!("image-commit event");
                pending_image_events.push(ServeEvent::ImageCommit {
                    candidate_id: CandidateId::PRIMARY,
                    image_id: image_id.to_string(),
                    elapsed_us: started.elapsed().as_micros() as u64,
                });
            }
            GenEvent::ImageDone {
                image_id,
                height,
                width,
                bytes,
                sha256,
                pixels_png_b64,
                public_commit,
            } => {
                ensure_output_ready!("image-done event");
                image_count = image_count.saturating_add(1);
                if let Some(commit) = public_commit {
                    pending_image_events.push(ServeEvent::PublicCommit { commit });
                }
                pending_image_events.push(ServeEvent::ImageDone {
                    candidate_id: CandidateId::PRIMARY,
                    image_id: image_id.to_string(),
                    width: Some(width),
                    height: Some(height),
                    bytes: Some(bytes),
                    sha256: Some(sha256),
                    pixels_png_b64: Some(pixels_png_b64),
                    elapsed_us: started.elapsed().as_micros() as u64,
                });
            }
            GenEvent::Finished {
                reason,
                stop_reason,
                prompt_tokens,
                completion_tokens,
                images,
            } => {
                ensure_output_ready!("terminal event");
                flush_pending_images!();
                let (last_chunk, _) =
                    decoder
                        .flush(None)
                        .map_err(|error| ServeError::OutputProcessing {
                            request_id: request_id.clone(),
                            message: error.to_string(),
                        })?;
                let finish_detail = generation_finish_detail(&reason).to_string();
                let finished = crate::text::Finished {
                    prompt_token_count: prompt_tokens,
                    output_token_count: completion_tokens,
                    internal_token_count: completion_tokens
                        .saturating_sub(emitted_output_tokens as usize),
                    finish_reason: generation_text_finish_reason(reason, stop_reason),
                };
                let done = emit_text_update(
                    &request_id,
                    &mut sink,
                    last_chunk.unwrap_or_default(),
                    Vec::new(),
                    None,
                    last_public_commit,
                    Some(finished),
                    &started,
                    &mut first_visible_output_us,
                    &mut y,
                )
                .await?
                .ok_or_else(|| ServeError::OutputProcessing {
                    request_id: request_id.clone(),
                    message: "output processor omitted the engine terminal event".to_string(),
                })?;
                image_count = image_count.max(images.min(u32::MAX as usize) as u32);
                emit_terminal(
                    &request_id,
                    &event_context,
                    &started,
                    queue_us,
                    first_visible_output_us,
                    image_count,
                    image_steps,
                    Some(finish_detail),
                    done,
                    &mut y,
                )
                .await;
                return Ok(());
            }
            GenEvent::Rejected { message } => {
                flush_pending_images!();
                y.yield_ok(ServeEvent::Rejected {
                    request_id: request_id.clone(),
                    message,
                })
                .await;
                return Ok(());
            }
            GenEvent::Error { message } => {
                flush_pending_images!();
                y.yield_ok(ServeEvent::Failed {
                    request_id: request_id.clone(),
                    message,
                })
                .await;
                return Ok(());
            }
        }
    }

    flush_pending_images!();
    Err(ServeError::OutputProcessing {
        request_id,
        message: "engine stream closed before a terminal event".to_string(),
    })
}

fn generation_text_finish_reason(
    reason: GenerationFinishReason,
    stop_reason: Option<String>,
) -> FinishReason {
    match reason {
        GenerationFinishReason::Eos | GenerationFinishReason::ImageDone => FinishReason::Stop(None),
        GenerationFinishReason::Stop => FinishReason::Stop(stop_reason.map(|reason| {
            reason
                .strip_prefix("token:")
                .and_then(|id| id.parse().ok())
                .map_or_else(|| StopReason::Text(reason), StopReason::TokenId)
        })),
        GenerationFinishReason::MaxTokens => FinishReason::Length,
        GenerationFinishReason::Cancelled => FinishReason::Cancelled,
        GenerationFinishReason::Aborted => FinishReason::Aborted,
        GenerationFinishReason::Repetition => FinishReason::Repetition,
        GenerationFinishReason::Error => FinishReason::Error,
    }
}

fn generation_finish_detail(reason: &GenerationFinishReason) -> &'static str {
    match reason {
        GenerationFinishReason::Eos => "eos",
        GenerationFinishReason::MaxTokens => "max_tokens",
        GenerationFinishReason::Stop => "stop",
        GenerationFinishReason::ImageDone => "image_done",
        GenerationFinishReason::Cancelled => "cancelled",
        GenerationFinishReason::Aborted => "aborted",
        GenerationFinishReason::Repetition => "repetition",
        GenerationFinishReason::Error => "error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_engine_gateway::GenerationTokenLogprob;
    use uniserve_engine_gateway::transport::GenerationEventStream;

    fn event_context() -> EventContext {
        EventContext {
            profile_id: "profile".to_string(),
            dialect_id: "bagel".to_string(),
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
                producer_op_id: event_seq,
                point_index: event_seq as u32,
                semantic_digest: format!("{event_seq:064x}"),
            },
        }
    }

    fn bagel_filter() -> OutputProcessorPolicy {
        let tokenizer = crate::test_support::configured_tokenizer();
        let dialect = uniserve_model_profile::dialect::resolve_generation_dialect(
            uniserve_model_profile::ModelDescription::Bagel,
            tokenizer.as_ref(),
        )
        .expect("resolve dialect")
        .expect("BAGEL dialect");
        OutputProcessorPolicy::Dialect(dialect.output_filter.clone())
    }

    #[tokio::test]
    async fn assembler_attaches_ranked_logprobs_to_text_delta() {
        let tokenizer = crate::test_support::configured_tokenizer();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(GenEvent::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        tx.try_send(GenEvent::TextToken {
            id: b'a' as u32,
            logprob: Some(-0.25),
            public_commit: None,
        })
        .unwrap();
        tx.try_send(GenEvent::TokenLogprobs {
            id: b'a' as u32,
            candidates: vec![GenerationTokenLogprob {
                token_id: b'a' as u32,
                logprob: -0.25,
                rank: 3,
            }],
        })
        .unwrap();
        tx.try_send(GenEvent::Finished {
            reason: GenerationFinishReason::MaxTokens,
            stop_reason: None,
            prompt_tokens: 1,
            completion_tokens: 1,
            images: 0,
        })
        .unwrap();
        drop(tx);

        let events = assemble_event_stream(
            "req".into(),
            event_context(),
            vec![b'p' as u32],
            tokenizer,
            false,
            true,
            true,
            TextDecodeOptions::default(),
            bagel_filter(),
            GenerationEventStream::new(rx),
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
        let tokenizer = crate::test_support::configured_tokenizer();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(GenEvent::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();

        let events = assemble_event_stream(
            "feedback-output".into(),
            event_context(),
            vec![b'p' as u32],
            tokenizer,
            false,
            false,
            false,
            TextDecodeOptions::default(),
            bagel_filter(),
            GenerationEventStream::new(rx),
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

        tx.send(GenEvent::TextToken {
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

        tx.send(GenEvent::ImageCommit { image_id: 0 })
            .await
            .unwrap();
        tx.send(GenEvent::ImageDone {
            image_id: 0,
            height: 1,
            width: 1,
            bytes: 3,
            sha256: "image".to_string(),
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

        tx.send(GenEvent::TextToken {
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
