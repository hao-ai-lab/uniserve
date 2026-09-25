//! Model-aware admission, engine submission, and output streaming.
//!
//! Requests follow one ownership chain:
//! `HTTP schema / text prompt -> InputProcessor -> GenerationRequest ->
//! EngineClient::submit_generation -> RequestOutput stream`.
//! Video requests follow the same chain with a `DiffusionRequest`,
//! `EngineClient::submit_media`, and `assembly::assemble_media_event_stream`.
//!
//! [`ServingRuntime`] owns each request from identity reservation to its terminal event. It
//! reserves the external identifier and an engine `RequestId` in the engine client's
//! `RequestRegistry` before preprocessing, runs model preprocessing on the blocking pool,
//! submits the prepared request, and wraps the engine receiver in the assembler and in
//! lifecycle tracking (`LifecycleGuard`, `LifecycleTrackedStream`).
//!
//! External cancel and abort commands are recorded in the registry and take precedence over
//! any later engine output: the caller sees `RequestOutput::Cancelled` or
//! `RequestOutput::Aborted` in place of the engine's own terminal event. Each request records
//! exactly one terminal outcome, including when the caller drops the pending future or the
//! response stream. Aggregate lifecycle counters reach the metrics route through
//! [`RuntimeMetricsSnapshot::state_counts`].

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod assembly;
/// Chat request rendering and structured output processing.
pub mod chat;
mod input;
mod model;
mod omni;
mod preprocessing;
mod sampling;
#[cfg(test)]
pub(crate) mod test_support;
/// Text tokenization, decoding, and sampling utilities.
pub mod text;

use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context as TaskContext, Poll};
use std::time::Instant;

use crate::engine_client::requests::RequestRegistry;
use crate::engine_client::{EngineClient, EventRx, StreamCancelCause};
use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uniserve_core::EngineCoreOutput;

pub use input::{
    DecodeControls, ImageGenControls, ImageInput, ModalitySelection, ModelEventIdentity,
    OutputDetail, OutputProcessorPolicy, PromptInput, ResponseOptions, SamplingConfig, StopConfig,
    TextPromptRequest,
};
pub use model::{
    InputProcessor, ModelSupport, ServedEndpoint, ServedFeature, ServedModality,
    ServedSamplingControl, WorkerCapabilities, default_video_seconds,
};

use crate::serving::chat::{AssistantBlockKind, AssistantContentBlock, Qwen3ChatOutputProcessor};
use crate::serving::omni::{SenseNovaOutputProcessor, SenseNovaTextDelta};
use crate::serving::text::output::stop_string_holdback_bytes;
use crate::serving::text::{
    DecodedLogprobs, DecodedPromptLogprobs, FinishReason, StopReason, TextDecodeOptions,
};

use assembly::{assemble_chat_event_stream, assemble_event_stream};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
/// Caller-visible request identifier used throughout serving.
///
/// The type performs no validation. The HTTP routes build it from the resolved request ID
/// with an endpoint prefix (for example `chatcmpl-`), and `RequestRegistry::register` refuses
/// an identifier that another live request already holds.
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

/// Boxed asynchronous stream of serving events.
pub type RequestOutputStream = Pin<Box<dyn Stream<Item = Result<RequestOutput>> + Send>>;
/// Result type returned by serving-runtime calls.
pub type Result<T> = std::result::Result<T, ServeError>;

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
    /// Requested computation exceeds loaded model capacity.
    #[error(transparent)]
    Capacity(#[from] uniserve_core::GenerationResourceError),
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
///
/// Returns `None` when both are absent, which leaves the request in the shared cache
/// partition. Otherwise the key is a 64-bit FNV-1a hash (the same function as
/// `model::stable_hash`) over length-prefixed material, so `("ab", "c")` and `("a", "bc")`
/// hash different bytes. The hash is deterministic across processes.
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

/// Tokenized input retains only the response processing its computation needs.
///
/// Media requests carry no `ResponseOptions`: their stream is assembled from the prompt
/// length and the runtime's model identity.
enum Prepared {
    Generation(Box<(uniserve_core::GenerationRequest, ResponseOptions)>),
    Diffusion(uniserve_core::DiffusionRequest),
}

impl From<(uniserve_core::GenerationRequest, ResponseOptions)> for Prepared {
    fn from((request, response): (uniserve_core::GenerationRequest, ResponseOptions)) -> Self {
        Self::Generation(Box::new((request, response)))
    }
}

/// Single-model serving runtime.
pub struct ServingRuntime {
    model: Arc<InputProcessor>,
    engine: Arc<EngineClient>,
    // Held only to keep the periodic logging task alive; dropping the logger aborts the task.
    _stats_logger: Option<Arc<crate::engine_client::generation::log_stats::StatsLogger>>,
    // Cumulative lifecycle counters shared with every request's lifecycle guard and assembler.
    metrics: Arc<RuntimeLifecycleMetrics>,
}

impl ServingRuntime {
    /// Creates a serving runtime for one resolved model and engine client.
    pub fn new(model: InputProcessor, engine: Arc<EngineClient>, log_stats: bool) -> Self {
        let stats_logger = log_stats.then(|| {
            Arc::new(
                crate::engine_client::generation::log_stats::StatsLogger::start(
                    engine.model_name().to_string(),
                    engine.engine_count(),
                ),
            )
        });
        Self {
            model: Arc::new(model),
            engine,
            _stats_logger: stats_logger,
            metrics: Arc::new(RuntimeLifecycleMetrics::default()),
        }
    }

    /// Returns the resolved model owned by this runtime.
    pub fn model(&self) -> &InputProcessor {
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

    /// Returns an aggregate point-in-time metrics snapshot.
    ///
    /// `active` is read from the request registry at call time; the other fields are
    /// cumulative counters since the runtime started.
    pub fn metrics_snapshot(&self) -> RuntimeMetricsSnapshot {
        let mut snapshot = self.metrics.snapshot();
        snapshot.active = self.engine.requests.active_count() as u64;
        snapshot
    }

    /// Returns current lifecycle statistics for one request.
    pub fn request_stats(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        self.engine.requests.stats(request_id)
    }

    /// Waits for one request to leave the active registry and returns its final snapshot.
    pub async fn drain_request(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        self.engine.requests.drain_request(request_id).await
    }

    /// Waits until every active request leaves the runtime.
    pub async fn drain(&self) {
        self.engine.requests.drain().await;
    }

    /// Preprocesses a chat request while retaining cancellation and identity ownership.
    pub async fn generate_chat(
        &self,
        request_id: ServeRequestId,
        request: crate::openai::ChatCompletionRequest,
    ) -> crate::openai::Result<RequestOutputStream> {
        crate::openai::chat_completions::validate_request_compat(
            &request,
            self.served_model_name(),
        )?;
        let input_id = request_id.clone();
        self.generate_with(request_id, move |model| {
            model
                .preprocess_chat_request(input_id, request)
                .map(Prepared::from)
        })
        .await
    }

    /// Preprocesses one image-generation API request and streams its output.
    pub async fn generate_image(
        &self,
        request_id: ServeRequestId,
        request: crate::openai::ImageGenerationRequest,
    ) -> crate::openai::Result<RequestOutputStream> {
        let input_id = request_id.clone();
        self.generate_with(request_id, move |model| {
            model
                .preprocess_image_request(input_id, request)
                .map(Prepared::from)
        })
        .await
    }

    /// Runs a programmatic text prompt, including model-specific context images.
    pub async fn generate_text(
        &self,
        request: TextPromptRequest,
    ) -> crate::openai::Result<RequestOutputStream> {
        self.generate_with(request.request_id.clone(), move |model| {
            model
                .preprocess_text_request(request)
                .map(Prepared::from)
                .map_err(crate::openai::serve_error_to_api)
        })
        .await
    }

    /// Tokenizes video prompts on the same blocking pool and lifecycle as text.
    pub async fn generate_video(
        &self,
        request_id: ServeRequestId,
        request: crate::openai::VideoGenerationRequest,
    ) -> crate::openai::Result<RequestOutputStream> {
        let input_id = request_id.clone();
        self.generate_with(request_id, move |model| {
            model
                .preprocess_video_request(&input_id, request)
                .map(Prepared::Diffusion)
        })
        .await
    }

    /// Owns the request across blocking preprocessing, submission, and public output.
    ///
    /// Registration precedes preprocessing, so a duplicate identifier is refused before any
    /// tokenization work and a cancel or abort can target a request that is still compiling.
    /// Every path after registration either returns a stream that carries the lifecycle guard
    /// or records a terminal outcome through it.
    async fn generate_with(
        &self,
        request_id: ServeRequestId,
        preprocess: impl FnOnce(&InputProcessor) -> crate::openai::Result<Prepared> + Send + 'static,
    ) -> crate::openai::Result<RequestOutputStream> {
        // `compile_us` and the lifecycle's elapsed times are measured from this instant, so
        // time spent queued for the blocking pool counts toward compilation.
        let compile_started = Instant::now();
        let identity = self.model.event_identity();

        let engine_request_id = self
            .engine
            .register_request(request_id.to_string(), Some(identity))
            .map_err(|error| {
                self.metrics.rejected.fetch_add(1, Ordering::Relaxed);
                match error {
                    crate::engine_client::Error::DuplicateRequestId { .. } => {
                        ServeError::DuplicateRequestId {
                            request_id: request_id.clone(),
                        }
                    }
                    error => ServeError::Engine(error),
                }
            })
            .map_err(crate::openai::serve_error_to_api)?;
        // Created before the first await: if the caller drops this future, the guard's `Drop`
        // records the request as cancelled, or as aborted when an abort is pending.
        let mut lifecycle = LifecycleGuard::new(
            request_id.clone(),
            Arc::clone(&self.metrics),
            Arc::clone(&self.engine.requests),
            compile_started,
        );

        // A control command that arrives during preprocessing ends the request immediately.
        // Dropping the `spawn_blocking` handle detaches the task rather than cancelling it, so
        // the preprocessing still runs to completion and its result is discarded.
        let model = Arc::clone(&self.model);
        let tokenize_request_id = request_id.clone();
        let tokenize_result = tokio::select! {
            terminal = self.engine.requests.wait_for_control(&request_id) => {
                return Ok(self.control_event_stream(request_id, terminal, lifecycle));
            }
            tokenized = tokio::task::spawn_blocking(move || preprocess(&model)) => {
                tokenized.unwrap_or_else(|error| Err(crate::openai::serve_error_to_api(ServeError::Tokenize {
                    request_id: tokenize_request_id,
                    source: TokenizeError::Task(error),
                })))
            }
        };
        let mut tokenized = match tokenize_result {
            Ok(tokenized) => tokenized,
            Err(error) => {
                lifecycle.terminal(
                    LifecycleTerminal::Rejected,
                    compile_started.elapsed().as_micros() as u64,
                );
                return Err(error);
            }
        };
        let compile_duration_us = compile_started.elapsed().as_micros() as u64;
        self.engine
            .requests
            .mark_submitting(&request_id, compile_duration_us);

        // Preprocessing fills a placeholder engine identifier. `EngineClient::submit_generation`
        // and `submit_media` accept only the identifier reserved at registration.
        match &mut tokenized {
            Prepared::Generation(prepared) => prepared.0.request_id = engine_request_id,
            Prepared::Diffusion(request) => request.request_id = engine_request_id,
        }
        self.submit_and_stream(tokenized, compile_duration_us, lifecycle)
            .await
            .map_err(crate::openai::serve_error_to_api)
    }

    /// Submits a tokenized request and wraps its output with model and lifecycle processing.
    ///
    /// A control command can arrive at any point, so the registry is checked before
    /// submission, after submission, and atomically when the request is accepted. A control
    /// found at any checkpoint yields an immediate terminal stream in place of engine output.
    /// A submission failure with no pending control records `Failed` and returns the error.
    ///
    /// A control found after a successful submission reaches the engine by dropping the
    /// unpolled output stream: its `EventRx` sends the engine a cancel at the acknowledged
    /// prefix, which releases the engine request for a cancel and an abort alike, and then
    /// releases the registry's engine side. The client's `Cancelled` or `Aborted` event comes
    /// from the registry's recorded control, not from the engine.
    async fn submit_and_stream(
        &self,
        prepared: Prepared,
        compile_duration_us: u64,
        mut lifecycle: LifecycleGuard,
    ) -> Result<RequestOutputStream> {
        let request_id = lifecycle.request_id.clone();
        if let Some(terminal) = self.engine.requests.control_terminal(&request_id) {
            return Ok(self.control_event_stream(request_id, terminal, lifecycle));
        }

        let stream_result: Result<RequestOutputStream> = match prepared {
            Prepared::Generation(prepared) => {
                let (request, response) = *prepared;
                let ResponseOptions {
                    request_id: _,
                    tokenizer,
                    prompt_token_ids,
                    decode,
                    emit_token_ids,
                    prompt_logprobs_requested,
                    generated_logprobs_requested,
                    output_processor,
                    identity,
                    cache,
                    resources,
                } = response;
                let engine_stream = self
                    .engine
                    .submit_generation(request_id.to_string(), request)
                    .await;

                let event_context = EventContext {
                    served_name: identity.served_name.clone(),
                    description: identity.description.clone(),
                    compile_duration_us,
                    started: lifecycle.started,
                    cache,
                    resources,
                    metrics: Arc::clone(&self.metrics),
                };

                match engine_stream {
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
                        let output: RequestOutputStream = match output_processor {
                            OutputProcessorPolicy::Qwen3(processor) => {
                                Box::pin(assemble_chat_event_stream(assembly, processor))
                            }
                            // `assemble_event_stream` applies the remaining policies as its
                            // output sink.
                            output_processor => {
                                Box::pin(assemble_event_stream(assembly, output_processor))
                            }
                        };
                        Ok(output)
                    }
                    Err(error) => Err(ServeError::Engine(error)),
                }
            }
            Prepared::Diffusion(request) => {
                let prompt_tokens = request.prompt_token_ids.len();
                let context = EventContext {
                    served_name: self.model.served_model_name().to_owned(),
                    description: self.model.event_identity().description,
                    compile_duration_us,
                    started: lifecycle.started,
                    cache: CacheAccounting::default(),
                    resources: ResourceAccounting::default(),
                    metrics: Arc::clone(&self.metrics),
                };
                self.engine
                    .submit_media(request_id.to_string(), request)
                    .await
                    .map(|stream| {
                        Box::pin(assembly::assemble_media_event_stream(
                            request_id.clone(),
                            context,
                            prompt_tokens,
                            stream,
                        )) as RequestOutputStream
                    })
                    .map_err(ServeError::Engine)
            }
        };

        // A control recorded before `claim_submission` was not forwarded to the engine (one
        // recorded after it was). Dropping the unpolled stream drops its `EventRx`, which
        // delivers it as a cancel at the acknowledged prefix and releases the registry's
        // engine side. A failed submission has no engine request to control.
        if let Some(terminal) = self.engine.requests.control_terminal(&request_id) {
            if let Ok(stream) = stream_result {
                drop(stream);
            }
            return Ok(self.control_event_stream(request_id, terminal, lifecycle));
        }
        let stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                lifecycle.terminal(
                    LifecycleTerminal::Failed,
                    lifecycle.started.elapsed().as_micros() as u64,
                );
                return Err(error);
            }
        };

        // `accept` refuses under the registry lock when a control was recorded after the
        // previous check or when the request has no lifecycle record. Such a control was
        // already forwarded to the engine when it was recorded; dropping the stream cancels
        // the engine request in any case.
        if !self.engine.requests.accept(&request_id) {
            let terminal = self
                .engine
                .requests
                .control_terminal(&request_id)
                .unwrap_or(LifecycleTerminal::Cancelled);
            drop(stream);
            return Ok(self.control_event_stream(request_id, terminal, lifecycle));
        }

        // Layering: the assembler produces public events, `control_aware_event_stream` ends
        // them promptly on a control, and `LifecycleTrackedStream` records statistics and the
        // single terminal outcome.
        self.metrics.accepted.fetch_add(1, Ordering::Relaxed);
        let stream = Box::pin(control_aware_event_stream(
            request_id.clone(),
            Arc::clone(&self.engine.requests),
            stream,
        )) as RequestOutputStream;
        Ok(Box::pin(LifecycleTrackedStream::new(stream, lifecycle)))
    }

    /// Builds an immediately terminal stream for a request controlled before engine ownership.
    ///
    /// Callers pass a control outcome (`Cancelled` or `Aborted`); any other terminal maps to
    /// `RequestOutput::Cancelled`.
    fn control_event_stream(
        &self,
        request_id: ServeRequestId,
        terminal: LifecycleTerminal,
        lifecycle: LifecycleGuard,
    ) -> RequestOutputStream {
        let event = match terminal {
            LifecycleTerminal::Aborted => RequestOutput::Aborted {
                request_id: request_id.clone(),
            },
            LifecycleTerminal::Cancelled
            | LifecycleTerminal::Finished
            | LifecycleTerminal::Rejected
            | LifecycleTerminal::Failed => RequestOutput::Cancelled {
                request_id: request_id.clone(),
            },
        };
        let inner = Box::pin(futures::stream::iter([Ok(event)])) as RequestOutputStream;
        Box::pin(LifecycleTrackedStream::new(inner, lifecycle))
    }

    /// Cancels a live request at its acknowledged output prefix.
    pub async fn cancel(
        &self,
        request_id: impl Into<ServeRequestId>,
    ) -> std::result::Result<(), ServeControlError> {
        let request_id = request_id.into();
        self.engine.cancel_request(&request_id).await?;
        Ok(())
    }

    /// Aborts a live request immediately.
    pub async fn abort(
        &self,
        request_id: impl Into<ServeRequestId>,
    ) -> std::result::Result<(), ServeControlError> {
        let request_id = request_id.into();
        self.engine.abort_request(&request_id).await?;
        Ok(())
    }

    /// Drains requests and shuts down the backing engine.
    pub async fn shutdown(self) -> std::result::Result<(), crate::engine_client::Error> {
        self.drain().await;
        self.engine.shutdown().await
    }
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
/// Point-in-time lifecycle state, usage, and timing statistics for one request.
///
/// `RequestRegistry` maintains it from the public events it observes and retains it for a
/// bounded number of completed requests.
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
    pub(crate) fn submitting(
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

/// Request-scoped metadata the assemblers stamp onto public events and usage.
#[derive(Debug, Clone)]
struct EventContext {
    started: Instant,
    served_name: String,
    description: String,
    compile_duration_us: u64,
    cache: CacheAccounting,
    resources: ResourceAccounting,
    // The assemblers increment `scheduled` when the engine reports scheduling.
    metrics: Arc<RuntimeLifecycleMetrics>,
}

/// Engine receiver and the response options a generation assembler consumes.
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

/// Cumulative per-runtime lifecycle counters. They use relaxed ordering: each is read
/// independently for reporting and none orders other memory.
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
/// Aggregate request lifecycle counts: the active requests plus cumulative milestone and
/// outcome counters.
pub struct RuntimeMetricsSnapshot {
    /// Number of registered requests that have not reached a terminal outcome, including
    /// requests still in preprocessing.
    pub active: u64,
    /// Cumulative number of engine-accepted requests.
    pub accepted: u64,
    /// Cumulative number of scheduled requests.
    pub scheduled: u64,
    /// Cumulative number of normally completed requests.
    pub finished: u64,
    /// Cumulative number of requests rejected before execution, including duplicate
    /// identifiers and preprocessing failures.
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

/// Terminal-outcome latch for one registered request.
///
/// The guard moves from `generate_with` into the response stream. Its first `terminal` call
/// completes the registry record and increments one outcome counter; later calls do nothing.
/// Dropping it without a terminal records a cancellation or abort.
struct LifecycleGuard {
    request_id: ServeRequestId,
    metrics: Arc<RuntimeLifecycleMetrics>,
    requests: Arc<RequestRegistry>,
    started: Instant,
    terminal: bool,
}

/// Response stream that feeds each event to the request registry and latches the terminal
/// outcome, replacing the engine's terminal with a control terminal when one takes precedence.
struct LifecycleTrackedStream {
    inner: RequestOutputStream,
    lifecycle: LifecycleGuard,
}

impl LifecycleTrackedStream {
    /// Transfers the preprocessing guard to the caller's response stream.
    fn new(inner: RequestOutputStream, lifecycle: LifecycleGuard) -> Self {
        Self { inner, lifecycle }
    }
}

impl Stream for LifecycleTrackedStream {
    type Item = Result<RequestOutput>;

    /// Polls the inner stream while enforcing control precedence and terminal accounting.
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(mut event))) => {
                let elapsed_us = self.lifecycle.started.elapsed().as_micros() as u64;

                // A pending control replaces this event and ends the stream; replacing
                // `inner` drops the previous stream and any engine receiver it holds.
                if let Some(control) =
                    self.lifecycle
                        .requests
                        .observe(&self.lifecycle.request_id, &event, elapsed_us)
                {
                    let terminal = self.lifecycle.terminal(control, elapsed_us);
                    event = control_terminal_event(&self.lifecycle.request_id, terminal);
                    self.inner = Box::pin(futures::stream::empty());
                    return Poll::Ready(Some(Ok(event)));
                }

                // `terminal` returns an outcome other than the engine's only when the registry
                // holds a cancel or abort, which is why `control_terminal_event` accepts only
                // those two outcomes.
                let terminal = match &event {
                    RequestOutput::Finished { .. } => Some(LifecycleTerminal::Finished),
                    RequestOutput::Rejected { .. } => Some(LifecycleTerminal::Rejected),
                    RequestOutput::Cancelled { .. } => Some(LifecycleTerminal::Cancelled),
                    RequestOutput::Aborted { .. } => Some(LifecycleTerminal::Aborted),
                    RequestOutput::Failed { .. } => Some(LifecycleTerminal::Failed),
                    _ => None,
                };
                if let Some(terminal) = terminal {
                    let actual = self.lifecycle.terminal(terminal, elapsed_us);
                    if actual != terminal {
                        event = control_terminal_event(&self.lifecycle.request_id, actual);
                    }
                    self.inner = Box::pin(futures::stream::empty());
                }
                Poll::Ready(Some(Ok(event)))
            }
            // An error is a failure unless a control takes precedence, in which case the caller
            // sees the control terminal instead of the error.
            Poll::Ready(Some(Err(error))) => {
                let elapsed_us = self.lifecycle.started.elapsed().as_micros() as u64;
                let actual = self
                    .lifecycle
                    .terminal(LifecycleTerminal::Failed, elapsed_us);
                self.inner = Box::pin(futures::stream::empty());
                if actual == LifecycleTerminal::Failed {
                    Poll::Ready(Some(Err(error)))
                } else {
                    Poll::Ready(Some(Ok(control_terminal_event(
                        &self.lifecycle.request_id,
                        actual,
                    ))))
                }
            }
            // An inner stream that ends without a terminal event is recorded as failed; only a
            // control outcome is surfaced to the caller as a synthesized event.
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
    /// Creates a guard for a request already registered in `requests`.
    fn new(
        request_id: ServeRequestId,
        metrics: Arc<RuntimeLifecycleMetrics>,
        requests: Arc<RequestRegistry>,
        started: Instant,
    ) -> Self {
        Self {
            request_id,
            metrics,
            requests,
            started,
            terminal: false,
        }
    }

    /// Records the first terminal transition and increments its aggregate metric exactly once.
    ///
    /// Returns the recorded outcome: `RequestRegistry::complete` converts `kind` into
    /// `Cancelled` or `Aborted` when a control is pending. After the first call, returns `kind`
    /// unchanged without recording anything.
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

/// Terminal outcome of one request, as recorded by `LifecycleGuard`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LifecycleTerminal {
    Finished,
    Rejected,
    Cancelled,
    Aborted,
    Failed,
}

/// Returns the terminal lifecycle event represented by a state.
///
/// Pending `Cancelling` and `Aborting` states map to their eventual outcomes.
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
fn control_terminal_event(
    request_id: &ServeRequestId,
    terminal: LifecycleTerminal,
) -> RequestOutput {
    match terminal {
        LifecycleTerminal::Aborted => RequestOutput::Aborted {
            request_id: request_id.clone(),
        },
        LifecycleTerminal::Cancelled => RequestOutput::Cancelled {
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
    /// Records an unfinished request as aborted when an abort is pending and as cancelled
    /// otherwise. This covers a caller that drops the pending future or the response stream.
    fn drop(&mut self) {
        if !self.terminal {
            let elapsed_us = self.started.elapsed().as_micros() as u64;
            let terminal = self.requests.drop_terminal(&self.request_id);
            let _ = self.terminal(terminal, elapsed_us);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Cache policy and encoder-cache pin count resolved for one request during preprocessing.
pub struct CacheAccounting {
    /// Whether the request may reuse existing cache entries.
    pub read_enabled: bool,
    /// Whether the request may publish reusable cache entries.
    pub write_enabled: bool,
    /// Number of encoder-cache entries pinned for the request lifetime.
    pub encoder_pin_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
/// KV, latent, and encoder-cache requirements computed for one request during preprocessing.
/// Media requests report `ResourceAccounting::default()`.
pub struct ResourceAccounting {
    /// KV-token capacity required for the request.
    pub expected_kv_tokens: u64,
    /// Latent image units reserved for generation.
    pub image_latent_units: u64,
    /// Number of encoder-cache entries retained by the request.
    pub encoder_cache_pins: usize,
    /// For a generation request, false when generated images feed back into the request's
    /// context (`GenerationRequest::feeds_back_images`).
    pub replayable: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Queue, execution, and end-to-end duration totals.
pub struct RuntimeTimings {
    /// Request validation and tokenization duration in microseconds, including time queued
    /// for the blocking pool.
    pub compile_us: u64,
    /// Queue duration in microseconds, when scheduling timestamps are available.
    pub queue_us: Option<u64>,
    /// Time from request start to the first visible output in microseconds.
    pub first_visible_output_us: Option<u64>,
    /// End-to-end request duration in microseconds.
    pub total_us: u64,
}

/// Runtime event consumed by the HTTP response layer.
///
/// A stream returned by [`ServingRuntime`] yields at most one terminal variant (`Finished`,
/// `Rejected`, `Cancelled`, `Aborted`, or `Failed`) or one error, and then ends.
#[derive(Debug, Clone)]
pub enum RequestOutput {
    /// Immutable generated media retained through response delivery.
    Artifact(uniserve_core::ArtifactEvent),
    /// Latest diffusion progress: the engine-reported phase and the number of completed
    /// denoising steps.
    MediaProgress { phase: String, completed_steps: u32 },
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
        /// Engine wall-clock time, in UNIX seconds, at which the request entered the
        /// scheduler's waiting queue.
        queued_at: Option<f64>,
        /// Engine wall-clock time, in UNIX seconds, at which the scheduler admitted the
        /// request for execution.
        scheduled_at: Option<f64>,
        /// Cache policy and pin counts.
        cache: CacheAccounting,
        /// KV and media resources reserved for the request.
        resources: ResourceAccounting,
    },
    /// Newly decoded user-visible text.
    TextDelta {
        /// Newly visible decoded text.
        text: String,
        /// Token identifiers represented by this update.
        token_ids: Vec<u32>,
        /// Per-position candidate log probabilities, when requested.
        logprobs: Option<DecodedLogprobs>,
    },
    /// Newly decoded text consumed by a model protocol rather than exposed as output.
    InternalTextDelta {
        /// Newly decoded internal text.
        text: String,
    },
    /// Newly decoded structured reasoning text.
    ReasoningDelta {
        /// Newly decoded reasoning text.
        text: String,
    },
    /// A structured assistant output block has opened.
    OutputBlockStart {
        /// Zero-based content-block index.
        index: usize,
        /// Semantic kind of the opened block.
        kind: AssistantBlockKind,
    },
    /// A structured assistant output block has closed.
    OutputBlockEnd {
        /// Zero-based content-block index.
        index: usize,
        /// Complete normalized block content.
        block: AssistantContentBlock,
    },
    /// A structured function-tool call has opened.
    ToolCallStart {
        /// Zero-based tool-call index.
        index: usize,
        /// Request-local tool-call identifier.
        id: String,
        /// Selected function name.
        name: String,
    },
    /// Newly decoded serialized tool arguments.
    ToolCallArgumentsDelta {
        /// Zero-based tool-call index.
        index: usize,
        /// Newly decoded argument text.
        delta: String,
    },
    /// A structured function-tool call has closed.
    ToolCallEnd {
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
        /// Request-local image identifier.
        image_id: String,
        /// Completed step number.
        step: u32,
        /// Elapsed request duration in microseconds.
        elapsed_us: u64,
    },
    /// Generated image state has been committed to the model context.
    ImageCommit {
        /// Request-local image identifier.
        image_id: String,
        /// Elapsed request duration in microseconds.
        elapsed_us: u64,
    },
    /// One generated image is complete and available to the response layer.
    ImageDone {
        /// Request-local image identifier.
        image_id: String,
        /// Output width in pixels, when reported.
        width: Option<u32>,
        /// Output height in pixels, when reported.
        height: Option<u32>,
        /// Encoded image size in bytes, when reported.
        bytes: Option<u64>,
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
        /// Semantic terminal status.
        reason: FinishStatus,
        /// Optional model- or engine-provided terminal detail.
        finish_detail: Option<String>,
    },
    /// The request was rejected before execution.
    Rejected {
        /// Identifier of the rejected request.
        request_id: ServeRequestId,
        /// Whether the request is unservable or the engine is full.
        kind: uniserve_core::RejectionKind,
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
                    StopReason::Token(id) => StopCause::TokenId(*id),
                    StopReason::String(text) => StopCause::Text(text.clone()),
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
///
/// Waiting on the registry alongside the engine stream ends the response promptly even while
/// the engine produces no events. Returning drops the inner stream and with it the engine
/// receiver.
async fn control_aware_event_stream(
    request_id: ServeRequestId,
    requests: Arc<RequestRegistry>,
    mut stream: RequestOutputStream,
    mut y: TryYielder<RequestOutput, ServeError>,
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
                    // A control recorded while this event was pending still wins over it.
                    if let Some(terminal) = requests.control_terminal(&request_id) {
                        y.yield_ok(control_terminal_event(&request_id, terminal)).await;
                        return Ok(());
                    }
                    let terminal = matches!(
                        event,
                        RequestOutput::Finished { .. }
                            | RequestOutput::Rejected { .. }
                            | RequestOutput::Cancelled { .. }
                            | RequestOutput::Aborted { .. }
                            | RequestOutput::Failed { .. }
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
            started: Instant::now(),
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
            metrics: Arc::new(RuntimeLifecycleMetrics::default()),
        }
    }

    #[tokio::test]
    async fn assembler_attaches_ranked_logprobs_to_text_delta() {
        let tokenizer = crate::serving::test_support::configured_tokenizer();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(EngineCoreOutput::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        tx.try_send(EngineCoreOutput::TextToken {
            id: b'a' as u32,
            logprob: Some(-0.25),
        })
        .unwrap();
        tx.try_send(EngineCoreOutput::TokenLogprobs {
            id: b'a' as u32,
            candidates: vec![TokenLogprob {
                token_id: b'a' as u32,
                logprob: -0.25,
                rank: 3,
            }],
        })
        .unwrap();
        tx.try_send(EngineCoreOutput::Finished {
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
            OutputProcessorPolicy::None,
        )
        .collect::<Vec<_>>()
        .await;

        let delta = events
            .iter()
            .find_map(|event| match event {
                Ok(RequestOutput::TextDelta {
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
            Some(Ok(RequestOutput::Finished {
                reason: FinishStatus::Length,
                ..
            }))
        ));
    }

    /// A request that finishes without publishing a text token (the engine
    /// withholds an EOS or excluded stop token, and image-only generation
    /// publishes none) returns no text. The prompt seeds decoding only.
    #[tokio::test]
    async fn assembler_emits_no_text_when_no_token_was_generated() {
        let tokenizer = crate::serving::test_support::configured_tokenizer();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(EngineCoreOutput::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        tx.try_send(EngineCoreOutput::Finished {
            reason: uniserve_core::FinishReason::Eos,
            stop_reason: None,
            prompt_tokens: 1,
            completion_tokens: 1,
            images: 0,
        })
        .unwrap();
        drop(tx);

        let events = assemble_event_stream(
            StreamInput {
                request_id: "eos-first".into(),
                event_context: event_context(),
                prompt_token_ids: vec![b'p' as u32],
                tokenizer,
                prompt_logprobs_requested: false,
                generated_logprobs_requested: false,
                emit_token_ids: false,
                decode_options: TextDecodeOptions::default(),
                stream: EventRx::from_receiver(rx),
            },
            OutputProcessorPolicy::None,
        )
        .collect::<Vec<_>>()
        .await;

        assert!(events.iter().all(std::result::Result::is_ok));
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                Ok(RequestOutput::TextDelta { text, .. }) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "");
        assert!(matches!(
            events.last(),
            Some(Ok(RequestOutput::Finished {
                reason: FinishStatus::Stop { .. },
                ..
            }))
        ));
    }

    /// SenseNova filtering holds back a trailing fragment that could begin a
    /// `<think>`, `</think>`, `<answer>` or `</answer>` delimiter. At the end of
    /// the stream no delimiter can complete, so the held fragment is visible
    /// text, whether the engine finished the request or a stop string did.
    #[tokio::test]
    async fn sensenova_assembler_releases_a_trailing_delimiter_prefix_at_the_end() {
        let policy = crate::profile::omni::OutputFilterPolicy {
            reasoning: Some(crate::profile::omni::DelimitedTextPolicy {
                start: "<think>".to_string(),
                end: "</think>".to_string(),
            }),
            visible_wrappers: vec![crate::profile::omni::DelimitedTextPolicy {
                start: "<answer>".to_string(),
                end: "</answer>".to_string(),
            }],
        };
        let text_token = |byte: u8| EngineCoreOutput::TextToken {
            id: u32::from(byte),
            logprob: None,
        };
        let finished = EngineCoreOutput::Finished {
            reason: uniserve_core::FinishReason::MaxTokens,
            stop_reason: None,
            prompt_tokens: 1,
            completion_tokens: 3,
            images: 0,
        };
        let cases = [
            (
                "engine-finished",
                vec![
                    text_token(b'x'),
                    text_token(b' '),
                    text_token(b'<'),
                    finished,
                ],
                None,
                "x <",
            ),
            (
                "stop-string",
                vec![text_token(b'x'), text_token(b'<'), text_token(b'!')],
                Some(vec!["!".to_string()]),
                "x<",
            ),
        ];

        for (request_id, events, stop_strings, expected) in cases {
            let (tx, rx) = tokio::sync::mpsc::channel(8);
            tx.try_send(EngineCoreOutput::Scheduled {
                queued_at: 1.0,
                scheduled_at: 2.0,
            })
            .unwrap();
            for event in events {
                tx.try_send(event).unwrap();
            }
            drop(tx);

            let outputs = assemble_event_stream(
                StreamInput {
                    request_id: request_id.into(),
                    event_context: event_context(),
                    prompt_token_ids: vec![b'p' as u32],
                    tokenizer: crate::serving::test_support::configured_tokenizer(),
                    prompt_logprobs_requested: false,
                    generated_logprobs_requested: false,
                    emit_token_ids: false,
                    decode_options: TextDecodeOptions {
                        stop_strings,
                        ..TextDecodeOptions::default()
                    },
                    stream: EventRx::from_receiver(rx),
                },
                OutputProcessorPolicy::SenseNova(policy.clone()),
            )
            .collect::<Vec<_>>()
            .await;

            assert!(
                outputs.iter().all(std::result::Result::is_ok),
                "{request_id}"
            );
            let visible: String = outputs
                .iter()
                .filter_map(|output| match output {
                    Ok(RequestOutput::TextDelta { text, .. }) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(visible, expected, "{request_id}");
            assert!(
                matches!(outputs.last(), Some(Ok(RequestOutput::Finished { .. }))),
                "{request_id}"
            );
        }
    }

    /// An engine failure of a running request (`Error`, or `ArtifactUnavailable`
    /// when its output cannot be delivered) is a `Failed` terminal carrying the
    /// engine message on every generation assembler, including the Qwen3 chat
    /// assembler, rather than an output-processing error.
    #[tokio::test]
    async fn assemblers_report_an_engine_failure_as_a_failed_terminal() {
        let failures = [
            EngineCoreOutput::Error {
                message: "worker call failed".to_string(),
            },
            EngineCoreOutput::ArtifactUnavailable {
                message: "worker call failed".to_string(),
            },
        ];
        for chat in [false, true] {
            for failure in failures.clone() {
                let tokenizer = crate::serving::test_support::configured_tokenizer();
                let (tx, rx) = tokio::sync::mpsc::channel(8);
                tx.try_send(EngineCoreOutput::Scheduled {
                    queued_at: 1.0,
                    scheduled_at: 2.0,
                })
                .unwrap();
                tx.try_send(EngineCoreOutput::TextToken {
                    id: b'a' as u32,
                    logprob: None,
                })
                .unwrap();
                tx.try_send(failure).unwrap();
                drop(tx);

                let input = StreamInput {
                    request_id: "engine-failure".into(),
                    event_context: event_context(),
                    prompt_token_ids: vec![b'p' as u32],
                    tokenizer: Arc::clone(&tokenizer),
                    prompt_logprobs_requested: false,
                    generated_logprobs_requested: false,
                    emit_token_ids: false,
                    decode_options: TextDecodeOptions::default(),
                    stream: EventRx::from_receiver(rx),
                };
                let events = if chat {
                    let processor = crate::serving::chat::Qwen3ChatOutputProcessor::new(
                        &mut crate::serving::chat::ChatRequest::for_test(),
                        tokenizer,
                        true,
                    )
                    .unwrap();
                    // Boxed like the runtime boxes it, since the chat pipeline
                    // is a large future.
                    Box::pin(assemble_chat_event_stream(input, processor))
                        .collect::<Vec<_>>()
                        .await
                } else {
                    assemble_event_stream(input, OutputProcessorPolicy::None)
                        .collect::<Vec<_>>()
                        .await
                };

                assert!(
                    events.iter().all(std::result::Result::is_ok),
                    "chat={chat}: {:?}",
                    events.last()
                );
                assert!(
                    matches!(
                        events.last(),
                        Some(Ok(RequestOutput::Failed { message, .. }))
                            if message == "worker call failed"
                    ),
                    "chat={chat}: {:?}",
                    events.last()
                );
            }
        }
    }

    /// Committed image events are held back and published immediately before the next text
    /// token (or before a terminal event or the end of the engine stream), so they stay
    /// ordered ahead of that token.
    #[tokio::test]
    async fn assembler_publishes_an_image_with_its_next_text_token() {
        let tokenizer = crate::serving::test_support::configured_tokenizer();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(EngineCoreOutput::Scheduled {
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
            OutputProcessorPolicy::None,
        );
        tokio::pin!(events);
        assert!(matches!(
            events.next().await,
            Some(Ok(RequestOutput::Accepted { .. }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(RequestOutput::Scheduled { .. }))
        ));

        tx.send(EngineCoreOutput::TextToken {
            id: b'a' as u32,
            logprob: None,
        })
        .await
        .unwrap();
        assert!(matches!(
            events.next().await,
            Some(Ok(RequestOutput::TextDelta { text, .. })) if text == "a"
        ));

        tx.send(EngineCoreOutput::ImageCommit { image_id: 0 })
            .await
            .unwrap();
        tx.send(EngineCoreOutput::ImageDone {
            image_id: 0,
            height: 1,
            width: 1,
            bytes: 3,
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

        tx.send(EngineCoreOutput::TextToken {
            id: b'b' as u32,
            logprob: None,
        })
        .await
        .unwrap();
        assert!(matches!(
            events.next().await,
            Some(Ok(RequestOutput::ImageCommit { .. }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(RequestOutput::ImageDone { .. }))
        ));
    }
}
