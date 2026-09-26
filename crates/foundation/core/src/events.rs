//! Requests and lifecycle events exchanged across the public engine boundary.
//!
//! The server submits each engine request as a [`Request`], choosing the
//! variant from the engine's [`RuntimeFamily`], and receives that request's
//! [`EngineCoreOutput`] events in order. `Finished`, `Rejected`, and `Error`
//! are terminal: the engine's `EventRx` marks the request finished when it
//! receives one of them.

use serde::{Deserialize, Serialize};

use crate::{RequestId, SharedMedia};
use std::sync::Arc;

/// Terminal cause for one generation request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Generation completed normally.
    Completed,
    /// An end-of-sequence token ended text generation.
    Eos,
    /// The request reached its generated-token limit.
    MaxTokens,
    /// A configured stop token or string matched.
    Stop,
    /// The image-only generation objective completed.
    ImageDone,
    /// The client cancelled the request at a public output boundary.
    Cancelled,
    /// The server aborted the request immediately.
    Aborted,
    /// Repetition policy ended generation.
    Repetition,
    /// Execution or output processing failed.
    Error,
}

/// Typed payload identifying the exact stop condition that ended generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum StopReason {
    /// Vocabulary token that matched a stop condition.
    Token(u32),
    /// Text sequence that matched a stop condition.
    String(String),
}

/// One ranked vocabulary candidate at a generated or prompt token position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    /// Vocabulary token identity.
    pub token_id: u32,
    /// Natural-log probability assigned to the token.
    pub logprob: f32,
    /// One-based competition rank at the position; tied log probabilities
    /// share a rank.
    pub rank: u32,
}

/// Ranked candidates for one scored token position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionLogprobs {
    /// Ranked token candidates for this position.
    pub entries: Vec<TokenLogprob>,
}

/// Typed event stream emitted by every engine runtime family.
#[derive(Debug, Clone)]
pub enum EngineCoreOutput {
    /// Reports scheduler admission timing.
    Scheduled {
        /// Time the request entered the queue, in seconds since the Unix epoch.
        queued_at: f64,
        /// Time the request was admitted, in seconds since the Unix epoch.
        scheduled_at: f64,
    },
    /// Publishes one generated text token.
    TextToken {
        /// Vocabulary token identity.
        id: u32,
        /// Token log probability when requested.
        logprob: Option<f32>,
    },
    /// Publishes generated text tokens committed together, in order, such as
    /// a stopped block-diffusion canvas; they carry no log probabilities.
    /// Consumers treat them as consecutive `TextToken`s.
    TextTokens {
        /// Vocabulary token identities.
        ids: Vec<u32>,
    },
    /// Publishes ranked candidates associated with a text token.
    TokenLogprobs {
        /// Generated token identity this payload scores.
        id: u32,
        /// Ranked vocabulary candidates.
        candidates: Vec<TokenLogprob>,
    },
    /// Publishes scores for prompt positions processed during prefill.
    PromptLogprobs {
        /// Scored prompt positions in prompt order.
        positions: Vec<PositionLogprobs>,
    },
    /// Publishes a readout request's answer, once, before its `Finished`.
    Readout {
        /// Natural-log probability of every candidate of every slot, in
        /// row, slot, and candidate order
        /// (`GenerationRequest::readout_candidates` values).
        candidate_logprobs: Vec<f32>,
    },
    /// Opens the lifecycle of one generated image.
    ///
    /// The engine emits it together with the image's first committed
    /// denoising step, before that step's `ImageStep`.
    ImageBegin {
        /// Request-local image identity.
        image_id: u32,
        /// Output height in pixels.
        height: u32,
        /// Output width in pixels.
        width: u32,
        /// Configured denoising step count.
        steps: u16,
    },
    /// Reports one committed diffusion step. Each committed step is reported
    /// once, in increasing order.
    ImageStep {
        /// Request-local image identity.
        image_id: u32,
        /// One-based committed step number.
        step: u16,
    },
    /// Marks the transition from denoising to image materialization.
    ///
    /// The engine emits it when the image-decoding call returns, before the
    /// image's `ImageDone`.
    ImageCommit {
        /// Request-local image identity.
        image_id: u32,
    },
    /// Publishes a materialized PNG image.
    ///
    /// The engine reads `height`, `width`, and `bytes` from the validated PNG
    /// payload; a payload that fails validation finishes the request with
    /// [`FinishReason::Error`] instead.
    ImageDone {
        /// Request-local image identity.
        image_id: u32,
        /// Decoded image height in pixels.
        height: u32,
        /// Decoded image width in pixels.
        width: u32,
        /// PNG payload size in bytes.
        bytes: u64,
        /// Base64-encoded PNG payload.
        pixels_png_b64: String,
    },
    /// Reports actual media computation progress committed by the engine.
    ///
    /// The engine emits it for diffusion media requests after each valid media
    /// call result, until the request fails. While delivery is backed up, a
    /// newer progress event replaces an undelivered one that is still the most
    /// recently queued event.
    MediaProgress {
        /// Current numerical or output phase: `preparing`, `denoising`,
        /// `decoding`, or `finalizing`.
        phase: String,
        /// Number of completed denoising steps.
        completed_steps: u32,
    },
    /// Publishes a transport-backed media artifact.
    Artifact(ArtifactEvent),
    /// Generated media could not be acquired from its published storage.
    ///
    /// Not terminal by itself: the engine also fails the call whose output
    /// could not be acquired.
    ArtifactUnavailable {
        /// Storage error presented to the artifact consumer.
        message: String,
    },
    /// Terminates a successfully accepted request.
    Finished {
        /// Terminal generation cause.
        reason: FinishReason,
        /// Exact matched stop condition, when applicable.
        stop_reason: Option<StopReason>,
        /// Number of prompt tokens charged to the request.
        prompt_tokens: usize,
        /// Number of generated tokens charged to the request.
        completion_tokens: usize,
        /// Number of generated images.
        images: usize,
    },
    /// Rejects a request before execution begins.
    Rejected {
        /// Whether the request itself is unservable or the engine is full.
        kind: RejectionKind,
        /// Human-readable rejection reason.
        message: String,
    },
    /// Terminates a request after an execution failure.
    Error {
        /// Human-readable failure reason.
        message: String,
    },
}

/// Why admission refused a request.
///
/// Callers map the kind to their own vocabulary: an invalid request fails the
/// same way on every retry, while an overloaded engine may accept the same
/// request once its waiting queue drains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectionKind {
    /// The deployment cannot serve the request as specified.
    Invalid,
    /// The waiting queue is at its bound.
    Overloaded,
}

/// Kind of media referenced by an artifact event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    /// Still-image artifact.
    Image,
    /// Video artifact.
    Video,
    /// Audio artifact.
    Audio,
}

/// Runtime family selected once for an engine configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeFamily {
    /// Autoregressive text runtime.
    Ar,
    /// Diffusion-only media runtime.
    Diffusion,
    /// Unified multimodal understanding and generation runtime.
    Umm,
    /// Block-diffusion token runtime: prompt prefill and denoising passes
    /// over token canvases that read the prompt's KV cache.
    BlockDiffusion,
}

/// One caller-visible artifact retaining its immutable shared-storage mapping.
#[derive(Debug, Clone)]
pub struct ArtifactEvent {
    /// Semantic media type of the artifact.
    pub media_kind: MediaKind,
    /// MIME content type of the artifact payload.
    pub content_type: String,
    /// Mapped payload retained through delivery, job retention, and active downloads.
    pub media: Arc<SharedMedia>,
}

/// Effective diffusion controls, resolved once by model preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionSamplingParams {
    /// Number of output frames after model-specific alignment.
    pub num_frames: u32,
    /// Video media units the decoder reconstructs, independent of GPU rank count.
    pub video_units: u32,
    /// Number of denoising steps in the trajectory.
    pub num_inference_steps: u32,
    /// Deterministic request-level noise seed.
    pub seed: u64,
}

/// Media request. Final media bytes are returned through shared storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionRequest {
    /// Engine request identity.
    pub request_id: RequestId,
    /// Tokenized media prompt.
    pub prompt_token_ids: Vec<u32>,
    /// Scheduler priority.
    pub priority: i32,
    /// Effective diffusion controls and model-preprocessed media unit count.
    pub sampling: DiffusionSamplingParams,
}

impl DiffusionRequest {
    /// Checks that the prompt is nonempty, that the frame, media-unit, and
    /// step counts are positive, and that the prompt token count fits in `u32`.
    ///
    /// The engine's media admission rejects a failing request with
    /// `RejectionKind::Invalid`.
    pub fn validate(&self) -> Result<(), DiffusionRequestError> {
        if self.prompt_token_ids.is_empty() {
            return Err(DiffusionRequestError::EmptyPromptTokens);
        }

        if self.sampling.num_frames == 0
            || self.sampling.video_units == 0
            || self.sampling.num_inference_steps == 0
            || self.prompt_token_ids.len() > u32::MAX as usize
        {
            return Err(DiffusionRequestError::InvalidSampling);
        }

        Ok(())
    }
}

/// Validation failures for a media generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DiffusionRequestError {
    /// The tokenized prompt contains no tokens.
    #[error("media prompt tokens must not be empty")]
    EmptyPromptTokens,
    /// A frame, media-unit, or step count is zero, or the prompt token count
    /// exceeds `u32::MAX`.
    #[error("diffusion parameters are invalid")]
    InvalidSampling,
}

/// Immutable request payload selected before it enters the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum Request {
    /// Autoregressive text request.
    Ar(crate::GenerationRequest),
    /// Diffusion media request.
    Diffusion(DiffusionRequest),
    /// Unified multimodal request.
    Umm(crate::GenerationRequest),
    /// Block-diffusion token request, such as a canvas readout.
    BlockDiffusion(crate::GenerationRequest),
}

impl From<crate::GenerationRequest> for Request {
    /// Wraps an autoregressive generation request in the shared request enum.
    fn from(request: crate::GenerationRequest) -> Self {
        Self::Ar(request)
    }
}

impl Request {
    /// Returns the request identifier shared by every runtime family.
    pub const fn request_id(&self) -> RequestId {
        match self {
            Self::Ar(request) | Self::Umm(request) | Self::BlockDiffusion(request) => {
                request.request_id
            }
            Self::Diffusion(request) => request.request_id,
        }
    }

    /// Returns the scheduling priority shared by every runtime family.
    pub const fn priority(&self) -> i32 {
        match self {
            Self::Ar(request) | Self::Umm(request) | Self::BlockDiffusion(request) => {
                request.priority
            }
            Self::Diffusion(request) => request.priority,
        }
    }

    /// Returns the runtime family that owns this request.
    pub const fn family(&self) -> RuntimeFamily {
        match self {
            Self::Ar(_) => RuntimeFamily::Ar,
            Self::Diffusion(_) => RuntimeFamily::Diffusion,
            Self::Umm(_) => RuntimeFamily::Umm,
            Self::BlockDiffusion(_) => RuntimeFamily::BlockDiffusion,
        }
    }
}
