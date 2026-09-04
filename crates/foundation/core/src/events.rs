//! Requests and lifecycle events exchanged across the public engine boundary.

use serde::{Deserialize, Serialize};

use crate::RequestId;

/// Terminal cause for one engine generation lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The requested generation program completed normally.
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
    /// Zero-based probability rank at the position.
    pub rank: u32,
}

/// Ranked candidates for one scored token position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionLogprobs {
    /// Ranked token candidates for this position.
    pub entries: Vec<TokenLogprob>,
}

/// Typed event stream emitted by every engine runtime family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// Reports scheduler admission timing.
    Scheduled {
        /// Unix timestamp when the request entered the queue.
        queued_at: f64,
        /// Unix timestamp when the request was admitted.
        scheduled_at: f64,
    },
    /// Publishes one generated text token.
    TextToken {
        /// Vocabulary token identity.
        id: u32,
        /// Token log probability when requested.
        logprob: Option<f32>,
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
    /// Opens the lifecycle of one generated image.
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
    /// Reports one committed diffusion step.
    ImageStep {
        /// Request-local image identity.
        image_id: u32,
        /// One-based committed step number.
        step: u16,
    },
    /// Marks the transition from denoising to image materialization.
    ImageCommit {
        /// Request-local image identity.
        image_id: u32,
    },
    /// Publishes a materialized PNG image.
    ImageDone {
        /// Request-local image identity.
        image_id: u32,
        /// Decoded image height in pixels.
        height: u32,
        /// Decoded image width in pixels.
        width: u32,
        /// PNG payload size in bytes.
        bytes: u64,
        /// Lowercase hexadecimal SHA-256 digest of the PNG bytes.
        sha256: String,
        /// Base64-encoded PNG payload.
        pixels_png_b64: String,
    },
    /// Publishes a transport-backed media artifact.
    Artifact(ArtifactEvent),
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
        /// Human-readable rejection reason.
        message: String,
    },
    /// Terminates a request after an execution failure.
    Error {
        /// Human-readable failure reason.
        message: String,
    },
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

/// Runtime family selected once for an engine deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeFamily {
    /// Autoregressive text runtime.
    Ar,
    /// Diffusion-only media runtime.
    Diffusion,
    /// Unified multimodal understanding and generation runtime.
    Umm,
}

/// One caller-visible artifact backed by an explicit transport handle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactEvent {
    /// Semantic media type of the artifact.
    pub media_kind: MediaKind,
    /// MIME content type of the artifact payload.
    pub content_type: String,
    /// Materialized payload length in bytes.
    pub bytes: u64,
    /// Transport handle for reading the payload.
    pub artifact: ArtifactHandle,
}

/// Process-independent handle for a materialized artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "transport", content = "value")]
pub enum ArtifactHandle {
    /// Artifact stored in a POSIX shared-memory object.
    PosixShm {
        /// Shared-memory object name.
        name: String,
    },
}

impl ArtifactHandle {
    /// Returns the POSIX shared-memory object name carried by this handle.
    pub fn posix_shm_name(&self) -> &str {
        match self {
            Self::PosixShm { name } => name,
        }
    }
}

/// Immutable request-shaped media geometry resolved by the serving admission layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaGeometry {
    /// Number of output frames.
    pub frame_count: u32,
    /// Number of latent units decoded into output media.
    pub decode_units: u32,
    /// Number of prompt tokens represented by the request.
    pub prompt_tokens: u32,
    /// Number of diffusion denoising steps.
    pub denoise_steps: u32,
}

/// Media request. Final media bytes are returned through shared memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionRequest {
    /// Engine request identity.
    pub request_id: RequestId,
    /// Tokenized media prompt.
    pub prompt_token_ids: Vec<u32>,
    /// Deterministic diffusion seed.
    pub seed: u64,
    /// Scheduler priority.
    pub priority: i32,
    /// Fully resolved media geometry.
    pub geometry: MediaGeometry,
}

impl DiffusionRequest {
    /// Validates prompt presence and all positive geometry constraints.
    pub fn validate(&self) -> Result<(), DiffusionRequestError> {
        if self.prompt_token_ids.is_empty() {
            return Err(DiffusionRequestError::EmptyPromptTokens);
        }

        // Geometry must describe positive work and carry the same logical
        // prompt size as the token payload.
        if self.geometry.frame_count == 0
            || self.geometry.decode_units == 0
            || self.geometry.prompt_tokens == 0
            || usize::try_from(self.geometry.prompt_tokens).ok()
                != Some(self.prompt_token_ids.len())
            || self.geometry.denoise_steps == 0
        {
            return Err(DiffusionRequestError::InvalidGeometry);
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
    /// A geometry value is zero or disagrees with the prompt length.
    #[error("media geometry is invalid")]
    InvalidGeometry,
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
            Self::Ar(request) | Self::Umm(request) => request.request_id,
            Self::Diffusion(request) => request.request_id,
        }
    }

    /// Returns the scheduling priority shared by every runtime family.
    pub const fn priority(&self) -> i32 {
        match self {
            Self::Ar(request) | Self::Umm(request) => request.priority,
            Self::Diffusion(request) => request.priority,
        }
    }

    /// Returns the runtime family that owns this request.
    pub const fn family(&self) -> RuntimeFamily {
        match self {
            Self::Ar(_) => RuntimeFamily::Ar,
            Self::Diffusion(_) => RuntimeFamily::Diffusion,
            Self::Umm(_) => RuntimeFamily::Umm,
        }
    }
}
