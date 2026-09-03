use serde::{Deserialize, Serialize};

use crate::RequestId;

/// Terminal cause for one engine generation lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Completed,
    Eos,
    MaxTokens,
    Stop,
    ImageDone,
    Cancelled,
    Aborted,
    Repetition,
    Error,
}

/// Typed payload identifying the exact stop condition that ended generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum StopReason {
    Token(u32),
    String(String),
}

/// One ranked vocabulary candidate at a generated or prompt token position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token_id: u32,
    pub logprob: f32,
    pub rank: u32,
}

/// Ranked candidates for one scored token position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionLogprobs {
    pub entries: Vec<TokenLogprob>,
}

/// Typed event stream emitted by every engine runtime family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Scheduled {
        queued_at: f64,
        scheduled_at: f64,
    },
    TextToken {
        id: u32,
        logprob: Option<f32>,
    },
    TokenLogprobs {
        id: u32,
        candidates: Vec<TokenLogprob>,
    },
    PromptLogprobs {
        positions: Vec<PositionLogprobs>,
    },
    ImageBegin {
        image_id: u32,
        height: u32,
        width: u32,
        steps: u16,
    },
    ImageStep {
        image_id: u32,
        step: u16,
    },
    ImageCommit {
        image_id: u32,
    },
    ImageDone {
        image_id: u32,
        height: u32,
        width: u32,
        bytes: u64,
        sha256: String,
        pixels_png_b64: String,
    },
    Artifact(ArtifactEvent),
    Finished {
        reason: FinishReason,
        stop_reason: Option<StopReason>,
        prompt_tokens: usize,
        completion_tokens: usize,
        images: usize,
    },
    Rejected {
        message: String,
    },
    Error {
        message: String,
    },
}

/// Kind of media referenced by an artifact event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    Image,
    Video,
    Audio,
}

/// Runtime family selected once for an engine deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeFamily {
    Ar,
    Diffusion,
    Umm,
}

/// One caller-visible artifact backed by an explicit transport handle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactEvent {
    pub media_kind: MediaKind,
    pub content_type: String,
    pub bytes: u64,
    pub artifact: ArtifactHandle,
}

/// Process-independent handle for a materialized artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "transport", content = "value")]
pub enum ArtifactHandle {
    PosixShm { name: String },
}

impl ArtifactHandle {
    pub fn posix_shm_name(&self) -> &str {
        match self {
            Self::PosixShm { name } => name,
        }
    }
}

/// Immutable request-shaped media geometry resolved by the serving admission layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaGeometry {
    pub frame_count: u32,
    pub decode_units: u32,
    pub prompt_tokens: u32,
    pub denoise_steps: u32,
}

/// Media request. Final media bytes are returned through shared memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionRequest {
    pub request_id: RequestId,
    pub prompt_token_ids: Vec<u32>,
    pub seed: u64,
    pub priority: i32,
    pub geometry: MediaGeometry,
}

impl DiffusionRequest {
    pub fn validate(&self) -> Result<(), DiffusionRequestError> {
        if self.prompt_token_ids.is_empty() {
            return Err(DiffusionRequestError::EmptyPromptTokens);
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DiffusionRequestError {
    #[error("media prompt tokens must not be empty")]
    EmptyPromptTokens,
    #[error("media geometry is invalid")]
    InvalidGeometry,
}

/// Immutable request payload selected before it enters the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum Request {
    Ar(crate::GenerationRequest),
    Diffusion(DiffusionRequest),
    Umm(crate::GenerationRequest),
}

impl From<crate::GenerationRequest> for Request {
    fn from(request: crate::GenerationRequest) -> Self {
        Self::Ar(request)
    }
}

impl Request {
    pub const fn request_id(&self) -> RequestId {
        match self {
            Self::Ar(request) | Self::Umm(request) => request.request_id,
            Self::Diffusion(request) => request.request_id,
        }
    }

    pub const fn priority(&self) -> i32 {
        match self {
            Self::Ar(request) | Self::Umm(request) => request.priority,
            Self::Diffusion(request) => request.priority,
        }
    }

    pub const fn family(&self) -> RuntimeFamily {
        match self {
            Self::Ar(_) => RuntimeFamily::Ar,
            Self::Diffusion(_) => RuntimeFamily::Diffusion,
            Self::Umm(_) => RuntimeFamily::Umm,
        }
    }
}
