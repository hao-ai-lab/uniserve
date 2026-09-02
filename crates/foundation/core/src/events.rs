use serde::{Deserialize, Serialize};

use crate::RequestId;

/// Terminal cause for one engine generation lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
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

/// Typed text and image event stream emitted by an engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GenerationEvent {
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
    MediaCompleted {
        bytes: u64,
    },
    MediaFailed {
        message: String,
    },
    MediaAborted,
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

/// Immutable request-shaped media geometry resolved by the serving admission layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaGeometry {
    pub frame_count: u32,
    pub video_reconstruction_units: u32,
    pub audio_latent_frames: u32,
    pub prompt_tokens: u32,
    pub denoise_steps: u32,
}

impl MediaGeometry {
    pub const fn required_audio_latent_frames(frame_count: u32) -> u64 {
        (frame_count as u64 * 40 + 23) / 24
    }
}

/// Media request. Final media bytes are returned through shared memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaRequest {
    pub request_id: RequestId,
    pub prompt_token_ids: Vec<u32>,
    pub seed: u64,
    pub priority: i32,
    pub geometry: MediaGeometry,
}

impl MediaRequest {
    pub fn validate(&self) -> Result<(), MediaRequestError> {
        if self.prompt_token_ids.is_empty() {
            return Err(MediaRequestError::EmptyPromptTokens);
        }
        if self.geometry.frame_count < 22
            || self.geometry.frame_count % 17 != 5
            || self.geometry.video_reconstruction_units != (self.geometry.frame_count - 5) / 17
            || u64::from(self.geometry.audio_latent_frames)
                != MediaGeometry::required_audio_latent_frames(self.geometry.frame_count)
            || self.geometry.prompt_tokens == 0
            || usize::try_from(self.geometry.prompt_tokens).ok()
                != Some(self.prompt_token_ids.len())
            || self.geometry.denoise_steps != 4
        {
            return Err(MediaRequestError::InvalidGeometry);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MediaRequestError {
    #[error("media prompt tokens must not be empty")]
    EmptyPromptTokens,
    #[error("media geometry is invalid")]
    InvalidGeometry,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaEvent {
    Completed { artifact: MediaArtifact },
    Rejected { message: String },
    Failed { message: String },
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaArtifact {
    pub handle: String,
    pub bytes: u64,
}
