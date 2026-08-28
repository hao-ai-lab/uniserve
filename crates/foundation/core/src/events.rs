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

/// Fixed-profile media request. Media bytes stay in the shared spool at `output_path`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaRequest {
    pub request_id: RequestId,
    pub prompt: String,
    pub seed: u64,
    pub priority: i32,
    pub output_path: String,
}

impl MediaRequest {
    pub fn validate(&self) -> Result<(), MediaRequestError> {
        if self.prompt.trim().is_empty() {
            return Err(MediaRequestError::EmptyPrompt);
        }
        if self.output_path.is_empty() {
            return Err(MediaRequestError::EmptyOutputPath);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MediaRequestError {
    #[error("media prompt must not be empty")]
    EmptyPrompt,
    #[error("media output path must not be empty")]
    EmptyOutputPath,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaEvent {
    Completed { bytes: u64 },
    Rejected { message: String },
    Failed { message: String },
    Aborted,
}
