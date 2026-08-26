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

/// Visible modality associated with one scheduler publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicModality {
    Text,
    Image,
}

/// Exact fixed state point that semantically owns a visible publication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticRoot {
    pub producer_op_id: u64,
    pub point_index: u32,
    pub semantic_digest: String,
}

/// Scheduler publication identity carried through decoding and protocol layers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublicCommit {
    pub event_seq: u64,
    pub modality: PublicModality,
    pub committed_at: f64,
    pub semantic_root: SemanticRoot,
}

impl PublicCommit {
    pub fn validate_for(&self, modality: PublicModality) -> Result<(), &'static str> {
        if self.event_seq == 0 {
            return Err("public event sequence must be positive");
        }
        if self.modality != modality {
            return Err("public event modality does not match its payload");
        }
        if !self.committed_at.is_finite() || self.committed_at < 0.0 {
            return Err("public commit timestamp must be finite and nonnegative");
        }
        if self.semantic_root.semantic_digest.len() != 64
            || !self
                .semantic_root
                .semantic_digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err("public semantic digest must be lowercase SHA-256 hex");
        }
        Ok(())
    }
}

/// Typed text and image event stream emitted by an engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GenEvent {
    Scheduled {
        queued_at: f64,
        scheduled_at: f64,
    },
    TextToken {
        id: u32,
        logprob: Option<f32>,
        public_commit: Option<PublicCommit>,
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
        public_commit: Option<PublicCommit>,
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
        stop_reason: Option<String>,
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
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.prompt.trim().is_empty() {
            return Err("media prompt must not be empty");
        }
        if self.output_path.is_empty() {
            return Err("media output path must not be empty");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaEvent {
    Completed { bytes: u64 },
    Rejected { message: String },
    Failed { message: String },
    Aborted,
}

impl MediaEvent {
    pub fn is_terminal(&self) -> bool {
        true
    }
}
