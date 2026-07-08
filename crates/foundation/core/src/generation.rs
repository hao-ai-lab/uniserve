//! Canonical generation request descriptors.
//!
//! These types describe the lowered generation paradigm before the scheduler
//! turns it into worker ops. They are data-only: no channels, worker handles,
//! scheduler state, or model-local logic belongs here.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::Modality;

/// Output constraint applied to the default generation paradigm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationConstraint {
    /// Both Und and Gen branches are enabled when the dialect supports them.
    #[default]
    Default,
    /// Only the Und branch may produce user-visible output.
    UndOnly,
    /// Only the Gen branch may produce user-visible output. Und tokens may still
    /// be generated internally when the dialect requires them for control.
    GenOnly,
}

impl GenerationConstraint {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::UndOnly => "und_only",
            Self::GenOnly => "gen_only",
        }
    }
}

impl FromStr for GenerationConstraint {
    type Err = GenerationConstraintParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "default" => Ok(Self::Default),
            "und_only" => Ok(Self::UndOnly),
            "gen_only" => Ok(Self::GenOnly),
            other => Err(GenerationConstraintParseError {
                value: other.to_string(),
            }),
        }
    }
}

/// A rejected [`GenerationConstraint`] string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported generation constraint: {value}")]
pub struct GenerationConstraintParseError {
    pub value: String,
}

/// Whether an Und token segment is user-visible or internal control/context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndVisibility {
    #[default]
    Visible,
    Internal,
}

/// One context segment in the lowered request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ContextSegment {
    UndTokens {
        token_ids: Vec<u32>,
        #[serde(default)]
        visibility: UndVisibility,
    },
    Image {
        image: ImageSegment,
        ingest: ImageIngestRecipe,
    },
}

/// Input image bytes plus placement in the rendered context stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSegment {
    pub hash: u64,
    pub b64: String,
    pub placement: SegmentPlacement,
}

/// Logical placement of an image segment in the already-rendered Und stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SegmentPlacement {
    /// The image encoder output fills the gap ending at this token index.
    AtToken { position: u32 },
    /// The image is appended after all Und tokens emitted by the context.
    Append,
}

/// Dialect-lowered recipe for turning an image segment into context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageIngestRecipe {
    pub steps: Vec<ImageIngestStep>,
    pub logical_positions: u32,
    pub physical_kv_tokens: ImageKvEffect,
    pub modality: Modality,
}

impl ImageIngestRecipe {
    pub fn vit_only(logical_positions: u32, physical_kv_tokens: ImageKvEffect) -> Self {
        Self {
            steps: vec![ImageIngestStep::VitEncode],
            logical_positions,
            physical_kv_tokens,
            modality: Modality::Und,
        }
    }

    pub fn vae_then_vit(logical_positions: u32, physical_kv_tokens: ImageKvEffect) -> Self {
        Self {
            steps: vec![ImageIngestStep::VaeEncode, ImageIngestStep::VitEncode],
            logical_positions,
            physical_kv_tokens,
            modality: Modality::Und,
        }
    }
}

/// One image ingest worker step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageIngestStep {
    VaeEncode,
    VitEncode,
}

/// Physical KV effect of an image ingest operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ImageKvEffect {
    WorkerDefined,
    Exact { tokens: u32 },
    Bounded { max_tokens: u32 },
}

/// Generated-image feedback recipe supplied by the dialect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratedImageFeedbackRecipe {
    pub commit: CommitRecipe,
    pub writeback: FeedbackWriteback,
    pub next_und_token: FeedbackNextToken,
}

/// Worker op sequence used to commit a generated image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommitRecipe {
    CommitGen,
    CommitGenThenWriteback,
}

/// How a committed Gen image becomes continued context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FeedbackWriteback {
    Disabled,
    DirectKv,
    Reingest { ingest: Box<ImageIngestRecipe> },
}

/// Und token used to continue after generated-image feedback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FeedbackNextToken {
    None,
    Bos,
    EndOfImage,
    Token { token_id: u32 },
}

/// Dialect token-trigger matching lowered to scheduler-readable data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum TriggerPolicyDescriptor {
    Disabled,
    Token {
        token_id: u32,
    },
    Suffix {
        token_ids: Vec<u32>,
    },
    RoundCloseThenSuffix {
        close_token_ids: Vec<u32>,
        trigger_token_ids: Vec<u32>,
    },
}

/// Scheduler action for generated Und tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndTokenAction {
    Emit,
    KeepInternal,
    Reject,
}

/// Visibility rules per output constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisibilityPolicyDescriptor {
    pub default: UndTokenAction,
    pub und_only: UndTokenAction,
    pub gen_only: UndTokenAction,
}

impl Default for VisibilityPolicyDescriptor {
    fn default() -> Self {
        Self {
            default: UndTokenAction::Emit,
            und_only: UndTokenAction::Emit,
            gen_only: UndTokenAction::KeepInternal,
        }
    }
}

/// Termination rules express whether branch completion finishes or continues.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminationPolicyDescriptor {
    pub eos_finishes: bool,
    pub stop_finishes: bool,
    pub max_tokens_finishes: bool,
    pub gen_commit_finishes_gen_only: bool,
    pub gen_commit_continues_default: bool,
}

impl Default for TerminationPolicyDescriptor {
    fn default() -> Self {
        Self {
            eos_finishes: true,
            stop_finishes: true,
            max_tokens_finishes: true,
            gen_commit_finishes_gen_only: true,
            gen_commit_continues_default: true,
        }
    }
}

/// Dialect-lowered generation policy consumed by the scheduler planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationPolicyDescriptor {
    pub trigger: TriggerPolicyDescriptor,
    pub visibility: VisibilityPolicyDescriptor,
    pub termination: TerminationPolicyDescriptor,
    pub feedback: Option<GeneratedImageFeedbackRecipe>,
}

impl Default for GenerationPolicyDescriptor {
    fn default() -> Self {
        Self {
            trigger: TriggerPolicyDescriptor::Disabled,
            visibility: VisibilityPolicyDescriptor::default(),
            termination: TerminationPolicyDescriptor::default(),
            feedback: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_constraint_strings_are_canonical() {
        for (name, constraint) in [
            ("default", GenerationConstraint::Default),
            ("und_only", GenerationConstraint::UndOnly),
            ("gen_only", GenerationConstraint::GenOnly),
        ] {
            assert_eq!(constraint.as_str(), name);
            assert_eq!(name.parse::<GenerationConstraint>().unwrap(), constraint);
        }
        assert!("text".parse::<GenerationConstraint>().is_err());
    }

    #[test]
    fn default_visibility_hides_gen_only_und_tokens() {
        let policy = VisibilityPolicyDescriptor::default();
        assert_eq!(policy.default, UndTokenAction::Emit);
        assert_eq!(policy.und_only, UndTokenAction::Emit);
        assert_eq!(policy.gen_only, UndTokenAction::KeepInternal);
    }
}
