//! Streaming reasoning parsers and semantic deltas.
//!
//! A parser splits a stream of decoded text deltas into reasoning text and
//! visible content by matching delimiter text such as `<think>` and
//! `</think>`. [`DelimitedReasoningParser`] implements the state machine;
//! [`Qwen3ReasoningParser`] and [`Gemma4ReasoningParser`] configure it for
//! their chat output formats behind the [`ReasoningParser`] interface that the
//! chat output stage consumes, and the multimodal output filter in
//! `serving::omni::output` configures it from a profile's
//! `OutputFilterPolicy`.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod delimited;
mod gemma4;
mod qwen3;

use thiserror::Error;

pub use self::delimited::DelimitedReasoningParser;
pub use self::gemma4::Gemma4ReasoningParser;
pub use self::qwen3::Qwen3ReasoningParser;

/// Incremental split of one generation's decoded text into reasoning and
/// visible content.
///
/// One instance serves one generation stream: call `initialize` with the
/// prompt, `push` for each decoded delta, then `finish` once at end of stream.
pub trait ReasoningParser: Send {
    /// Chooses the region generation starts in from the prompt token IDs.
    fn initialize(&mut self, prompt_token_ids: &[u32]);

    /// Parses one decoded text delta into its reasoning and content parts.
    ///
    /// Text that could begin a delimiter stays buffered until a later delta
    /// completes or rules it out, so the returned delta may be empty.
    fn push(&mut self, delta: &str) -> ReasoningDelta;

    /// Flushes buffered text at end of stream as text of the current region.
    fn finish(&mut self) -> ReasoningDelta;
}

/// Result alias for reasoning parser calls.
pub type Result<T> = std::result::Result<T, ReasoningError>;

/// One parsed streaming delta split into reasoning and visible content.
///
/// The push methods ignore empty text, so a field they populate always holds
/// non-empty text.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReasoningDelta {
    /// Incremental hidden reasoning text.
    pub reasoning: Option<String>,
    /// Incremental user-visible assistant text.
    pub content: Option<String>,
}

impl ReasoningDelta {
    /// Returns `true` when this delta carries neither reasoning nor content text.
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_none() && self.content.is_none()
    }

    /// Appends text to the reasoning portion, creating it on first use.
    pub(crate) fn push_reasoning(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match &mut self.reasoning {
            Some(existing) => existing.push_str(text),
            None => self.reasoning = Some(text.to_string()),
        }
    }

    /// Appends text to the visible content portion, creating it on first use.
    pub(crate) fn push_content(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match &mut self.content {
            Some(existing) => existing.push_str(text),
            None => self.content = Some(text.to_string()),
        }
    }
}

/// Errors produced while creating or running reasoning parsers.
#[derive(Debug, Error)]
pub enum ReasoningError {
    /// A required delimiter token is absent from the tokenizer vocabulary.
    #[error("tokenizer is missing reasoning delimiter token `{token}`")]
    MissingToken {
        /// Missing delimiter token text.
        token: String,
    },
    /// A configured delimiter has no text.
    #[error("reasoning delimiter {field} must not be empty")]
    EmptyDelimiter {
        /// Configuration field that contains the empty delimiter.
        field: &'static str,
    },
}

#[cfg(test)]
mod tests;
