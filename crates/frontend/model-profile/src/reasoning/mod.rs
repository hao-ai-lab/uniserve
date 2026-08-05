//! Qwen3 streaming reasoning parsing.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod delimited;
mod qwen3;

use thiserror::Error;

pub use self::delimited::DelimitedReasoningParser;
pub use self::qwen3::Qwen3ReasoningParser;

/// Result alias for reasoning parser operations.
pub type Result<T> = std::result::Result<T, ReasoningError>;

/// One parsed streaming delta split into reasoning and visible content.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReasoningDelta {
    pub reasoning: Option<String>,
    pub content: Option<String>,
}

impl ReasoningDelta {
    /// Return true when this delta carries neither reasoning nor content text.
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_none() && self.content.is_none()
    }

    /// Append text to the reasoning portion, creating it on first use.
    pub(crate) fn push_reasoning(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match &mut self.reasoning {
            Some(existing) => existing.push_str(text),
            None => self.reasoning = Some(text.to_string()),
        }
    }

    /// Append text to the visible content portion, creating it on first use.
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
    #[error("tokenizer is missing reasoning delimiter token `{token}`")]
    MissingToken { token: String },
    #[error("reasoning delimiter {field} must not be empty")]
    EmptyDelimiter { field: &'static str },
}

#[cfg(test)]
mod tests;
