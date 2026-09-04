//! Qwen3 reasoning parser configuration.

use crate::profile::tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, Result};

/// Reasoning parser for the configured Qwen3 description.
///
/// This parser uses standard `<think>...</think>` delimiters and defaults to
/// waiting for an explicit start token when prompt initialization finds no
/// reasoning boundary.
pub struct Qwen3ReasoningParser {
    inner: DelimitedReasoningParser,
}

impl Qwen3ReasoningParser {
    /// Creates a Qwen3 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false)?,
        })
    }
    /// Initializes delimiter state from the rendered prompt suffix.
    pub fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids);
        Ok(())
    }

    /// Returns whether decoding must preserve reasoning delimiters.
    pub const fn preserve_special_tokens(&self) -> bool {
        false
    }

    /// Applies one decoded text delta to the parser.
    pub fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        Ok(self.inner.push(delta))
    }

    /// Flushes buffered text at end of generation.
    pub fn finish(&mut self) -> Result<ReasoningDelta> {
        Ok(self.inner.finish())
    }
}
