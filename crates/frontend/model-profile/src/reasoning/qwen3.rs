use crate::tokenizer::DynTokenizer;

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
    /// Create a Qwen3 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false)?,
        })
    }
    pub fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids);
        Ok(())
    }

    pub const fn preserve_special_tokens(&self) -> bool {
        false
    }

    pub fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        Ok(self.inner.push(delta))
    }

    pub fn finish(&mut self) -> Result<ReasoningDelta> {
        Ok(self.inner.finish())
    }
}
