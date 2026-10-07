//! Qwen3 reasoning parser configuration.

use crate::profile::tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

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
    ///
    /// Fails when the tokenizer lacks `<think>` or `</think>` as single
    /// tokens.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false)?,
        })
    }
}

impl ReasoningParser for Qwen3ReasoningParser {
    /// Initializes delimiter state from the rendered prompt suffix.
    fn initialize(&mut self, prompt_token_ids: &[u32]) {
        self.inner.initialize(prompt_token_ids);
    }

    /// Applies one decoded text delta to the parser.
    fn push(&mut self, delta: &str) -> ReasoningDelta {
        self.inner.push(delta)
    }

    /// Flushes buffered text at end of generation.
    fn finish(&mut self) -> ReasoningDelta {
        self.inner.finish()
    }
}
