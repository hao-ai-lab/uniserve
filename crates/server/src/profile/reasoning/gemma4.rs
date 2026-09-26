//! Gemma-4 reasoning parser configuration.

use crate::profile::tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Special token that opens a Gemma-4 channel.
const CHANNEL_START: &str = "<|channel>";

/// Name line of the thought channel, written right after [`CHANNEL_START`].
const THOUGHT_CHANNEL_LABEL: &str = "thought\n";

/// Special token that closes a Gemma-4 channel.
const CHANNEL_END: &str = "<channel|>";

/// Reasoning parser for Gemma-4 chat output.
///
/// Gemma-4 writes reasoning in a thought channel: the `<|channel>` token, the
/// channel name `thought` and a newline, the reasoning text, then the
/// `<channel|>` token, for example `<|channel>thought\nreasoning<channel|>`.
/// Both tokens are tokenizer special tokens, so the parser only sees them when
/// generated text is decoded with special tokens kept.
///
/// Generation starts in visible content unless the prompt ends inside an open
/// thought channel. A new model turn (`<|turn>model\n`) starts in content;
/// with thinking enabled the model then opens the channel itself, and with
/// thinking disabled it may still emit an empty channel
/// (`<|channel>thought\n<channel|>`), which yields no reasoning text. A prompt
/// that ends with `<|channel>thought\n`, as the Gemma-4 template renders a
/// thinking continuation after tool responses, starts inside reasoning.
pub struct Gemma4ReasoningParser {
    inner: DelimitedReasoningParser,
}

impl Gemma4ReasoningParser {
    /// Creates a Gemma-4 parser backed by the shared delimited state machine.
    ///
    /// # Errors
    ///
    /// Fails when the tokenizer lacks `<|channel>` or `<channel|>` as single
    /// tokens.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        let inner = DelimitedReasoningParser::new(tokenizer, CHANNEL_START, CHANNEL_END, false)?
            .with_start_label(THOUGHT_CHANNEL_LABEL);
        Ok(Self { inner })
    }
}

impl ReasoningParser for Gemma4ReasoningParser {
    /// Starts in reasoning when the prompt's last channel token after its last
    /// other special token is an open `<|channel>`, and in content otherwise.
    fn initialize(&mut self, prompt_token_ids: &[u32]) {
        self.inner.initialize(prompt_token_ids);
    }

    /// Applies one decoded text delta, which may span whole channels.
    fn push(&mut self, delta: &str) -> ReasoningDelta {
        self.inner.push(delta)
    }

    /// Flushes buffered text at end of generation.
    fn finish(&mut self) -> ReasoningDelta {
        self.inner.finish()
    }
}
