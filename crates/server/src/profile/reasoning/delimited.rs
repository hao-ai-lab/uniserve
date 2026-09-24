//! Incremental parser for reasoning enclosed by configured text delimiters.
//!
//! Delimiters are matched as text in the decoded stream, so text that may be
//! the start of a delimiter split across deltas is held back until later text
//! completes or rules it out, or the stream ends. Each delimiter must also be
//! a single vocabulary token, whose id `initialize` looks for in the prompt to
//! choose the initial region.

use crate::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};

use super::{ReasoningDelta, ReasoningError, Result};

/// Shared incremental state machine for configured tag-delimited reasoning.
///
/// One instance serves one generation stream: optionally call
/// [`DelimitedReasoningParser::initialize`] with the prompt, then
/// [`DelimitedReasoningParser::push`] for each decoded delta, then
/// [`DelimitedReasoningParser::finish`] once at end of stream.
pub struct DelimitedReasoningParser {
    tokenizer: DynTokenizer,
    current_in_reasoning: bool,
    /// Trailing text that may be the start of a delimiter, held until a later
    /// delta disambiguates it or `finish` flushes it.
    buffer: String,
    start_token: String,
    end_token: String,
    start_token_id: u32,
    end_token_id: u32,
    default_in_reasoning: bool,
}

impl DelimitedReasoningParser {
    /// Creates one delimited parser state machine.
    ///
    /// `default_in_reasoning` is the initial region when `initialize` finds no
    /// delimiter token after the prompt's last other special token, and also
    /// when `initialize` is never called.
    ///
    /// # Errors
    ///
    /// Fails when either delimiter is empty or is not a single token in the
    /// tokenizer vocabulary.
    pub fn new(
        tokenizer: DynTokenizer,
        start_token: impl Into<String>,
        end_token: impl Into<String>,
        default_in_reasoning: bool,
    ) -> Result<Self> {
        let start_token = start_token.into();
        let end_token = end_token.into();
        if start_token.is_empty() {
            return Err(ReasoningError::EmptyDelimiter { field: "start" });
        }
        if end_token.is_empty() {
            return Err(ReasoningError::EmptyDelimiter { field: "end" });
        }
        let start_token_id =
            tokenizer
                .token_to_id(&start_token)
                .ok_or_else(|| ReasoningError::MissingToken {
                    token: start_token.clone(),
                })?;
        let end_token_id =
            tokenizer
                .token_to_id(&end_token)
                .ok_or_else(|| ReasoningError::MissingToken {
                    token: end_token.clone(),
                })?;

        Ok(Self {
            tokenizer,
            current_in_reasoning: default_in_reasoning,
            buffer: String::new(),
            start_token,
            end_token,
            start_token_id,
            end_token_id,
            default_in_reasoning,
        })
    }

    /// Initializes parser state from the prompt token IDs.
    ///
    /// The last delimiter token in the prompt decides where generation starts
    /// when no other special token follows it: with `<think>` delimiters, a
    /// prompt ending in `<think>` starts inside reasoning and one ending in
    /// `</think>` starts in visible content. Otherwise the parser starts in the
    /// `default_in_reasoning` region.
    pub fn initialize(&mut self, prompt_token_ids: &[u32]) {
        self.current_in_reasoning = last_reasoning_boundary(
            prompt_token_ids,
            self.start_token_id,
            self.end_token_id,
            self.tokenizer.as_ref(),
        )
        .unwrap_or(self.default_in_reasoning);
    }

    /// Parses one decoded text delta and returns its reasoning/content split.
    ///
    /// Text that could begin a delimiter stays buffered, so the returned
    /// delta may be empty.
    pub fn push(&mut self, delta: &str) -> ReasoningDelta {
        self.buffer.push_str(delta);

        // Keep only the possible partial delimiter in the buffer and parse
        // everything before it.
        let partial_suffix_len = self.partial_suffix_len(&self.buffer);
        let stable_len = self.buffer.len() - partial_suffix_len;
        let pending_suffix = self.buffer.split_off(stable_len);
        let stable_text = std::mem::replace(&mut self.buffer, pending_suffix);

        self.parse_stable_text(&stable_text)
    }

    /// Flushes any buffered partial delimiter suffix at end of stream.
    ///
    /// The unfinished delimiter text is emitted as ordinary text of the
    /// current region.
    pub fn finish(&mut self) -> ReasoningDelta {
        let stable_text = std::mem::take(&mut self.buffer);
        self.parse_stable_text(&stable_text)
    }

    /// Parses text that is known not to end with a partial delimiter suffix.
    fn parse_stable_text(&mut self, mut stable: &str) -> ReasoningDelta {
        let mut delta = ReasoningDelta::default();

        while !stable.is_empty() {
            if self.current_in_reasoning {
                if let Some(end_idx) = stable.find(&self.end_token) {
                    delta.push_reasoning(&stable[..end_idx]);
                    stable = &stable[end_idx + self.end_token.len()..];
                    self.current_in_reasoning = false;
                } else {
                    delta.push_reasoning(stable);
                    break;
                }
            } else if let Some(start_idx) = stable.find(&self.start_token) {
                delta.push_content(&stable[..start_idx]);
                stable = &stable[start_idx + self.start_token.len()..];
                self.current_in_reasoning = true;
            } else {
                delta.push_content(stable);
                break;
            }
        }

        delta
    }

    /// Returns the longest trailing suffix that could still complete a
    /// delimiter.
    ///
    /// A trailing suffix can only be a *strict* prefix of a delimiter when it
    /// is shorter than that delimiter, so only the final
    /// `max(start_token, end_token)` bytes of `text` can ever match. Scanning
    /// just that trailing window keeps the cost bounded by the delimiter
    /// length rather than the length of `text`.
    fn partial_suffix_len(&self, text: &str) -> usize {
        let max_token_len = self.start_token.len().max(self.end_token.len());
        // Suffixes at least `max_token_len` bytes long cannot be a strict
        // prefix of either delimiter, so start scanning from there. Clamp to a
        // char boundary so the slice below is always valid.
        let mut window_start = text.len().saturating_sub(max_token_len);
        while window_start < text.len() && !text.is_char_boundary(window_start) {
            window_start += 1;
        }

        let mut best = 0;
        for (idx, _) in text[window_start..].char_indices() {
            let idx = window_start + idx;
            let suffix = &text[idx..];
            if self.start_token.starts_with(suffix) && self.start_token != suffix {
                best = best.max(text.len() - idx);
            }
            if self.end_token.starts_with(suffix) && self.end_token != suffix {
                best = best.max(text.len() - idx);
            }
        }

        best
    }
}

/// Determines the reasoning state implied by the last prompt boundary, if any.
///
/// Scans backwards from the end of the prompt. The first delimiter found
/// decides: the start delimiter means inside reasoning, the end delimiter
/// means outside. Reaching any other special token first, or the start of the
/// prompt, yields `None`.
fn last_reasoning_boundary(
    prompt_token_ids: &[u32],
    start_token_id: u32,
    end_token_id: u32,
    tokenizer: &HuggingFaceTokenizer,
) -> Option<bool> {
    for token_id in prompt_token_ids.iter().rev() {
        if *token_id == start_token_id {
            return Some(true);
        }
        if *token_id == end_token_id {
            return Some(false);
        }
        if tokenizer.is_special_id(*token_id) {
            return None;
        }
    }

    None
}
