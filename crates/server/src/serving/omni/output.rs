//! Multimodal output filters selected by model profiles.
//!
//! `serving::assembly` builds a [`SenseNovaOutputProcessor`] when a SenseNova
//! profile returns `OutputProcessorPolicy::SenseNova`, then pushes each decoded
//! text fragment through it. The processor splits text into reasoning and
//! visible output with a `DelimitedReasoningParser`, then strips the profile's
//! visible-wrapper delimiters from the visible part. Both stages match
//! delimiters as text and hold back a trailing fragment that may begin a
//! delimiter split across pushes; the assembler calls
//! [`SenseNovaOutputProcessor::finish`] at the end of the stream to release
//! it.

use crate::profile::omni::{DelimitedTextPolicy, OutputFilterPolicy};
use crate::profile::reasoning::DelimitedReasoningParser;
use crate::serving::text::tokenizer::DynTokenizer;

/// Stateful SenseNova reasoning and visible-wrapper filter for one request.
pub(crate) struct SenseNovaOutputProcessor {
    reasoning: Option<DelimitedReasoningParser>,
    visible_wrappers: VisibleWrapperFilter,
}

#[derive(Debug, Default, PartialEq, Eq)]
/// Visible and reasoning text emitted by one filter update.
pub(crate) struct SenseNovaTextDelta {
    pub(crate) visible: String,
    pub(crate) reasoning: String,
}

impl SenseNovaOutputProcessor {
    /// Creates the output filter selected by a SenseNova profile.
    ///
    /// The prompt token IDs decide whether generation starts inside the
    /// reasoning section (see `DelimitedReasoningParser::initialize`).
    ///
    /// # Errors
    ///
    /// Fails when a reasoning delimiter is empty or is not a single token in
    /// the tokenizer vocabulary.
    pub(crate) fn new(
        policy: OutputFilterPolicy,
        tokenizer: DynTokenizer,
        prompt_token_ids: &[u32],
    ) -> crate::profile::reasoning::Result<Self> {
        let reasoning = if let Some(reasoning) = policy.reasoning.clone() {
            let mut parser =
                DelimitedReasoningParser::new(tokenizer, reasoning.start, reasoning.end, false)?;
            parser.initialize(prompt_token_ids);
            Some(parser)
        } else {
            None
        };
        Ok(Self {
            reasoning,
            visible_wrappers: VisibleWrapperFilter::new(policy.visible_wrappers),
        })
    }

    /// Applies one decoded text fragment and returns semantic deltas.
    ///
    /// Either part of the delta may be empty while a possible delimiter prefix
    /// is held back until a later push or `finish` releases it.
    pub(crate) fn push(&mut self, text: &str) -> SenseNovaTextDelta {
        let (content, reasoning) = if let Some(parser) = self.reasoning.as_mut() {
            let delta = parser.push(text);
            (
                delta.content.unwrap_or_default(),
                delta.reasoning.unwrap_or_default(),
            )
        } else {
            (text.to_string(), String::new())
        };
        SenseNovaTextDelta {
            visible: self.visible_wrappers.push(&content),
            reasoning,
        }
    }

    /// Releases every held-back fragment at the end of the stream.
    ///
    /// Call once, after the final `push`. No later text can complete a
    /// delimiter, so a held fragment is ordinary text of its current region:
    /// the reasoning parser's buffered partial delimiter becomes reasoning or
    /// visible text (`DelimitedReasoningParser::finish`), visible text still
    /// passes the wrapper filter, and the filter's held partial wrapper
    /// delimiter is emitted as visible text.
    pub(crate) fn finish(&mut self) -> SenseNovaTextDelta {
        let (content, reasoning) = if let Some(parser) = self.reasoning.as_mut() {
            let delta = parser.finish();
            (
                delta.content.unwrap_or_default(),
                delta.reasoning.unwrap_or_default(),
            )
        } else {
            (String::new(), String::new())
        };
        let mut visible = self.visible_wrappers.push(&content);
        visible.push_str(&self.visible_wrappers.finish());
        SenseNovaTextDelta { visible, reasoning }
    }
}

/// Removes wrapper delimiters from visible text while keeping the wrapped text.
struct VisibleWrapperFilter {
    wrappers: Vec<DelimitedTextPolicy>,
    /// Text not yet emitted: at most a trailing fragment that is a proper
    /// prefix of some delimiter.
    pending: String,
}

impl VisibleWrapperFilter {
    /// Creates a filter for the configured wrapper delimiters.
    fn new(wrappers: Vec<DelimitedTextPolicy>) -> Self {
        Self {
            wrappers,
            pending: String::new(),
        }
    }

    /// Removes configured wrapper delimiters while retaining incomplete marker prefixes.
    ///
    /// Start and end delimiters are removed independently, so an unmatched
    /// delimiter is also dropped.
    fn push(&mut self, text: &str) -> String {
        if self.wrappers.is_empty() {
            return text.to_string();
        }
        self.pending.push_str(text);
        let mut visible = String::new();

        // Strip complete markers from the front, then emit everything except a
        // trailing fragment that could still grow into a marker.
        loop {
            if let Some((start, marker_len)) = self.first_marker() {
                visible.push_str(&self.pending[..start]);
                self.pending.drain(..start + marker_len);
                continue;
            }

            let keep = trailing_marker_prefix_len_any(&self.pending, &self.markers());
            let emit_to = self.pending.len().saturating_sub(keep);
            visible.push_str(&self.pending[..emit_to]);
            self.pending.drain(..emit_to);
            break;
        }

        visible
    }

    /// Returns the held trailing fragment at the end of the stream.
    ///
    /// The fragment is a proper prefix of a delimiter that can no longer
    /// complete, so it is visible text.
    fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Returns all configured output markers.
    fn markers(&self) -> Vec<&str> {
        let mut markers = Vec::new();
        for wrapper in &self.wrappers {
            markers.push(wrapper.start.as_str());
            markers.push(wrapper.end.as_str());
        }
        markers
    }

    /// Returns the byte offset and length of the earliest complete marker in
    /// `pending`.
    fn first_marker(&self) -> Option<(usize, usize)> {
        let mut matches = Vec::new();
        for wrapper in &self.wrappers {
            matches.push(wrapper.start.as_str());
            matches.push(wrapper.end.as_str());
        }
        matches
            .into_iter()
            .filter_map(|marker| self.pending.find(marker).map(|idx| (idx, marker.len())))
            .min_by_key(|(idx, _)| *idx)
    }
}

/// Returns the longest suffix matching any marker prefix.
fn trailing_marker_prefix_len_any(text: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .map(|marker| trailing_marker_prefix_len(text, marker))
        .max()
        .unwrap_or(0)
}

/// Returns the length of the longest suffix of `text` that is a proper prefix of `marker`.
///
/// A full marker is not counted; `VisibleWrapperFilter::push` strips complete
/// markers before calling this. Lengths are in bytes and only split at UTF-8
/// character boundaries.
fn trailing_marker_prefix_len(text: &str, marker: &str) -> usize {
    let max = text.len().min(marker.len().saturating_sub(1));
    for len in (1..=max).rev() {
        if text.is_char_boundary(text.len() - len)
            && marker.is_char_boundary(len)
            && text[text.len() - len..].eq(&marker[..len])
        {
            return len;
        }
    }
    0
}
