//! Streaming tool-call parsers and normalized tool descriptors.
//!
//! The Qwen3 chat output stage (`serving::chat::output::qwen3`) feeds visible
//! assistant text to `Qwen3XmlToolParser` chunk by chunk. The parser turns
//! each chunk into an ordered sequence of plain text and `ToolCallDelta`
//! updates, holding back bytes it cannot classify yet, such as a partial
//! marker or an incomplete tool-call header.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#[macro_use]
mod error;
mod json;
#[cfg(any(test, feature = "test-util"))]
pub mod test_utils;
mod utils;

pub use error::{Result, ToolParserError};
pub use json::Qwen3XmlToolParser;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One function-style tool made available to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
    /// Function name presented to the model.
    pub name: String,
    /// Optional natural-language function description.
    pub description: Option<String>,
    /// JSON Schema describing accepted function arguments.
    pub parameters: Value,
    /// Optional strict-schema enforcement preference.
    pub strict: Option<bool>,
}

/// One tool-call update emitted while parsing assistant text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallDelta {
    /// Stable parser-local tool index for this call within one assistant turn.
    pub tool_index: usize,
    /// Function name, present on the first update for one tool call.
    pub name: Option<String>,
    /// Arguments text contributed by this update. Concatenating every update
    /// for one `tool_index` yields the arguments JSON exactly as the model
    /// wrote it; the text is neither parsed as JSON nor normalized.
    pub arguments: String,
}

/// One piece of parser output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolParserItem {
    /// Plain assistant text that is not part of any tool call.
    Text(String),
    /// One tool-call update.
    Call(ToolCallDelta),
}

/// Result of advancing tool parsing with one or more assistant-text inputs.
///
/// Items follow the order of the input they were parsed from, so text written
/// before a tool call precedes that call's deltas and text written after it
/// follows them. Adjacent text is merged into one nonempty item. A call's
/// deltas are contiguous, because text resumes only after its end delimiter.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolParserOutput {
    /// Text runs and tool-call updates in input order.
    pub items: Vec<ToolParserItem>,
}

impl ToolParserOutput {
    /// Appends plain text, merging it into a trailing text item.
    ///
    /// Empty text is ignored, so every text item is nonempty.
    fn push_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }

        match self.items.last_mut() {
            Some(ToolParserItem::Text(last)) => last.push_str(text),
            _ => self.items.push(ToolParserItem::Text(text.to_string())),
        }
    }

    /// Appends one tool-call update after the existing items.
    fn push_call(&mut self, call: ToolCallDelta) {
        self.items.push(ToolParserItem::Call(call));
    }

    /// Appends another parser output onto this one, keeping item order.
    ///
    /// Text on both sides of the boundary is merged. Deltas for the same tool
    /// call are not; call `coalesce_calls` afterwards to merge them.
    pub fn append(&mut self, other: Self) {
        for item in other.items {
            match item {
                ToolParserItem::Text(text) => self.push_text(&text),
                ToolParserItem::Call(call) => self.push_call(call),
            }
        }
    }

    /// Returns the concatenation of every text item.
    pub fn normal_text(&self) -> String {
        self.items
            .iter()
            .filter_map(|item| match item {
                ToolParserItem::Text(text) => Some(text.as_str()),
                ToolParserItem::Call(_) => None,
            })
            .collect()
    }

    /// Returns the tool-call updates in order.
    pub fn calls(&self) -> impl Iterator<Item = &ToolCallDelta> {
        self.items.iter().filter_map(|item| match item {
            ToolParserItem::Call(call) => Some(call),
            ToolParserItem::Text(_) => None,
        })
    }

    /// Merges consecutive deltas for the same tool call into one item.
    ///
    /// Each merged call takes the first name any of its deltas carries and the
    /// concatenation of their arguments. A call's deltas are contiguous in
    /// parser output, so this yields one item per call there.
    /// `Qwen3XmlToolParser::parse_complete` and the test helper
    /// `collect_stream` use this to collapse streamed argument fragments into
    /// final tool calls.
    pub fn coalesce_calls(self) -> Self {
        let mut coalesced = Self::default();

        for item in self.items {
            match item {
                ToolParserItem::Text(text) => coalesced.push_text(&text),
                ToolParserItem::Call(call) => match coalesced.items.last_mut() {
                    Some(ToolParserItem::Call(existing))
                        if existing.tool_index == call.tool_index =>
                    {
                        if existing.name.is_none() {
                            existing.name = call.name;
                        }
                        existing.arguments.push_str(&call.arguments);
                    }
                    _ => coalesced.push_call(call),
                },
            }
        }

        coalesced
    }
}
