//! Streaming tool-call parsers and normalized tool descriptors.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#[macro_use]
mod error;
mod json;
#[cfg(any(test, feature = "test-util"))]
pub mod test_utils;
mod utils;

use std::collections::{BTreeMap, btree_map};

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
    /// Arguments text contributed by this update.
    pub arguments: String,
}

/// Result of advancing tool parsing with one assistant-text input.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolParserOutput {
    /// Plain assistant text that is not part of any tool call.
    pub normal_text: String,
    /// Tool-call updates extracted from this input.
    pub calls: Vec<ToolCallDelta>,
}

impl ToolParserOutput {
    /// Appends another parser output onto this one.
    ///
    /// Note that this does not attempt to merge multiple deltas for the same
    /// tool call into one complete item. Call `coalesce_calls` after if
    /// that behavior is desired.
    pub fn append(&mut self, mut other: Self) {
        self.normal_text.push_str(&other.normal_text);
        self.calls.append(&mut other.calls);
    }

    /// Merges multiple deltas for the same tool call into one complete item.
    ///
    /// This is primarily used by the default `parse_complete` implementation,
    /// which delegates through the incremental parser lifecycle and then
    /// needs to collapse streaming-style argument fragments into one final
    /// tool call.
    pub fn coalesce_calls(mut self) -> Self {
        let mut merged = BTreeMap::<usize, ToolCallDelta>::new();
        let mut order = Vec::new();

        for call in self.calls {
            match merged.entry(call.tool_index) {
                btree_map::Entry::Vacant(entry) => {
                    order.push(call.tool_index);
                    entry.insert(call);
                }
                btree_map::Entry::Occupied(mut entry) => {
                    let existing = entry.get_mut();
                    if existing.name.is_none() {
                        existing.name = call.name;
                    }
                    existing.arguments.push_str(&call.arguments);
                }
            }
        }

        self.calls = order
            .into_iter()
            .filter_map(|tool_index| merged.remove(&tool_index))
            .collect();
        self
    }
}
