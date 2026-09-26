//! Streaming tool-call parsers and normalized tool descriptors.
//!
//! The chat output tool stage (`serving::chat::output`) feeds visible
//! assistant text chunk by chunk to a [`ToolParser`]: `Qwen3XmlToolParser`
//! for Qwen3 and `Gemma4ToolParser` for Gemma-4. The parser turns each chunk
//! into an ordered sequence of plain text and `ToolCallDelta` updates, holding
//! back bytes it cannot classify yet, such as a partial marker or an
//! incomplete tool call.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#[macro_use]
mod error;
mod gemma4;
mod json;
#[cfg(any(test, feature = "test-util"))]
pub mod test_utils;
mod utils;

pub use error::{Result, ToolParserError};
pub use gemma4::Gemma4ToolParser;
pub use json::Qwen3XmlToolParser;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Incremental extraction of tool calls from one assistant turn's visible
/// text.
///
/// Call `parse_into` for each text chunk and `finish` once at end of
/// generation. After a failure, `reset` recovers the input that emitted
/// output does not represent, so the caller can surface it as text.
pub trait ToolParser: Send {
    /// Parses one text chunk into an existing output accumulator.
    ///
    /// Fails with `ToolParserError::ParsingFailed` when the input cannot be
    /// interpreted under the parser's grammar or its pending buffer exceeds
    /// the parser's size cap. Events parsed before the failure remain in
    /// `output`, and `reset` returns the input that `output` does not
    /// represent.
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()>;

    /// Flushes buffered parser state at end of generation.
    ///
    /// Buffered input that belongs to no published tool call is returned as
    /// plain text. Fails, leaving the state unchanged, when a published call
    /// is still open.
    fn finish(&mut self) -> Result<ToolParserOutput>;

    /// Resets parser state and returns the buffered input that no emitted
    /// output represents.
    fn reset(&mut self) -> String;

    #[cfg(any(test, feature = "test-util"))]
    /// Parses one incremental text chunk.
    fn parse_chunk(&mut self, chunk: &str) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        self.parse_into(chunk, &mut output)?;
        Ok(output)
    }

    #[cfg(any(test, feature = "test-util"))]
    /// Parses a complete assistant response and flushes all state.
    fn parse_complete(&mut self, text: &str) -> Result<ToolParserOutput> {
        let mut output = self.parse_chunk(text)?;
        output.append(self.finish()?);
        Ok(output.coalesce_calls())
    }
}

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
    /// for one `tool_index` yields the call's arguments as JSON text.
    /// `Qwen3XmlToolParser` forwards the JSON exactly as the model wrote it,
    /// neither parsed nor normalized; `Gemma4ToolParser` translates Gemma-4's
    /// native argument encoding into JSON.
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
    /// `ToolParser::parse_complete` and the test helper `collect_stream` use
    /// this to collapse streamed argument fragments into final tool calls.
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
