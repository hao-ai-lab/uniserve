//! Incremental assistant block assembly into the public request output.

use crate::serving::RequestOutput;
use crate::serving::chat::{AssistantBlockKind, AssistantContentBlock, Error, Result};

/// One currently open assistant text-like block being assembled from streamed
/// deltas.
struct OpenTextBlock {
    /// Stable position of this block in the final assistant message.
    index: usize,
    /// Semantic kind of the block being assembled.
    kind: AssistantBlockKind,
    /// Accumulated text payload for the block.
    text: String,
}

/// One currently open assistant tool call being assembled from streamed deltas.
struct OpenToolCall {
    /// Stable ordinal of this tool call in the assistant tool-call list.
    index: usize,
    /// Stable tool-call ID exposed northbound.
    id: String,
    /// Function name.
    name: String,
    /// Incremental JSON arguments accumulated so far.
    arguments: String,
}

/// Per-stream block assembly state.
///
/// The processor maintains at most one open text block and one open tool call,
/// and appends deltas to them until the semantic kind changes or the stream
/// terminates.
pub(crate) struct OutputProcessor {
    /// Number of blocks already delivered to the output consumer.
    num_completed_blocks: usize,
    /// Currently open text or reasoning block, if any.
    open_text_block: Option<OpenTextBlock>,
    /// Currently open tool call, if any.
    open_tool_call: Option<OpenToolCall>,
    /// Next OpenAI-compatible tool-call ordinal.
    next_tool_call_index: usize,
}

impl OutputProcessor {
    /// Creates one fresh assembly state for a new streamed response.
    pub(crate) fn new() -> Self {
        Self {
            num_completed_blocks: 0,
            open_text_block: None,
            open_tool_call: None,
            next_tool_call_index: 0,
        }
    }

    /// Converts one parsed text delta into zero or more structured chat events.
    pub(crate) fn process_text_delta(
        &mut self,
        kind: AssistantBlockKind,
        delta: String,
    ) -> Vec<RequestOutput> {
        let mut events = Vec::new();
        self.close_open_tool_call(&mut events);
        self.push_text_delta(kind, delta, &mut events);
        events
    }

    /// Starts one new tool call, closing any incompatible open block first.
    pub(crate) fn start_tool_call(&mut self, id: String, name: String) -> Vec<RequestOutput> {
        let mut events = Vec::new();
        self.close_open_text_block(&mut events);
        self.close_open_tool_call(&mut events);

        let index = self.next_tool_call_index;
        self.next_tool_call_index += 1;
        self.open_tool_call = Some(OpenToolCall {
            index,
            id: id.clone(),
            name: name.clone(),
            arguments: String::new(),
        });
        events.push(RequestOutput::ToolCallStart { index, id, name });
        events
    }

    /// Appends one incremental tool-call arguments delta.
    pub(crate) fn push_tool_call_arguments(&mut self, delta: String) -> Result<Vec<RequestOutput>> {
        let mut events = Vec::new();
        let Some(open_tool_call) = self.open_tool_call.as_mut() else {
            return Err(Error::ToolCallStreamInvariant {
                message: "received tool-call arguments delta without an open tool call".to_string(),
            });
        };
        open_tool_call.arguments.push_str(&delta);
        events.push(RequestOutput::ToolCallArgumentsDelta {
            index: open_tool_call.index,
            delta,
        });
        Ok(events)
    }

    /// Closes the remaining blocks before the caller emits terminal usage.
    pub(crate) fn finish(&mut self) -> Vec<RequestOutput> {
        let mut events = Vec::new();
        self.close_open_text_block(&mut events);
        self.close_open_tool_call(&mut events);
        events
    }

    /// Appends one semantic text delta to the current block, or open a new block
    /// when the semantic kind changes.
    fn push_text_delta(
        &mut self,
        kind: AssistantBlockKind,
        delta: String,
        events: &mut Vec<RequestOutput>,
    ) {
        if delta.is_empty() {
            return;
        }

        match self.open_text_block.as_mut() {
            // If there's a currently open block of the same kind, append to it.
            Some(open_block) if open_block.kind == kind => {
                open_block.text.push_str(&delta);
                push_delta(events, kind, delta);
            }
            // Otherwise, close the currently open block (if any) and start a
            // new one.
            _ => {
                self.close_open_text_block(events);
                let index = self.num_completed_blocks;
                self.open_text_block = Some(OpenTextBlock {
                    index,
                    kind,
                    text: delta.clone(),
                });
                events.push(RequestOutput::OutputBlockStart { index, kind });
                push_delta(events, kind, delta);
            }
        }
    }

    /// Finalizes the currently open text block, if present.
    fn close_open_text_block(&mut self, events: &mut Vec<RequestOutput>) {
        let Some(open_block) = self.open_text_block.take() else {
            return;
        };

        let block = match open_block.kind {
            AssistantBlockKind::Text => AssistantContentBlock::Text {
                text: open_block.text,
            },
            AssistantBlockKind::Reasoning => AssistantContentBlock::Reasoning {
                text: open_block.text,
            },
            AssistantBlockKind::ToolCall => {
                unreachable!("tool calls must not be assembled as text blocks")
            }
        };
        self.num_completed_blocks += 1;
        events.push(RequestOutput::OutputBlockEnd {
            index: open_block.index,
            block,
        });
    }

    /// Finalizes the currently open tool call, if present.
    fn close_open_tool_call(&mut self, events: &mut Vec<RequestOutput>) {
        let Some(open_tool_call) = self.open_tool_call.take() else {
            return;
        };

        self.num_completed_blocks += 1;
        events.push(RequestOutput::ToolCallEnd {
            index: open_tool_call.index,
            id: open_tool_call.id,
            name: open_tool_call.name,
            arguments: open_tool_call.arguments,
        });
    }
}

/// Text and reasoning are distinct public deltas; block indices are carried
/// by the opening and closing events.
fn push_delta(events: &mut Vec<RequestOutput>, kind: AssistantBlockKind, text: String) {
    events.push(match kind {
        AssistantBlockKind::Text => RequestOutput::TextDelta {
            text,
            token_ids: Vec::new(),
            logprobs: None,
        },
        AssistantBlockKind::Reasoning => RequestOutput::ReasoningDelta { text },
        AssistantBlockKind::ToolCall => unreachable!("tool calls use argument deltas"),
    });
}
