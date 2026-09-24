//! Incremental assistant block assembly into the public request output.
//!
//! `assemble_chat_event_stream` feeds parsed assistant events into
//! [`OutputProcessor`], which frames them as public events: text and reasoning
//! become `OutputBlockStart`, deltas, and `OutputBlockEnd`; tool calls become
//! `ToolCallStart`, `ToolCallArgumentsDelta`, and `ToolCallEnd`. Two index
//! spaces are in use. Content-block indices count every closed block, tool
//! calls included, so they give the block's position among all assistant
//! content blocks. Tool-call indices are ordinals among tool calls only; the
//! Chat Completions stream publishes them as the tool-call `index`.

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
/// At most one block, text-like or tool call, is open at a time: a text delta
/// closes an open tool call, a change of text kind closes the open text block,
/// and a tool-call start closes whichever block is open. Deltas append to the
/// open block until it closes or the stream terminates.
pub(crate) struct OutputProcessor {
    /// Number of closed blocks, text-like and tool calls alike. The next
    /// opened text block takes this value as its content-block index.
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
    ///
    /// Any open tool call is closed first, even when `delta` is empty.
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
    ///
    /// # Errors
    ///
    /// Returns `Error::ToolCallStreamInvariant` when no tool call is open.
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

    /// Appends one semantic text delta to the current block, or opens a new
    /// block when the semantic kind changes. Empty deltas are ignored.
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

/// Emits the public delta event for one text or reasoning chunk.
///
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
