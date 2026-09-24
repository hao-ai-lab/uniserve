//! Structured assistant message content.
//!
//! These types describe assistant output in both directions. The block
//! assembler in `output::structured` publishes finished text and reasoning
//! blocks in `RequestOutput::OutputBlockEnd`, and the non-streaming Chat
//! Completions response collects finished blocks and tool calls into an
//! [`AssistantMessage`]. Chat requests carry prior assistant turns as
//! `ChatMessage::Assistant` for the template renderer. The incremental events
//! produced while parsing live in `output::processor`.

use std::ops::Deref;

use serde::{Deserialize, Serialize};

/// One finalized assistant tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantToolCall {
    /// Request-local identifier used to associate a tool response.
    pub id: String,
    /// Function name selected by the model.
    pub name: String,
    /// Function arguments as JSON text. `HfChatRenderer` parses this string
    /// when rendering assistant history and rejects invalid JSON.
    pub arguments: String,
}

/// Semantic kind of one assistant output block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantBlockKind {
    /// Visible final-answer text.
    Text,
    /// Extracted reasoning content.
    Reasoning,
    /// One finalized tool call.
    ToolCall,
}

/// One structured assistant output block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantContentBlock {
    /// Visible final-answer text.
    Text {
        /// Visible text carried by the block.
        text: String,
    },
    /// Extracted reasoning content.
    Reasoning {
        /// Reasoning text carried by the block.
        text: String,
    },
    /// One finalized tool call.
    ToolCall(AssistantToolCall),
}

impl AssistantContentBlock {
    /// Returns the semantic kind of this block.
    pub fn kind(&self) -> AssistantBlockKind {
        match self {
            Self::Text { .. } => AssistantBlockKind::Text,
            Self::Reasoning { .. } => AssistantBlockKind::Reasoning,
            Self::ToolCall(..) => AssistantBlockKind::ToolCall,
        }
    }

    /// Returns this block as one finalized tool call, if applicable.
    pub fn as_tool_call(&self) -> Option<&AssistantToolCall> {
        match self {
            Self::ToolCall(call) => Some(call),
            _ => None,
        }
    }

    /// Returns this block with leading and trailing whitespace trimmed from all text
    /// fields and tool call arguments, or `None` if the resulting text would be empty.
    ///
    /// Tool-call blocks are always kept, even with empty arguments.
    pub fn trim(mut self) -> Option<Self> {
        match &mut self {
            Self::Text { text } | Self::Reasoning { text } => {
                let trimmed_text = text.trim();
                if trimmed_text.is_empty() {
                    return None;
                } else {
                    *text = trimmed_text.to_string();
                }
            }
            Self::ToolCall(call) => {
                call.arguments = call.arguments.trim().to_string();
            }
        }
        Some(self)
    }
}

/// Final structured assistant message assembled from the event stream.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantMessage {
    /// Ordered semantic content blocks in the assistant response.
    pub content: Vec<AssistantContentBlock>,
}

impl Deref for AssistantMessage {
    type Target = [AssistantContentBlock];

    /// Exposes the ordered content blocks as a slice.
    fn deref(&self) -> &Self::Target {
        &self.content
    }
}

impl AssistantMessage {
    /// Concatenates all visible final-answer text blocks.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                AssistantContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Concatenates all extracted reasoning blocks, or returns `None` when the
    /// concatenation is empty.
    pub fn reasoning(&self) -> Option<String> {
        Some(
            self.content
                .iter()
                .filter_map(|block| match block {
                    AssistantContentBlock::Reasoning { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect(),
        )
        .filter(|text: &String| !text.is_empty())
    }

    /// Returns whether this assistant message contains any non-empty reasoning text blocks.
    pub fn has_reasoning(&self) -> bool {
        self.content.iter().any(|block| match block {
            AssistantContentBlock::Reasoning { text } => !text.is_empty(),
            _ => false,
        })
    }

    /// Returns finalized assistant tool calls in encounter order.
    pub fn tool_calls(&self) -> impl Iterator<Item = &AssistantToolCall> {
        self.content
            .iter()
            .filter_map(AssistantContentBlock::as_tool_call)
    }

    /// Returns whether this assistant message contains any tool-call blocks.
    pub fn has_tool_calls(&self) -> bool {
        self.content
            .iter()
            .any(|block| matches!(block, AssistantContentBlock::ToolCall(_)))
    }

    /// Pushes one new block to the end of the message content.
    pub fn push_block(&mut self, block: AssistantContentBlock) {
        self.content.push(block);
    }

    /// Returns this message with leading and trailing whitespace trimmed from all text
    /// fields and tool call arguments, and with any blocks that are empty after trimming removed.
    pub fn trim(mut self) -> Self {
        self.content = self
            .content
            .into_iter()
            .filter_map(|block| block.trim())
            .collect();
        self
    }
}
