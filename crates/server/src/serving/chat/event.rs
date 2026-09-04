//! Structured assistant messages and incremental chat events.

use std::ops::Deref;
use std::sync::Arc;

use crate::serving::text::{DecodedLogprobs, DecodedPromptLogprobs};
use serde::{Deserialize, Serialize};

use crate::serving::text::FinishReason;

/// One finalized assistant tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantToolCall {
    /// Request-local identifier used to associate a tool response.
    pub id: String,
    /// Function name selected by the model.
    pub name: String,
    /// Serialized function arguments.
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

    /// Returns shared access to the wrapped value.
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

    /// Concatenates all extracted reasoning blocks, if any.
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

/// Streamed chat event emitted by the chat facade.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatEvent {
    /// The request was accepted, streaming has started, and prompt metadata is
    /// ready.
    Start {
        /// The actual prompt token IDs for this request.
        prompt_token_ids: Arc<[u32]>,
        /// Once-only prompt logprobs metadata, when requested.
        prompt_logprobs: Option<DecodedPromptLogprobs>,
        /// Monotonic timestamp at which the request entered the serving queue.
        queued_at: Option<f64>,
        /// Monotonic timestamp at which engine execution began.
        scheduled_at: Option<f64>,
    },
    /// A new assistant output block has started.
    BlockStart {
        /// Zero-based content-block index.
        index: usize,
        /// Semantic kind of the opened block.
        kind: AssistantBlockKind,
    },
    /// A newly observed delta for one open assistant output block.
    BlockDelta {
        /// Zero-based content-block index.
        index: usize,
        /// Semantic kind of the open block.
        kind: AssistantBlockKind,
        /// Newly decoded block text.
        delta: String,
    },
    /// Per-decoded-update sample metadata: logprobs and/or output token IDs.
    LogprobsDelta {
        /// Per-position candidate log probabilities, when requested.
        logprobs: Option<DecodedLogprobs>,
        /// Token identifiers represented by this update.
        token_ids: Vec<u32>,
    },
    /// One assistant output block has ended.
    BlockEnd {
        /// Zero-based content-block index.
        index: usize,
        /// Complete normalized block content.
        block: AssistantContentBlock,
    },
    /// One tool call has started.
    ToolCallStart {
        /// Zero-based tool-call index.
        index: usize,
        /// Request-local tool-call identifier.
        id: String,
        /// Selected function name.
        name: String,
    },
    /// One incremental tool-call arguments delta for the currently open tool
    /// call.
    ToolCallArgumentsDelta {
        /// Zero-based tool-call index.
        index: usize,
        /// Newly decoded serialized argument text.
        delta: String,
    },
    /// One tool call has ended.
    ToolCallEnd {
        /// Zero-based tool-call index.
        index: usize,
        /// Complete normalized tool call.
        call: AssistantToolCall,
    },
    /// Terminal event carrying the final assembled assistant message and finish
    /// metadata.
    Done {
        /// Final assembled assistant message.
        message: AssistantMessage,
        /// Number of prompt tokens actually sent to the engine after chat
        /// template rendering and tokenization.
        prompt_token_count: usize,
        /// Number of output tokens generated.
        output_token_count: usize,
        /// Number of generated tokens included in user-visible output.
        visible_output_token_count: usize,
        /// Number of generated tokens consumed by internal protocol sections.
        internal_token_count: usize,
        /// Terminal condition reported by the engine or output assembler.
        finish_reason: FinishReason,
    },
}
