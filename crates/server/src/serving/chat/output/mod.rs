//! Assistant output processing for chat generation.
//!
//! Decoded text passes through the Qwen3 reasoning stage and then the tool-call
//! stage (`qwen3`), which produce the [`AssistantEvent`] stream defined in
//! `processor`. `assemble_chat_event_stream` drives that stream and uses
//! `structured::OutputProcessor` to turn assistant events into content-block
//! and tool-call serving events.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub(crate) mod processor;
mod qwen3;
pub(crate) mod structured;
pub use processor::AssistantEvent;

pub use qwen3::Qwen3ChatOutputProcessor;

pub use crate::profile::reasoning::ReasoningDelta;
pub use crate::profile::tools::ToolParserError;
pub use crate::serving::chat::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantToolCall, ChatRequest,
    ChatToolChoice, Tool,
};
pub use crate::serving::text::FinishReason;
