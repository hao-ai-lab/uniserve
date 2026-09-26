//! Assistant output processing for chat generation.
//!
//! Decoded text passes through the reasoning stage (`reasoning`) and then the
//! tool-call stage (`tool`), which produce the [`AssistantEvent`] stream
//! defined in `processor`. Each model family composes the two stages with its
//! own parsers: `qwen3` for Qwen3 and `gemma4` for Gemma-4.
//! `assemble_chat_event_stream` drives that stream and uses
//! `structured::OutputProcessor` to turn assistant events into content-block
//! and tool-call serving events.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod gemma4;
pub(crate) mod processor;
mod qwen3;
mod reasoning;
pub(crate) mod structured;
mod tool;
pub use processor::AssistantEvent;

pub use gemma4::Gemma4ChatOutputProcessor;
pub use qwen3::{Qwen3ChatOutputProcessor, template_instructs_json_tool_calls};

pub use crate::profile::reasoning::ReasoningDelta;
pub use crate::profile::tools::ToolParserError;
pub use crate::serving::chat::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantToolCall, ChatRequest,
    ChatToolChoice, Tool,
};
pub use crate::serving::text::FinishReason;
