//! Assistant output-processor traits and implementations.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub(crate) mod processor;
mod qwen3;
mod structured;

pub use qwen3::Qwen3ChatOutputProcessor;

pub use crate::profile::reasoning::ReasoningDelta;
pub use crate::profile::tools::ToolParserError;
pub use crate::serving::chat::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantToolCall, ChatEvent,
    ChatRequest, ChatToolChoice, Tool,
};
pub use crate::serving::text::FinishReason;
