//! Chat assistant output processing.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub mod error;
pub(crate) mod processor;
mod qwen3;
mod structured;

pub use error::{Error, Result};
pub use qwen3::Qwen3ChatOutputProcessor;

pub use crate::profile::reasoning::ReasoningDelta;
pub use crate::profile::tools::ToolParserError;
pub use crate::serving::chat::protocol::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantToolCall, ChatEvent,
    ChatRequest, ChatToolChoice, Tool,
};
pub use crate::serving::text::FinishReason;
