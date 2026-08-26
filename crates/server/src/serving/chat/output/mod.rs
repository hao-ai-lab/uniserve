//! Chat assistant output processing.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub mod error;
pub mod parser;
pub(crate) mod processor;
mod qwen3;
mod structured;

pub use error::{Error, Result};
pub use processor::{DynChatEventStream, DynDecodedTextEventStream};
pub use qwen3::Qwen3ChatOutputProcessor;

pub use crate::profile::reasoning::ReasoningDelta;
pub use crate::profile::tools::ToolParserError;
pub use crate::serving::chat::protocol::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall, ChatEvent, ChatRequest, ChatToolChoice, Tool,
};
pub use crate::serving::text::FinishReason;

pub mod event {
    pub use crate::serving::chat::protocol::{
        AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
        AssistantToolCall, ChatEvent,
    };
}

pub mod request {
    pub use crate::serving::chat::protocol::{
        ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole,
        ChatToolChoice, GenerationPromptMode, ReasoningEffort, Tool,
    };
}
