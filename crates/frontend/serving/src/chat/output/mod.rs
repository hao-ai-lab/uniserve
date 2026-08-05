//! Chat assistant output processing.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub mod error;
pub mod parser;

#[path = "output/mod.rs"]
pub(crate) mod processor;

pub use error::{Error, Result};
pub use processor::{DynChatEventStream, DynDecodedTextEventStream, Qwen3ChatOutputProcessor};

pub use crate::chat::protocol::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall, ChatEvent, ChatRequest, ChatTool, ChatToolChoice,
};
pub use crate::text::FinishReason;
pub use uniserve_model_profile::reasoning::ReasoningDelta;
pub use uniserve_model_profile::tools::ToolParserError;

pub mod event {
    pub use crate::chat::protocol::{
        AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
        AssistantToolCall, ChatEvent,
    };
}

pub mod request {
    pub use crate::chat::protocol::{
        ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool,
        ChatToolChoice, GenerationPromptMode, ReasoningEffort,
    };
}
