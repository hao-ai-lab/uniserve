//! Chat request and event vocabulary shared by API conversion, rendering, and
//! chat facades.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub mod error;
pub mod event;
pub mod request;

pub use error::{Error, Result};
pub use event::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantToolCall, ChatEvent,
};
pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatToolChoice,
    GenerationPromptMode, ReasoningEffort, Tool,
};
