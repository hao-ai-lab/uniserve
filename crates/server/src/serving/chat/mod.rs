//! Chat rendering and output-processing library.
//!
//! This description-owned library provides the fixed Hugging Face chat renderer,
//! chat request and event values, and the request-scoped Qwen3 output processor.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub use crate::profile::reasoning::{ReasoningDelta, ReasoningError};
pub use crate::profile::tools::ToolParserError;
pub use crate::serving::text::{FinishReason, StopReason};
pub use error::{Error, Result};
pub use event::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantToolCall, ChatEvent,
};
pub use output::Qwen3ChatOutputProcessor;
pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatToolChoice,
    GenerationPromptMode, ReasoningEffort, Tool,
};
pub use stream::CollectedAssistantMessage;
pub use template::ChatTemplateLoadOptions;
pub use template::renderer::hf::{ChatTemplateContentFormatOption, HfChatRenderer};

mod error;
pub mod event;
pub mod output;
pub mod request;
pub mod stream;
pub mod template;
