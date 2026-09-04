//! Chat request rendering, structured events, and output processing.
//!
//! Model profiles choose a renderer and request-scoped output processor while
//! this module owns the common chat values and streaming contracts.

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
/// Structured assistant and chat lifecycle events.
pub mod event;
/// Model-selected chat output processors.
pub mod output;
/// Chat messages, options, tools, and validation.
pub mod request;
/// Collection of structured chat event streams.
pub mod stream;
/// Chat-template loading and rendering.
pub mod template;
