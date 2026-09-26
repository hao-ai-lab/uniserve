//! Chat request rendering, structured events, and output processing.
//!
//! This module owns the common chat values and streaming contracts. For a
//! Qwen3 model, a chat request flows through it twice. Before submission,
//! `InputProcessor` validates the [`ChatRequest`], builds a request-scoped
//! [`Qwen3ChatOutputProcessor`], and renders the request into a prompt with
//! [`HfChatRenderer`]. During generation, `assemble_chat_event_stream` feeds
//! decoded text through that processor and assembles the resulting assistant
//! events into public serving events. [`Gemma4ChatOutputProcessor`] composes
//! the same output stages for Gemma-4's chat format. The omni models render
//! chat requests with the same renderer, but their output bypasses this
//! module's output processors.
//!
//! The template comes from the model files or [`ChatTemplateLoadOptions`], and
//! the server's `reasoning_parsing` setting decides whether Qwen3 `<think>`
//! sections become reasoning blocks.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub use crate::profile::reasoning::{ReasoningDelta, ReasoningError};
pub use crate::profile::tools::ToolParserError;
pub use crate::serving::text::{FinishReason, StopReason};
pub use error::{Error, Result};
pub use event::{AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantToolCall};
pub use output::{Gemma4ChatOutputProcessor, Qwen3ChatOutputProcessor};
pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatToolChoice,
    GenerationPromptMode, ReasoningEffort, Tool,
};
pub use template::ChatTemplateLoadOptions;
pub use template::renderer::hf::{ChatTemplateContentFormatOption, HfChatRenderer};

mod error;
/// Structured assistant message content.
pub mod event;
/// Model-selected chat output processors.
pub mod output;
/// Chat messages, options, tools, and validation.
pub mod request;
/// Chat-template loading and rendering.
pub mod template;
