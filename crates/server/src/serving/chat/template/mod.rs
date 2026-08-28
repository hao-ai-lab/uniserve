//! Chat template rendering implementations.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub mod error;
pub mod renderer;

pub use error::{Error, Result};
pub use renderer::hf::{ChatTemplateContentFormatOption, HfChatRenderer};

pub use crate::serving::chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart, ChatMessage,
    ChatRequest, ChatRole, ChatToolChoice, GenerationPromptMode, ReasoningEffort, Tool,
};

use std::collections::HashMap;

use serde_json::Value;

/// Options needed to load a chat template renderer from model files.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChatTemplateLoadOptions {
    pub chat_template_content_format: ChatTemplateContentFormatOption,
    pub chat_template: Option<String>,
    pub default_chat_template_kwargs: HashMap<String, Value>,
}
