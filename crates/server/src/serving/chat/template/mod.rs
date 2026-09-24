//! Chat template loading options and renderer implementations.
//!
//! The only renderer is the Hugging Face Jinja renderer in `renderer::hf`.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
/// Chat-template renderer implementations.
pub mod renderer;

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
    /// Message-content representation expected by the template.
    pub chat_template_content_format: ChatTemplateContentFormatOption,
    /// Optional template that overrides the model's own template: either a
    /// path to a template file (JSON when its extension is `.json`, Jinja
    /// otherwise) or inline Jinja source.
    pub chat_template: Option<String>,
    /// Default keyword arguments supplied to every template render.
    pub default_chat_template_kwargs: HashMap<String, Value>,
}
