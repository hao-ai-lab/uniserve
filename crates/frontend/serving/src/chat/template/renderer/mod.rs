use std::sync::Arc;

use crate::text::Prompt;

use crate::chat::template::error::Result;
use crate::chat::template::request::ChatRequest;

pub mod hf;

/// Rendered chat prompt submitted to the text backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPrompt {
    pub prompt: Prompt,
}

/// Minimal chat-prompt renderer used by `chat`.
pub trait ChatRenderer: Send + Sync {
    /// Render one chat request into the text prompt submitted to the text
    /// backend.
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt>;
}

/// Shared trait-object form of [`ChatRenderer`].
pub type DynChatRenderer = Arc<dyn ChatRenderer>;
