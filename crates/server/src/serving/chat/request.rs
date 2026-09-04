//! Internal chat messages, content parts, tools, and render options.

pub use crate::profile::tools::Tool;
use crate::serving::text::TextDecodeOptions;
use llm_multimodal::ImageDetail;
use serde::{Deserialize, Serialize};

use crate::serving::chat::error::{Error, Result};
use crate::serving::chat::event::{AssistantContentBlock, AssistantMessage};

/// Role label for one text-only chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    /// System-level instruction message.
    System,
    /// Developer-level instruction message.
    Developer,
    /// End-user input message.
    User,
    /// Model-generated response message.
    Assistant,
    /// Result of a previously requested tool call.
    ToolResponse,
}

/// One text-only chat content part in OpenAI-style block format.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    /// One plain-text content block.
    Text {
        /// Plain text carried by the block.
        text: String,
    },
    /// One image URL/data URL content block.
    ImageUrl {
        /// URL or data URL containing the image.
        image_url: String,
        /// Requested image-detail policy.
        detail: Option<ImageDetail>,
        /// Optional stable image identity used for request-local reuse.
        uuid: Option<String>,
    },
}

impl ChatContentPart {
    /// Constructs one text content part with plain string content.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Constructs one image URL content part with the given URL string.
    pub fn image_url(image_url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: image_url.into(),
            detail: None,
            uuid: None,
        }
    }

    /// Returns the text content of this part when it is a text block, or an
    /// "unsupported multimodal content" error otherwise.
    pub fn as_text(&self) -> Result<&str> {
        match self {
            Self::Text { text } => Ok(text),
            Self::ImageUrl { .. } => Err(Error::UnsupportedMultimodalContent("image_url")),
        }
    }

    /// Returns whether this part is a text block with empty content.
    pub fn is_empty_text(&self) -> bool {
        matches!(self, Self::Text { text } if text.is_empty())
    }

    /// Returns whether this part contains any multimodal content.
    pub fn is_multimodal(&self) -> bool {
        match self {
            Self::Text { .. } => false,
            Self::ImageUrl { .. } => true,
        }
    }
}

/// Text-only chat content.
///
/// This supports either a simple string or an OpenAI-style list of text blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    /// Simple text content.
    Text(String),
    /// OpenAI-style blocks.
    Parts(Vec<ChatContentPart>),
}

impl ChatContent {
    /// Flattens text-only content or returns an error for multimodal parts.
    pub fn try_flatten_to_text(&self) -> Result<String> {
        Ok(match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .map(ChatContentPart::as_text)
                .collect::<Result<Vec<_>>>()?
                .concat(),
        })
    }

    /// Returns whether there is no text content or only empty text blocks.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Parts(parts) => parts.iter().all(ChatContentPart::is_empty_text),
        }
    }

    /// Returns whether this content contains any multimodal parts.
    pub fn has_multimodal(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts.iter().any(ChatContentPart::is_multimodal),
        }
    }
}

impl From<String> for ChatContent {
    /// Converts the source value into this type.
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for ChatContent {
    /// Converts the source value into this type.
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<Vec<ChatContentPart>> for ChatContent {
    /// Converts the source value into this type.
    fn from(value: Vec<ChatContentPart>) -> Self {
        Self::Parts(value)
    }
}

/// One chat message.
///
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ChatMessage {
    /// System message content.
    System {
        /// System instruction content.
        content: ChatContent,
    },
    /// Developer message content plus optional message-local tools.
    Developer {
        /// Developer instruction content.
        content: ChatContent,
        /// Message-local function tools made available to the model.
        tools: Option<Vec<Tool>>,
    },
    /// User message content.
    User {
        /// End-user content.
        content: ChatContent,
    },
    /// Assistant history content assembled from structured assistant blocks.
    Assistant {
        /// Structured assistant response content.
        content: AssistantMessage,
    },
    /// Tool response content associated with one prior assistant tool call.
    ToolResponse {
        /// Tool result content returned to the model.
        content: ChatContent,
        /// Identifier of the assistant tool call being answered.
        tool_call_id: String,
    },
}

impl ChatMessage {
    /// Constructs one chat message with plain string content.
    ///
    /// # Panics
    ///
    /// Panics for [`ChatRole::ToolResponse`], which requires a tool-call identifier.
    pub fn text(role: ChatRole, text: impl Into<String>) -> Self {
        let content: String = text.into();

        match role {
            ChatRole::System => Self::system(content),
            ChatRole::Developer => Self::developer(content, None),
            ChatRole::User => Self::user(content),
            ChatRole::Assistant => Self::assistant_text(content),
            ChatRole::ToolResponse => {
                panic!(
                    "tool response messages require a tool_call_id; \
                     use ChatMessage::tool_response() instead"
                )
            }
        }
    }

    /// Constructs one chat message with system role.
    pub fn system(content: impl Into<ChatContent>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    /// Constructs one chat message with developer role.
    pub fn developer(content: impl Into<ChatContent>, tools: Option<Vec<Tool>>) -> Self {
        Self::Developer {
            content: content.into(),
            tools,
        }
    }

    /// Constructs one chat message with user role.
    pub fn user(content: impl Into<ChatContent>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    /// Constructs one chat message with assistant role and plain string content.
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::Assistant {
            content: AssistantMessage {
                content: vec![AssistantContentBlock::Text { text: text.into() }],
            },
        }
    }

    /// Constructs one chat message with assistant role and structured content
    /// blocks.
    pub fn assistant_blocks(content: Vec<AssistantContentBlock>) -> Self {
        Self::Assistant {
            content: AssistantMessage { content },
        }
    }

    /// Constructs one tool-role message.
    pub fn tool_response(content: impl Into<ChatContent>, tool_call_id: impl Into<String>) -> Self {
        Self::ToolResponse {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
        }
    }

    /// Returns the chat role of this message.
    pub fn role(&self) -> ChatRole {
        match self {
            Self::System { .. } => ChatRole::System,
            Self::Developer { .. } => ChatRole::Developer,
            Self::User { .. } => ChatRole::User,
            Self::Assistant { .. } => ChatRole::Assistant,
            Self::ToolResponse { .. } => ChatRole::ToolResponse,
        }
    }

    /// Concatenates the visible text carried by this message.
    pub fn text_content(&self) -> Result<String> {
        match self {
            Self::System { content }
            | Self::Developer { content, .. }
            | Self::User { content }
            | Self::ToolResponse { content, .. } => content.try_flatten_to_text(),
            Self::Assistant { content } => Ok(content.text()),
        }
    }

    /// Concatenates assistant reasoning text when present.
    pub fn reasoning_content(&self) -> Option<String> {
        match self {
            Self::Assistant { content } => content.reasoning(),
            Self::System { .. }
            | Self::Developer { .. }
            | Self::User { .. }
            | Self::ToolResponse { .. } => None,
        }
    }

    /// Returns whether this message contains any multimodal content.
    pub fn has_multimodal(&self) -> bool {
        match self {
            Self::System { content }
            | Self::Developer { content, .. }
            | Self::User { content }
            | Self::ToolResponse { content, .. } => content.has_multimodal(),
            Self::Assistant { .. } => false,
        }
    }
}

impl From<AssistantMessage> for ChatMessage {
    /// Converts the source value into this type.
    fn from(value: AssistantMessage) -> Self {
        Self::Assistant { content: value }
    }
}

/// Controls how prompt rendering should end after the existing chat history.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationPromptMode {
    /// Append a generation prompt for a new assistant turn.
    ///
    /// Equivalent to `add_generation_prompt = true` and `continue_final_message
    /// = false`.
    #[default]
    StartNewAssistant,
    /// Leave the final assistant message open so generation continues it.
    ///
    /// Equivalent to `add_generation_prompt = false` and
    /// `continue_final_message = true`.
    ContinueFinalAssistant,
    /// Render the existing chat history without adding any trailing generation
    /// prompt.
    ///
    /// Equivalent to `add_generation_prompt = false` and
    /// `continue_final_message = false`.
    NoGenerationPrompt,
}

/// Semantic effort level for reasoning models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Disables model reasoning.
    None,
    /// Requests the smallest available reasoning budget.
    Minimal,
    /// Requests a low reasoning budget.
    Low,
    /// Requests a medium reasoning budget.
    Medium,
    /// Requests a high reasoning budget.
    High,
    /// Requests an extra-high reasoning budget.
    XHigh,
    /// Requests the maximum available reasoning budget.
    Max,
}

/// Chat-template-related request options.
///
/// The options are exposed to prompt templates before tokenization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatOptions {
    /// Controls whether rendering starts a new assistant turn, continues the
    /// final assistant message, or emits no trailing generation prompt at
    /// all.
    pub generation_prompt_mode: GenerationPromptMode,

    /// Effort level exposed to chat templates for reasoning models.
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl Default for ChatOptions {
    /// Returns the default value.
    fn default() -> Self {
        Self {
            generation_prompt_mode: GenerationPromptMode::StartNewAssistant,
            reasoning_effort: None,
        }
    }
}

impl ChatOptions {
    /// Returns whether to add a generation prompt for a new assistant turn after the
    /// existing chat history.
    pub fn add_generation_prompt(&self) -> bool {
        matches!(
            self.generation_prompt_mode,
            GenerationPromptMode::StartNewAssistant
        )
    }

    /// Returns whether to leave the final assistant message open so generation
    /// continues it.
    pub fn continue_final_message(&self) -> bool {
        matches!(
            self.generation_prompt_mode,
            GenerationPromptMode::ContinueFinalAssistant
        )
    }
}

/// Tool-choice semantics supported by `chat`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatToolChoice {
    /// Allows the model to choose between text and tool calls.
    Auto,
    /// Prevents the model from emitting tool calls.
    #[default]
    None,
}

/// One chat request ready to be rendered into a prompt and lowered into a
/// generate request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Ordered chat history to render.
    pub messages: Vec<ChatMessage>,
    /// Chat-specific rendering options.
    pub chat_options: ChatOptions,
    /// Function tools made available to the model for this request.
    pub tools: Vec<Tool>,
    /// Tool-choice behavior for this request.
    pub tool_choice: ChatToolChoice,
    /// Text decode options for incremental detokenization.
    pub decode_options: TextDecodeOptions,
}

impl ChatRequest {
    /// Returns one minimal valid request fixture for tests.
    pub fn for_test() -> Self {
        Self {
            messages: vec![ChatMessage::text(ChatRole::User, "test")],
            chat_options: ChatOptions::default(),
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            decode_options: TextDecodeOptions::default(),
        }
    }

    /// Validates basic request invariants before rendering.
    pub fn validate(&self) -> Result<()> {
        if self.messages.is_empty() {
            return Err(Error::EmptyMessages);
        }
        match (
            self.chat_options.generation_prompt_mode,
            self.messages.last().map(ChatMessage::role),
        ) {
            (GenerationPromptMode::ContinueFinalAssistant, Some(ChatRole::Assistant)) => {}
            (GenerationPromptMode::ContinueFinalAssistant, _) => {
                return Err(Error::ContinueFinalAssistantWithoutFinalAssistant);
            }
            (GenerationPromptMode::NoGenerationPrompt, _)
            | (GenerationPromptMode::StartNewAssistant, _) => {}
        }
        Ok(())
    }

    /// Returns true if this request contains any multimodal content in its
    /// messages.
    pub fn has_multimodal(&self) -> bool {
        self.messages.iter().any(ChatMessage::has_multimodal)
    }

    /// Returns true if this request should enable tool parsing based on the tool
    /// choice and tool list.
    pub fn tool_parsing_enabled(&self) -> bool {
        matches!(self.tool_choice, ChatToolChoice::Auto) && !self.tools.is_empty()
    }
}

impl ChatRole {
    /// Returns the chat-template role string used by the current text-only chat
    /// backend.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::ToolResponse => "tool_response",
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, to_value};

    use super::{ChatContent, ChatContentPart, ChatMessage, ChatRole, Tool};
    use crate::serving::chat::event::AssistantContentBlock;

    #[test]
    fn chat_content_deserializes_from_raw_string() {
        let content: ChatContent = serde_json::from_value(json!("hello")).unwrap();
        assert_eq!(content, ChatContent::Text("hello".to_string()));
    }

    #[test]
    fn chat_content_deserializes_from_openai_text_blocks() {
        let content: ChatContent =
            serde_json::from_value(json!([{ "type": "text", "text": "hello" }])).unwrap();
        assert_eq!(
            content,
            ChatContent::Parts(vec![ChatContentPart::text("hello")])
        );
    }

    #[test]
    fn chat_content_from_string_like_values_builds_text() {
        assert_eq!(
            ChatContent::from("hello"),
            ChatContent::Text("hello".to_string())
        );
        assert_eq!(
            ChatContent::from("hello".to_string()),
            ChatContent::Text("hello".to_string())
        );
    }

    #[test]
    fn chat_content_try_flattens_text_parts_without_separators() {
        let content = ChatContent::Parts(vec![
            ChatContentPart::text("hello"),
            ChatContentPart::text(" world"),
        ]);
        assert_eq!(content.try_flatten_to_text().unwrap(), "hello world");
    }

    #[test]
    fn assistant_message_collects_visible_and_reasoning_text() {
        let message = ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::Reasoning {
                text: "inner".to_string(),
            },
            AssistantContentBlock::Text {
                text: "outer".to_string(),
            },
        ]);

        assert_eq!(message.role(), ChatRole::Assistant);
        assert_eq!(message.text_content().unwrap(), "outer");
        assert_eq!(message.reasoning_content().as_deref(), Some("inner"));
    }

    #[test]
    fn developer_message_round_trips_through_serde() {
        let message = ChatMessage::developer(
            "hello",
            Some(vec![Tool {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }),
                strict: Some(true),
            }]),
        );

        let value = to_value(&message).unwrap();
        let decoded: ChatMessage = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, message);
    }
}
