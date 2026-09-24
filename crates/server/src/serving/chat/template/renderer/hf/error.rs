//! Hugging Face chat-template loading and rendering errors.

use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
/// Error returned while loading, detecting, or rendering a chat template.
///
/// `HfChatRenderer` converts these into `chat::Error::ChatTemplate` with the
/// full error report as its message.
pub enum TemplateError {
    /// minijinja failed to compile or render the template.
    #[error("failed to render jinja template")]
    Jinja(#[from] minijinja::Error),
    /// A template file could not be read.
    #[error("failed to read chat template file")]
    ReadTemplateFile(#[source] std::io::Error),
    /// A configured template is neither an existing path nor recognizable as
    /// inline Jinja (it contains no `{`, `}`, or newline).
    #[error("chat template looks like a file path but does not exist")]
    MissingTemplatePath,
    /// A `.json` template file is not valid JSON.
    #[error("failed to parse chat_template.json")]
    ParseTemplateJson(#[source] serde_json::Error),
    /// A `.json` template file is neither a JSON string nor an object with a
    /// string `chat_template` field.
    #[error("chat_template.json does not contain a valid template")]
    InvalidTemplateJson,
}
