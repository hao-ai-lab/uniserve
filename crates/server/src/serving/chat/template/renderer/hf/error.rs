//! Hugging Face chat-template loading and rendering errors.

use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
/// Error returned while loading, detecting, or rendering a chat template.
pub enum TemplateError {
    #[error("failed to render jinja template")]
    Jinja(#[from] minijinja::Error),
    #[error("failed to read chat template file")]
    ReadTemplateFile(#[source] std::io::Error),
    #[error("chat template looks like a file path but does not exist")]
    MissingTemplatePath,
    #[error("failed to parse chat_template.json")]
    ParseTemplateJson(#[source] serde_json::Error),
    #[error("chat_template.json does not contain a valid template")]
    InvalidTemplateJson,
}
