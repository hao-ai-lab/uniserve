//! Tool-parser construction and streaming parse errors.

use thiserror::Error;
use thiserror_ext::Macro;

/// Result alias for tool parser operations.
pub type Result<T> = std::result::Result<T, ToolParserError>;

/// Errors produced while creating or running tool parsers.
#[derive(Debug, Error, Macro)]
#[thiserror_ext(macro(path = "crate::profile::tools::error"))]
pub enum ToolParserError {
    /// Assistant text cannot be interpreted under the selected tool-call grammar.
    #[error("tool parser parsing failed: {message}")]
    ParsingFailed {
        /// Parser-specific failure description.
        message: String,
    },
}
