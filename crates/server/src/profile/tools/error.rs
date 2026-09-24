//! Tool-parser construction and streaming parse errors.

use thiserror::Error;
use thiserror_ext::Macro;

/// Result alias for tool parser calls.
pub type Result<T> = std::result::Result<T, ToolParserError>;

/// Errors produced while creating or running tool parsers.
///
/// The `Macro` derive generates the format-style `parsing_failed!`
/// constructor; `profile::tools` imports this module with `#[macro_use]` so its
/// submodules can call it.
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
