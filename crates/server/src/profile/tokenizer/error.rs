//! Tokenizer loading, encoding, and decoding errors.

use thiserror::Error;
use thiserror_ext::Macro;

/// Result type returned by tokenizer calls.
pub type Result<T> = std::result::Result<T, TokenizerError>;

#[derive(Debug, Error, Macro)]
#[thiserror_ext(macro(path = "crate::profile::tokenizer::error"))]
#[error("tokenizer error: {0}")]
/// Error returned while loading, encoding, or decoding tokens.
pub struct TokenizerError(#[message] pub String);
