//! OpenAI-compatible schemas, validation, lowering, and response helpers.
//!
//! HTTP routes provide configuration state and convert [`ApiError`] values into
//! transport responses; this module remains independent of routing.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
/// Chat-completion request lowering and response assembly.
pub mod chat_completions;
/// OpenAI-compatible API errors.
pub mod error;
/// Image-generation request and response schemas.
pub mod images;
/// Token log-probability response values.
pub mod logprobs;
mod types;
/// Shared request conversion helpers.
pub mod utils;

pub use error::{ApiError, Result, serve_error_to_api};
pub use types::*;
pub use utils::convert_logit_bias;
