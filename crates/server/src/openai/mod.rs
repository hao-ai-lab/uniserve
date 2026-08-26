//! OpenAI-compatible request validation and conversion.
//!
//! This crate owns schema-only validation, OpenAI-to-UniServe lowering, and
//! OpenAI response helper conversion. HTTP routes supply state and map
//! [`ApiError`] into transport responses.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub mod chat_completions;
pub mod error;
pub mod images;
pub mod logprobs;
mod types;
pub mod utils;
pub mod videos;

pub use error::{ApiError, Result, serve_error_to_api};
pub use types::*;
pub use utils::{ResolvedRequestContext, convert_logit_bias};
