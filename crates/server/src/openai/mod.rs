//! OpenAI-compatible wire schemas, validation, errors, and response helpers.
//!
//! HTTP routes provide configuration state and convert [`ApiError`] values into
//! transport responses; this module remains independent of routing.
//!
//! A chat request is deserialized into the strict schemas in `types`
//! (unknown fields are rejected), normalized and checked by the route's
//! `ValidatedJson` extractor, and checked against the served configuration by
//! `chat_completions::validate_request_compat`. Lowering into serving inputs
//! lives in `crate::serving` (`InputProcessor::preprocess_chat_request`); the
//! resulting `RequestOutput` stream comes back here for response assembly.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
/// Chat-completion validation and response assembly.
pub mod chat_completions;
/// OpenAI-compatible API errors.
pub mod error;
/// Collection of image-generation output into the OpenAI response.
pub mod images;
/// Token log-probability response values.
pub mod logprobs;
mod types;
/// Validation, conversion, and usage-accounting helpers shared by request
/// families.
pub mod utils;

pub use error::{ApiError, Result, serve_error_to_api};
pub use types::*;
pub use utils::convert_logit_bias;
