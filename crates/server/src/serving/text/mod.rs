//! Text tokenization, decoding, sampling lowering, and output helpers.
//!
//! Model profiles own request tokenization; this module provides reusable text
//! primitives for profiles and response assemblers.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub use error::{Error, Result};
pub use output::{
    CollectedTextOutput, DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs,
    DecodedTextEvent, DecodedTokenLogprob, FinishReason, Finished, StopReason, TextDecodeOptions,
};

mod error;
/// Incremental decoded-text and log-probability output values.
pub mod output;
pub use crate::profile::tokenizer;

/// Resolves the effective `max_tokens` for generation.
///
/// Takes the minimum of all available limits (user, generation-config default,
/// and `max_model_len - prompt_len`), falling back to `u32::MAX` when nothing is
/// known so the engine can apply its own context-window limit.
pub fn resolve_max_tokens(
    user_max_tokens: Option<u32>,
    default_max_tokens: Option<u32>,
    max_model_len: Option<u32>,
    prompt_len: u32,
) -> Result<u32> {
    let model_max_tokens = match max_model_len {
        Some(max_model_len) if prompt_len >= max_model_len => {
            return Err(Error::PromptTooLong {
                max_model_len,
                prompt_len,
            });
        }
        Some(max_model_len) => Some(max_model_len - prompt_len),
        None => None,
    };

    let fallback_max_tokens = user_max_tokens.or(default_max_tokens);
    Ok([fallback_max_tokens, model_max_tokens]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_max_tokens_caps_by_model_len() {
        assert_eq!(
            resolve_max_tokens(Some(150), None, Some(200), 100).unwrap(),
            100
        );
    }

    #[test]
    fn resolve_max_tokens_uses_default_when_user_omits() {
        assert_eq!(
            resolve_max_tokens(None, Some(64), Some(200), 100).unwrap(),
            64
        );
    }

    #[test]
    fn resolve_max_tokens_no_limits_known_falls_back_to_u32_max() {
        assert_eq!(resolve_max_tokens(None, None, None, 100).unwrap(), u32::MAX);
    }

    #[test]
    fn resolve_max_tokens_prompt_too_long() {
        assert!(matches!(
            resolve_max_tokens(Some(10), None, Some(100), 100),
            Err(Error::PromptTooLong {
                max_model_len: 100,
                prompt_len: 100,
            })
        ));
    }
}
