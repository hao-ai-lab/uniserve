//! Shared text-generation support: incremental detokenization, decode helpers,
//! tokenizer/model-derived sampling hints, and max-token resolution.
//!
//! Under the S02 funnel this module is a decode + lowering-helper library. Model
//! tokenization is owned by [`crate::model::ResolvedModel`]; there is no separate
//! text backend tower, request class, or structured-output surface here.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{BTreeSet, HashMap};

use enum_as_inner::EnumAsInner;
use serde::{Deserialize, Serialize};

pub use error::{Error, Result};
pub use output::{
    CollectedTextOutput, DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs,
    DecodedTextEvent, DecodedTokenLogprob, FinishReason, Finished, StopReason, TextDecodeOptions,
    TextOutputStreamExt,
};

mod error;
pub mod output;
pub use uniserve_model_profile::tokenizer;

use futures::Stream;
use trait_set::trait_set;

trait_set! {
    /// Shared streamed decoded-text output type.
    pub trait TextOutputStream = Stream<Item = Result<DecodedTextEvent>> + Send + 'static;
}

/// One rendered chat prompt, kept as an enum for renderer compatibility.
///
/// The configured funnel only ever produces [`Prompt::Text`]; the pre-tokenized
/// escape hatch is retained solely so the shared renderer contract stays stable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, EnumAsInner)]
#[serde(untagged)]
pub enum Prompt {
    /// Untokenized prompt text that still needs tokenizer work.
    Text(String),
    /// Pre-tokenized prompt IDs.
    TokenIds(Vec<u32>),
}

impl Default for Prompt {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// User-facing sampling parameters accepted by the internal chat render input.
///
/// Every field is optional so that model and generation-config defaults apply
/// when the caller omits a value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub seed: Option<i64>,
    pub max_tokens: Option<u32>,
    pub min_tokens: Option<u32>,
    pub logprobs: Option<i32>,
    pub prompt_logprobs: Option<i32>,
    pub min_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub ignore_eos: bool,
    pub logit_bias: Option<HashMap<u32, f32>>,
    pub allowed_token_ids: Option<Vec<u32>>,
    pub bad_words: Option<Vec<String>>,
    pub logprob_token_ids: Option<Vec<u32>>,
    pub skip_reading_prefix_cache: Option<bool>,
    pub write_prefix_cache: Option<bool>,
}

#[allow(clippy::derivable_impls)]
impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            max_tokens: None,
            min_tokens: None,
            logprobs: None,
            prompt_logprobs: None,
            min_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            stop_token_ids: None,
            ignore_eos: false,
            logit_bias: None,
            allowed_token_ids: None,
            bad_words: None,
            logprob_token_ids: None,
            skip_reading_prefix_cache: None,
            write_prefix_cache: None,
        }
    }
}

/// Tokenizer/model-derived hints used to enrich sampling parameters before they
/// are lowered into an engine request.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingHints {
    pub primary_eos_token_id: Option<u32>,
    pub extra_eos_token_ids: BTreeSet<u32>,
    pub default_temperature: Option<f32>,
    pub default_top_p: Option<f32>,
    pub default_top_k: Option<u32>,
    pub default_min_p: Option<f32>,
    pub default_repetition_penalty: Option<f32>,
    pub default_max_tokens: Option<u32>,
    /// Model context window size (`max_position_embeddings`).
    pub max_model_len: Option<u32>,
}

/// Resolve the effective `max_tokens` for generation.
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
