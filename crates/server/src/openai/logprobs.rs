//! Conversion from decoded engine logprobs to OpenAI response values.
//!
//! Inputs are the text layer's decoded candidates (`crate::serving::text`),
//! where each position lists its candidate tokens in engine order and a
//! generated position lists the sampled token first. With
//! `return_tokens_as_token_ids`, every token string is rendered as
//! `token_id:<id>` instead of its decoded text.

use std::collections::HashMap;

use crate::serving::text::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
};
use itertools::Itertools as _;

use crate::openai::error::{ApiError, server_error};
use crate::openai::types::{ChatLogProbs, ChatLogProbsContent, TopLogProb};

/// Converts decoded prompt positions into token-to-logprob maps.
///
/// Index `i` of the result describes prompt position `i`. Entry `0` is `None`
/// because the first prompt token has no preceding context to be scored
/// against; entry `i` comes from `scored_positions[i - 1]`. Candidates whose
/// rendered token strings are equal share one map key, and the later
/// candidate's logprob is kept.
pub fn decoded_prompt_logprobs_to_maps(
    prompt_logprobs: &DecodedPromptLogprobs,
    return_tokens_as_token_ids: bool,
) -> Vec<Option<HashMap<String, f32>>> {
    std::iter::once(None)
        .chain(prompt_logprobs.scored_positions.iter().map(|position| {
            Some(position_top_logprobs_map(
                position,
                return_tokens_as_token_ids,
            ))
        }))
        .collect()
}

/// Converts decoded generated-token candidates into chat logprob content.
///
/// # Errors
///
/// Returns a server error when a position has no candidates, since the
/// sampled token is read from the first candidate.
pub fn decoded_logprobs_to_openai_chat(
    logprobs: &DecodedLogprobs,
    return_tokens_as_token_ids: bool,
) -> Result<ChatLogProbs, ApiError> {
    let content = logprobs
        .positions
        .iter()
        .map(|position| position_to_chat_logprobs_content(position, return_tokens_as_token_ids))
        .try_collect()?;
    Ok(ChatLogProbs {
        content: Some(content),
    })
}

/// Formats a token for an OpenAI log-probability response.
fn format_token(entry: &DecodedTokenLogprob, as_token_id: bool) -> String {
    if as_token_id {
        format!("token_id:{}", entry.token_id)
    } else {
        entry.token.clone()
    }
}

/// Returns the top-log-probability map for one token position.
fn position_top_logprobs_map(
    position: &DecodedPositionLogprobs,
    return_tokens_as_token_ids: bool,
) -> HashMap<String, f32> {
    position
        .entries
        .iter()
        .map(|entry| {
            (
                format_token(entry, return_tokens_as_token_ids),
                clamp_logprob(entry.logprob),
            )
        })
        .collect()
}

/// Converts one decoded candidate distribution into OpenAI chat log-probability content.
///
/// The first candidate is the sampled token. `top_logprobs` lists every
/// candidate at the position, including the sampled token. `bytes` holds the
/// UTF-8 encoding of the rendered token string, which is the `token_id:<id>`
/// placeholder when `return_tokens_as_token_ids` is set.
fn position_to_chat_logprobs_content(
    position: &DecodedPositionLogprobs,
    return_tokens_as_token_ids: bool,
) -> Result<ChatLogProbsContent, ApiError> {
    let chosen = position.entries.first().ok_or_else(|| {
        server_error!("decoded chat logprobs position unexpectedly had no token candidates")
    })?;
    let token = format_token(chosen, return_tokens_as_token_ids);
    Ok(ChatLogProbsContent {
        bytes: Some(token.as_bytes().to_vec()),
        token,
        logprob: clamp_logprob(chosen.logprob),
        top_logprobs: position
            .entries
            .iter()
            .map(|entry| {
                let token = format_token(entry, return_tokens_as_token_ids);
                TopLogProb {
                    bytes: Some(token.as_bytes().to_vec()),
                    token,
                    logprob: clamp_logprob(entry.logprob),
                }
            })
            .collect(),
    })
}

/// Floors a log probability at `-9999.0`, so `-inf` (and NaN, which
/// `f32::max` discards) reaches the response as a finite number.
fn clamp_logprob(logprob: f32) -> f32 {
    logprob.max(-9999.0)
}
