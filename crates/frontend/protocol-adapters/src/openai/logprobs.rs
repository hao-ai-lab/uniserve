use std::collections::HashMap;

use itertools::Itertools as _;
use uniserve_serving::text::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
};

use crate::openai::error::{ApiError, server_error};
use crate::openai::types::{ChatLogProbs, ChatLogProbsContent, TopLogProb};

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

fn format_token(entry: &DecodedTokenLogprob, as_token_id: bool) -> String {
    if as_token_id {
        format!("token_id:{}", entry.token_id)
    } else {
        entry.token.clone()
    }
}

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

fn clamp_logprob(logprob: f32) -> f32 {
    logprob.max(-9999.0)
}
