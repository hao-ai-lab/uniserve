//! Shared normalization for token sampling; model callers supply numerical defaults.

use super::{SamplingConfig, StopConfig, TokenizeError};
use crate::profile::tokenizer::HuggingFaceTokenizer;
use uniserve_core::SamplingParams;

/// Applies public controls once, preserving model biases and an explicit image seed.
pub(super) fn apply_sampling(
    tokenizer: &HuggingFaceTokenizer,
    controls: &SamplingConfig,
    stop: &StopConfig,
    sampling: &mut SamplingParams,
) -> Result<(), TokenizeError> {
    sampling.temperature = controls.temperature.unwrap_or(sampling.temperature);
    sampling.top_k = controls.top_k.unwrap_or(sampling.top_k);
    sampling.top_p = controls.top_p.unwrap_or(sampling.top_p);
    sampling.min_p = controls.min_p.unwrap_or(sampling.min_p);
    sampling.repetition_penalty = controls
        .repetition_penalty
        .unwrap_or(sampling.repetition_penalty);
    sampling.ignore_eos = controls.ignore_eos;
    // Signed API seeds preserve their bit pattern for every model.
    sampling.seed = sampling.seed.or(controls.seed.map(|seed| seed as u64));
    sampling.min_tokens = controls.min_tokens.unwrap_or(0) as usize;
    sampling.frequency_penalty = controls.frequency_penalty.unwrap_or(0.0);
    sampling.presence_penalty = controls.presence_penalty.unwrap_or(0.0);
    let mut biases = sampling
        .logit_bias
        .drain(..)
        .collect::<std::collections::BTreeMap<_, _>>();
    if let Some(request) = &stop.logit_bias {
        for (&token, &bias) in request {
            *biases.entry(token).or_insert(0.0) += bias;
        }
    }
    sampling.logit_bias = biases.into_iter().collect();
    sampling.return_logprobs = stop.logprobs.is_some() || stop.logprob_token_ids.is_some();
    sampling.n_logprobs = logprob_count("logprobs", stop.logprobs)?;
    sampling.return_prompt_logprobs = stop.prompt_logprobs.is_some();
    sampling.n_prompt_logprobs = logprob_count("prompt_logprobs", stop.prompt_logprobs)?;
    sampling.logprob_token_ids = stop.logprob_token_ids.clone().unwrap_or_default();
    sampling.allowed_token_ids = stop.allowed_token_ids.clone();
    sampling.bad_words_ids = tokenize_bad_words(&stop.bad_words, tokenizer)?.unwrap_or_default();
    sampling.validate()?;
    Ok(())
}

fn logprob_count(field: &'static str, value: Option<i32>) -> Result<u32, TokenizeError> {
    match value {
        Some(-1) => Ok(u32::MAX),
        Some(value) if value < -1 => Err(TokenizeError::InvalidLogprobCount { field, value }),
        Some(value) => Ok(value as u32),
        None => Ok(0),
    }
}

/// Converts bad-word strings into token-ID sequences, encoding each word both
/// with and without a leading space (prefix-space convention) and deduping.
fn tokenize_bad_words(
    bad_words: &[String],
    tokenizer: &crate::profile::tokenizer::HuggingFaceTokenizer,
) -> std::result::Result<Option<Vec<Vec<u32>>>, crate::profile::tokenizer::TokenizerError> {
    if bad_words.is_empty() {
        return Ok(None);
    }
    let mut all_token_ids = Vec::new();
    for bad_word in bad_words {
        let without_space = tokenizer.encode(bad_word, false)?;
        let with_space = tokenizer.encode(&format!(" {}", bad_word.trim_start()), false)?;
        let keep_with_space = !with_space.is_empty()
            && (without_space.is_empty()
                || (with_space[0] != without_space[0] && with_space.len() == without_space.len()));
        if !without_space.is_empty() {
            all_token_ids.push(without_space);
        }
        if keep_with_space {
            all_token_ids.push(with_space);
        }
    }
    all_token_ids.sort();
    all_token_ids.dedup();
    Ok((!all_token_ids.is_empty()).then_some(all_token_ids))
}
