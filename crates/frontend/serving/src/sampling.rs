use std::collections::{BTreeMap, BTreeSet};

use uniserve_core::SamplingParams;
use uniserve_model_profile::tokenizer::HuggingFaceTokenizer;

use crate::input::{GenerateReqInput, SamplingConfig, StopConfig};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SamplingFallbacks {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SamplingDefaults {
    pub additional_eos_token_ids: BTreeSet<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub max_tokens: Option<u32>,
    pub max_model_tokens: u32,
    pub fallbacks: SamplingFallbacks,
}

pub(crate) struct LoweredSampling {
    pub params: SamplingParams,
    pub stop_token_ids: Vec<u32>,
}

pub(crate) fn lower_sampling(
    tokenizer: &HuggingFaceTokenizer,
    request: &GenerateReqInput,
    defaults: &SamplingDefaults,
) -> Result<LoweredSampling, String> {
    let SamplingConfig {
        temperature,
        top_p,
        top_k,
        min_p,
        seed,
        min_tokens,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        ignore_eos,
        ..
    } = &request.sampling;
    let StopConfig {
        stop_token_ids,
        bad_words,
        allowed_token_ids,
        logit_bias,
        logprobs,
        prompt_logprobs,
        logprob_token_ids,
        ..
    } = &request.stop;

    let seed = seed
        .map(|value| {
            u64::try_from(value).map_err(|_| "seed must be a non-negative integer".to_string())
        })
        .transpose()?;
    let n_logprobs = logprob_count("logprobs", *logprobs)?;
    let n_prompt_logprobs = logprob_count("prompt_logprobs", *prompt_logprobs)?;
    let params = SamplingParams {
        temperature: temperature
            .or(defaults.temperature)
            .unwrap_or(defaults.fallbacks.temperature),
        top_k: top_k.or(defaults.top_k).unwrap_or(defaults.fallbacks.top_k),
        top_p: top_p.or(defaults.top_p).unwrap_or(defaults.fallbacks.top_p),
        ignore_eos: *ignore_eos,
        seed,
        min_p: min_p.or(defaults.min_p).unwrap_or(0.0),
        repetition_penalty: repetition_penalty
            .or(defaults.repetition_penalty)
            .unwrap_or(1.0),
        frequency_penalty: frequency_penalty.unwrap_or(0.0),
        presence_penalty: presence_penalty.unwrap_or(0.0),
        logit_bias: canonical_logit_bias(logit_bias),
        min_tokens: min_tokens.unwrap_or(0) as usize,
        return_logprobs: logprobs.is_some() || logprob_token_ids.is_some(),
        n_logprobs,
        return_prompt_logprobs: prompt_logprobs.is_some(),
        n_prompt_logprobs,
        logprob_token_ids: canonical_ids(logprob_token_ids.as_deref().unwrap_or_default()),
        bad_words_ids: tokenize_bad_words(bad_words, tokenizer)?,
        allowed_token_ids: allowed_token_ids.as_deref().map(canonical_ids),
        typical_p: 1.0,
        forced_token_ids: Vec::new(),
    };
    params.validate().map_err(|error| error.to_string())?;

    let mut stops = stop_token_ids.iter().copied().collect::<BTreeSet<_>>();
    if !params.ignore_eos {
        stops.extend(defaults.additional_eos_token_ids.iter().copied());
    }
    Ok(LoweredSampling {
        params,
        stop_token_ids: stops.into_iter().collect(),
    })
}

fn logprob_count(field: &'static str, value: Option<i32>) -> Result<u32, String> {
    match value {
        Some(-1) => Ok(u32::MAX),
        Some(value) if value >= 0 => Ok(value as u32),
        Some(value) => Err(format!("{field} must be non-negative or -1, got {value}")),
        None => Ok(0),
    }
}

fn canonical_logit_bias(value: &Option<std::collections::HashMap<u32, f32>>) -> Vec<(u32, f32)> {
    value
        .as_ref()
        .map(|biases| {
            biases
                .iter()
                .map(|(&token_id, &bias)| (token_id, bias))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect()
        })
        .unwrap_or_default()
}

fn canonical_ids(values: &[u32]) -> Vec<u32> {
    values
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn tokenize_bad_words(
    bad_words: &[String],
    tokenizer: &HuggingFaceTokenizer,
) -> Result<Vec<Vec<u32>>, String> {
    let mut token_sequences = BTreeSet::new();
    for bad_word in bad_words {
        let without_space = tokenizer
            .encode(bad_word, false)
            .map_err(|error| error.to_string())?;
        let with_space = tokenizer
            .encode(&format!(" {}", bad_word.trim_start()), false)
            .map_err(|error| error.to_string())?;
        let keep_with_space = !with_space.is_empty()
            && (without_space.is_empty()
                || (with_space[0] != without_space[0] && with_space.len() == without_space.len()));
        if !without_space.is_empty() {
            token_sequences.insert(without_space);
        }
        if keep_with_space {
            token_sequences.insert(with_space);
        }
    }
    Ok(token_sequences.into_iter().collect())
}
