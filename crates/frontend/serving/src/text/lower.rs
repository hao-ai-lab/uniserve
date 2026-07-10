use std::collections::{BTreeMap, BTreeSet};

use uniserve_core::{
    ContextSegment, GenerationBehaviorDescriptor, GenerationCachePolicyDescriptor,
    GenerationConstraint, GenerationPolicyDescriptor, GenerationRequest, GenerationResourceBounds,
    ImageParams, RequestId, SamplingParams as CoreSamplingParams, UndVisibility,
};
use uniserve_engine_gateway::generation::GenerationSubmission;
use uniserve_model_profile::tokenizer::Tokenizer;

use crate::text::backend::SamplingHints;
use crate::text::error::{Error, Result};
use crate::text::request::{SamplingParams, TextRequest};
use crate::text::structured_output::compile_structured_output;

/// One text request after canonical generation lowering.
#[derive(Debug)]
pub(crate) struct PreparedTextRequest {
    /// The original high-level request, preserved for response-side metadata
    /// and decoding options.
    pub text_request: TextRequest,
    /// Canonical generation request and typed transport metadata.
    pub submission: GenerationSubmission,
}

/// Convert a high-level [`TextRequest`] into one canonical generation submission.
pub(crate) fn lower_text_request(
    request: TextRequest,
    prompt_token_ids: Vec<u32>,
    sampling_hints: SamplingHints,
    tokenizer: &dyn Tokenizer,
) -> Result<PreparedTextRequest> {
    let prompt_len = prompt_token_ids.len() as u32;
    let lowered = lower_sampling_params(
        request.sampling_params.clone(),
        sampling_hints,
        prompt_len,
        tokenizer,
    )?;
    let constraint = GenerationConstraint::UndOnly;
    let mut policy = GenerationPolicyDescriptor::default();
    policy.termination.emit_stop_token = request.decode_options.include_stop_str_in_output;
    let cache = GenerationCachePolicyDescriptor {
        read: !lowered.skip_reading_prefix_cache && !lowered.sampling.prompt_logprobs_requested(),
        write: lowered.write_prefix_cache,
        isolation_key: request.cache_salt.as_deref().map(stable_hash),
    };
    let generation = GenerationRequest {
        request_id: RequestId(stable_hash(&request.request_id)),
        context: vec![ContextSegment::UndTokens {
            token_ids: prompt_token_ids,
            visibility: UndVisibility::Internal,
        }],
        negative_context: Vec::new(),
        constraint,
        behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
        sampling: lowered.sampling,
        image: ImageParams::default(),
        max_und_tokens: lowered.max_tokens as usize,
        stop_strings: request
            .decode_options
            .stop_strings
            .clone()
            .unwrap_or_default(),
        stop_token_ids: lowered.stop_token_ids,
        priority: request.priority,
        lora_id: match &request.adapter {
            crate::AdapterSelection::Base => None,
            crate::AdapterSelection::Adapter { internal_id, .. } => Some(
                u32::try_from(*internal_id).map_err(|_| Error::AdapterIdOutOfRange {
                    request_id: request.request_id.clone(),
                })?,
            ),
        },
        grammar: lowered.grammar,
        cache,
        policy,
        resources: GenerationResourceBounds {
            context_tokens: prompt_len as usize,
            max_kv_tokens: (prompt_len as usize).saturating_add(lowered.max_tokens as usize),
            ..GenerationResourceBounds::default()
        },
    };
    generation
        .validate()
        .map_err(|error| Error::InvalidGenerationRequest(error.to_string()))?;
    let mut submission = GenerationSubmission::new(request.request_id.clone(), generation);
    submission.data_parallel_rank = request.data_parallel_rank;
    submission.trace_headers = (!request.trace_context.is_empty()).then(|| {
        request
            .trace_context
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<BTreeMap<_, _>>()
    });

    Ok(PreparedTextRequest {
        text_request: request,
        submission,
    })
}

#[derive(Debug)]
pub(crate) struct LoweredSamplingParams {
    pub sampling: CoreSamplingParams,
    pub max_tokens: u32,
    pub stop_token_ids: Vec<u32>,
    pub grammar: Option<uniserve_core::GrammarSpec>,
    pub skip_reading_prefix_cache: bool,
    pub write_prefix_cache: bool,
}

/// Convert [`SamplingParams`] into canonical sampling and execution policy, enriching
/// omitted user values with tokenizer/model-derived hints when available.
pub(crate) fn lower_sampling_params(
    sampling_params: SamplingParams,
    SamplingHints {
        primary_eos_token_id,
        extra_eos_token_ids,
        default_temperature,
        default_top_p,
        default_top_k,
        default_min_p,
        default_repetition_penalty,
        default_max_tokens,
        max_model_len,
    }: SamplingHints,
    prompt_len: u32,
    tokenizer: &dyn Tokenizer,
) -> Result<LoweredSamplingParams> {
    let SamplingParams {
        temperature,
        top_p,
        top_k,
        seed,
        max_tokens,
        min_tokens,
        logprobs,
        prompt_logprobs,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        stop_token_ids,
        ignore_eos,
        logit_bias,
        allowed_token_ids,
        bad_words,
        logprob_token_ids,
        structured_outputs,
        skip_reading_prefix_cache,
        write_prefix_cache,
    } = sampling_params;

    // Typed request values take precedence over model generation defaults. The
    // neutral sampling values apply when neither source declares an override.
    let temperature = temperature.or(default_temperature).unwrap_or(1.0);
    let top_p = top_p.or(default_top_p).unwrap_or(1.0);
    let top_k = top_k.or(default_top_k).unwrap_or(0);
    let min_p = min_p.or(default_min_p).unwrap_or(0.0);
    let repetition_penalty = repetition_penalty
        .or(default_repetition_penalty)
        .unwrap_or(1.0);
    let max_tokens = resolve_max_tokens(max_tokens, default_max_tokens, max_model_len, prompt_len)?;
    let min_tokens = min_tokens.unwrap_or(0);
    let frequency_penalty = frequency_penalty.unwrap_or(0.0);
    let presence_penalty = presence_penalty.unwrap_or(0.0);

    let mut stop_token_ids = stop_token_ids.unwrap_or_default();
    let mut all_stop_token_ids = BTreeSet::from_iter(stop_token_ids.iter().copied());
    if let Some(primary_eos_token_id) = primary_eos_token_id {
        all_stop_token_ids.insert(primary_eos_token_id);
    }
    all_stop_token_ids.extend(extra_eos_token_ids.iter().copied());

    if !ignore_eos {
        merge_unique_token_ids(&mut stop_token_ids, extra_eos_token_ids.iter().copied());
    }
    let grammar = compile_structured_output(
        structured_outputs.as_ref(),
        tokenizer,
        &all_stop_token_ids.iter().copied().collect::<Vec<_>>(),
    )?;
    let bad_words_token_ids = tokenize_bad_words(bad_words.as_deref(), tokenizer)?;

    for (field, value) in [("logprobs", logprobs), ("prompt_logprobs", prompt_logprobs)] {
        if let Some(value) = value
            && value < -1
        {
            return Err(Error::InvalidLogprobCount { field, value });
        }
    }
    if min_tokens > max_tokens {
        return Err(Error::MinTokensExceedsMaximum {
            min_tokens,
            max_tokens,
        });
    }

    let mut canonical_logit_bias = logit_bias
        .as_ref()
        .map(|biases| biases.iter().map(|(&token, &bias)| (token, bias)).collect())
        .unwrap_or_else(Vec::new);
    canonical_logit_bias.sort_by_key(|(token, _)| *token);
    let sampling = CoreSamplingParams {
        temperature,
        top_k,
        top_p,
        ignore_eos,
        seed: seed.map(|value| value as u64),
        min_p,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        logit_bias: canonical_logit_bias,
        min_tokens: min_tokens as usize,
        return_logprobs: logprobs.is_some() || logprob_token_ids.is_some(),
        n_logprobs: match logprobs {
            Some(-1) => u32::MAX,
            Some(value) => value as u32,
            None => 0,
        },
        return_prompt_logprobs: prompt_logprobs.is_some(),
        n_prompt_logprobs: match prompt_logprobs {
            Some(-1) => u32::MAX,
            Some(value) => value as u32,
            None => 0,
        },
        logprob_token_ids: logprob_token_ids.clone().unwrap_or_default(),
        bad_words_ids: bad_words_token_ids.clone().unwrap_or_default(),
        allowed_token_ids: allowed_token_ids.clone(),
    };
    sampling.validate()?;

    Ok(LoweredSamplingParams {
        sampling,
        max_tokens,
        stop_token_ids,
        grammar,
        skip_reading_prefix_cache: skip_reading_prefix_cache.unwrap_or(false),
        write_prefix_cache: write_prefix_cache.unwrap_or(true),
    })
}

fn stable_hash(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}

/// Convert bad-word strings into token-ID sequences, following the reference
/// logic in `SamplingParams.update_from_tokenizer`.
///
/// Each word is encoded both with and without a leading space so that the ban
/// applies regardless of whether the word appears at the beginning or in the
/// middle of generated text (this accounts for tokenizers that use an
/// `add_prefix_space` convention).
///
fn tokenize_bad_words(
    bad_words: Option<&[String]>,
    tokenizer: &dyn Tokenizer,
) -> Result<Option<Vec<Vec<u32>>>> {
    let bad_words = bad_words.filter(|w| !w.is_empty());
    let mut all_token_ids = Vec::new();

    for bad_word in bad_words.into_iter().flatten() {
        // Without a leading space we always keep the encoding.
        // With a leading space we only keep it when the prefix-space variant produces a
        // distinct first token but the same sequence length *relative to this word's
        // no-space variant* — this mirrors the Python dedup condition that avoids
        // redundant entries. Comparing against this word's own `without_space` (rather
        // than `all_token_ids.last`) keeps the dedup correct even when `without_space`
        // was empty (and thus not pushed) or when the previous entry belongs to a
        // different bad word.
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

    Ok((!all_token_ids.is_empty()).then_some(all_token_ids))
}

/// Resolve the effective `max_tokens` for generation, mirroring the reference
/// `get_max_tokens`.
///
/// Takes the minimum of all available limits (user-specified, generation-config
/// default, and `max_model_len - prompt_len`). When nothing is known, falls
/// back to `u32::MAX` so the engine can apply its own context-window
/// limit.
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

fn merge_unique_token_ids(
    stop_token_ids: &mut Vec<u32>,
    extra_token_ids: impl Iterator<Item = u32>,
) {
    // Keep user-provided ordering stable while still folding in backend-derived EOS
    // aliases.
    for token_id in extra_token_ids {
        if !stop_token_ids.contains(&token_id) {
            stop_token_ids.push(token_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::text::backend::SamplingHints;
    use crate::text::request::{Prompt, TextRequest};

    /// Stub tokenizer that returns empty token IDs — sufficient for tests that
    /// don't exercise bad-words tokenization.
    struct StubTokenizer;

    impl Tokenizer for StubTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_special_tokens: bool,
        ) -> uniserve_model_profile::tokenizer::Result<Vec<u32>> {
            Ok(vec![])
        }

        fn decode(
            &self,
            _token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> uniserve_model_profile::tokenizer::Result<String> {
            Ok(String::new())
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }
    }

    fn stub_tokenizer() -> StubTokenizer {
        StubTokenizer
    }

    fn sample_request() -> TextRequest {
        TextRequest {
            prompt: Prompt::TokenIds(vec![1, 2, 3]),
            request_id: "text-1".to_string(),
            ..TextRequest::for_test()
        }
    }

    fn sample_sampling_hints() -> SamplingHints {
        SamplingHints {
            primary_eos_token_id: Some(99),
            extra_eos_token_ids: BTreeSet::from([77]),
            default_temperature: None,
            default_top_p: None,
            default_top_k: None,
            default_min_p: None,
            default_repetition_penalty: None,
            default_max_tokens: None,
            max_model_len: None,
        }
    }

    #[test]
    fn lower_text_request_applies_model_eos_hints() {
        let prepared = lower_text_request(
            sample_request(),
            vec![1, 2, 3],
            sample_sampling_hints(),
            &stub_tokenizer(),
        )
        .unwrap();

        let request = prepared.submission.request;
        assert_eq!(request.stop_token_ids, vec![77]);
        assert!(!request.sampling.ignore_eos);
        assert_eq!(request.max_und_tokens, u32::MAX as usize);
    }

    #[test]
    fn lower_text_request_does_not_capture_wall_clock_submission_time() {
        let prepared = lower_text_request(
            sample_request(),
            vec![1, 2, 3],
            sample_sampling_hints(),
            &stub_tokenizer(),
        )
        .unwrap();

        assert_eq!(prepared.submission.arrival_time, None);
    }

    #[test]
    fn lower_text_request_respects_ignore_eos_for_stop_token_ids() {
        let mut request = sample_request();
        request.sampling_params.ignore_eos = true;

        let prepared = lower_text_request(
            request,
            vec![1, 2, 3],
            sample_sampling_hints(),
            &stub_tokenizer(),
        )
        .unwrap();

        let request = prepared.submission.request;
        assert!(request.sampling.ignore_eos);
        assert!(request.stop_token_ids.is_empty());
    }

    #[test]
    fn lower_sampling_params_preserves_explicit_stop_token_ids_in_all_stop_set() {
        let sampling_params = SamplingParams {
            stop_token_ids: Some(vec![11, 77]),
            ..SamplingParams::default()
        };

        let params = lower_sampling_params(
            sampling_params,
            SamplingHints {
                primary_eos_token_id: Some(99),
                extra_eos_token_ids: BTreeSet::from([77, 88]),
                default_temperature: None,
                default_top_p: None,
                default_top_k: None,
                default_min_p: None,
                default_repetition_penalty: None,
                default_max_tokens: None,
                max_model_len: None,
            },
            3,
            &stub_tokenizer(),
        )
        .unwrap();

        assert_eq!(params.stop_token_ids, vec![11, 77, 88]);
    }

    #[test]
    fn lower_sampling_params_prefers_user_values_over_generation_defaults() {
        let sampling_params = SamplingParams {
            temperature: Some(0.2),
            top_p: Some(0.3),
            top_k: Some(4),
            max_tokens: Some(32),
            min_tokens: Some(2),
            ..Default::default()
        };

        let params = lower_sampling_params(
            sampling_params,
            SamplingHints {
                primary_eos_token_id: None,
                extra_eos_token_ids: BTreeSet::new(),
                default_temperature: Some(0.8),
                default_top_p: Some(0.9),
                default_top_k: Some(12),
                default_min_p: Some(0.1),
                default_repetition_penalty: Some(1.2),
                default_max_tokens: Some(128),
                max_model_len: None,
            },
            3,
            &stub_tokenizer(),
        )
        .unwrap();

        assert_eq!(params.max_tokens, 32);
        assert_eq!(params.sampling.temperature, 0.2);
        assert_eq!(params.sampling.top_p, 0.3);
        assert_eq!(params.sampling.top_k, 4);
        assert_eq!(params.sampling.min_tokens, 2);
        assert_eq!(params.sampling.min_p, 0.1);
        assert_eq!(params.sampling.repetition_penalty, 1.2);
    }

    #[test]
    fn lower_sampling_params_passes_logprobs_fields_through() {
        let sampling_params = SamplingParams {
            logprobs: Some(3),
            prompt_logprobs: Some(-1),
            ..Default::default()
        };

        let params = lower_sampling_params(
            sampling_params,
            SamplingHints {
                primary_eos_token_id: None,
                extra_eos_token_ids: BTreeSet::new(),
                default_temperature: None,
                default_top_p: None,
                default_top_k: None,
                default_min_p: None,
                default_repetition_penalty: None,
                default_max_tokens: None,
                max_model_len: None,
            },
            3,
            &stub_tokenizer(),
        )
        .unwrap();

        assert!(params.sampling.return_logprobs);
        assert_eq!(params.sampling.n_logprobs, 3);
        assert!(params.sampling.return_prompt_logprobs);
        assert_eq!(params.sampling.n_prompt_logprobs, u32::MAX);
    }

    #[test]
    fn lower_sampling_params_uses_generation_defaults_when_user_omits_values() {
        let params = lower_sampling_params(
            SamplingParams::default(),
            SamplingHints {
                primary_eos_token_id: None,
                extra_eos_token_ids: BTreeSet::new(),
                default_temperature: Some(0.8),
                default_top_p: Some(0.9),
                default_top_k: Some(12),
                default_min_p: Some(0.1),
                default_repetition_penalty: Some(1.2),
                default_max_tokens: Some(128),
                max_model_len: None,
            },
            3,
            &stub_tokenizer(),
        )
        .unwrap();

        assert_eq!(params.max_tokens, 128);
        assert_eq!(params.sampling.temperature, 0.8);
        assert_eq!(params.sampling.top_p, 0.9);
        assert_eq!(params.sampling.top_k, 12);
        assert_eq!(params.sampling.min_p, 0.1);
        assert_eq!(params.sampling.repetition_penalty, 1.2);
    }

    /// Tokenizer that maps exact input strings to canned token-id sequences,
    /// used to exercise the bad-words prefix-space dedup logic.
    struct MapTokenizer {
        map: std::collections::HashMap<String, Vec<u32>>,
    }

    impl Tokenizer for MapTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> uniserve_model_profile::tokenizer::Result<Vec<u32>> {
            Ok(self.map.get(text).cloned().unwrap_or_default())
        }

        fn decode(
            &self,
            _token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> uniserve_model_profile::tokenizer::Result<String> {
            Ok(String::new())
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }
    }

    #[test]
    fn tokenize_bad_words_dedup_uses_same_word_no_space_variant() {
        // Two bad words. For "alpha" the prefix-space variant differs in its first
        // token but has the same length, so it must be kept. For "beta" the
        // prefix-space variant matches the no-space variant's first token, so it
        // must be dropped. The dropped "beta" prefix variant must NOT be compared
        // against the prior-pushed "alpha" entry.
        let mut map = std::collections::HashMap::new();
        map.insert("alpha".to_string(), vec![10, 11]);
        map.insert(" alpha".to_string(), vec![20, 11]);
        map.insert("beta".to_string(), vec![30, 31]);
        map.insert(" beta".to_string(), vec![30, 31]);
        let tokenizer = MapTokenizer { map };

        let words = vec!["alpha".to_string(), "beta".to_string()];
        let result = tokenize_bad_words(Some(&words), &tokenizer).unwrap();

        assert_eq!(result, Some(vec![vec![10, 11], vec![20, 11], vec![30, 31]]));
    }

    #[test]
    fn tokenize_bad_words_keeps_prefix_variant_when_no_space_variant_empty() {
        // If the no-space variant tokenizes to nothing, the dedup must not compare
        // against an unrelated previous word's entry (the original bug, where
        // `all_token_ids.last` pointed at "alpha"). The prefix-space variant is
        // kept on its own merits.
        let mut map = std::collections::HashMap::new();
        map.insert("alpha".to_string(), vec![10, 11]);
        map.insert(" alpha".to_string(), vec![20, 11]);
        // "gamma" has no no-space entry (maps to empty); its prefix variant happens
        // to equal alpha's prefix entry [20, 11]. The OLD code compared against
        // `all_token_ids.last` == [20, 11] and (20 == 20) DROPPED it. The correct
        // behaviour keeps it, because there is no same-word no-space variant for it
        // to be redundant against.
        map.insert(" gamma".to_string(), vec![20, 11]);
        let tokenizer = MapTokenizer { map };

        let words = vec!["alpha".to_string(), "gamma".to_string()];
        let result = tokenize_bad_words(Some(&words), &tokenizer).unwrap();

        assert_eq!(result, Some(vec![vec![10, 11], vec![20, 11], vec![20, 11]]));
    }

    #[test]
    fn resolve_max_tokens_caps_by_model_len() {
        let result = resolve_max_tokens(Some(150), None, Some(200), 100);
        assert_eq!(result.unwrap(), 100);
    }

    #[test]
    fn lower_text_request_preserves_non_streaming_request_metadata() {
        let mut request = sample_request();
        request.intermediate = false;

        let prepared = lower_text_request(
            request,
            vec![1, 2, 3],
            sample_sampling_hints(),
            &stub_tokenizer(),
        )
        .unwrap();

        assert!(!prepared.text_request.intermediate);
        assert_eq!(prepared.submission.external_request_id, "text-1");
    }

    #[test]
    fn resolve_max_tokens_user_smaller_than_model_limit() {
        let result = resolve_max_tokens(Some(50), None, Some(200), 100);
        assert_eq!(result.unwrap(), 50);
    }

    #[test]
    fn resolve_max_tokens_uses_default_when_user_omits() {
        let result = resolve_max_tokens(None, Some(64), Some(200), 100);
        assert_eq!(result.unwrap(), 64);
    }

    #[test]
    fn resolve_max_tokens_default_capped_by_model_len() {
        let result = resolve_max_tokens(None, Some(256), Some(200), 100);
        assert_eq!(result.unwrap(), 100);
    }

    #[test]
    fn resolve_max_tokens_no_model_len_falls_back() {
        let result = resolve_max_tokens(Some(9999), None, None, 100);
        assert_eq!(result.unwrap(), 9999);
    }

    #[test]
    fn resolve_max_tokens_no_limits_known_falls_back_to_u32_max() {
        let result = resolve_max_tokens(None, None, None, 100);
        assert_eq!(result.unwrap(), u32::MAX);
    }

    #[test]
    fn resolve_max_tokens_prompt_too_long() {
        let result = resolve_max_tokens(Some(10), None, Some(100), 100);
        assert!(matches!(
            result,
            Err(Error::PromptTooLong {
                max_model_len: 100,
                prompt_len: 100,
            })
        ));
    }

    #[test]
    fn resolve_max_tokens_prompt_exceeds_model_len() {
        let result = resolve_max_tokens(Some(10), None, Some(100), 200);
        assert!(matches!(
            result,
            Err(Error::PromptTooLong {
                max_model_len: 100,
                prompt_len: 200,
            })
        ));
    }
}
