use std::collections::HashMap;

use serde_json::Value;

use crate::error::ApiError;

// ---- Transport-agnostic compatibility predicates ----

// These encode the shared OpenAI compatibility rules that apply identically to
// every request surface (chat/completions, text/completions, and the native
// /generate route). They are written against scalar inputs rather than a
// concrete request struct so that callers with different request shapes
// (`model: String` vs `Option<String>`, `prompt_logprobs` nested under
// `sampling_params`, etc.) can route through one definition. They return the
// canonical [`ApiError`]; the native HTTP layer converts via its existing
// `From<uniserve_openai_api::ApiError>` impl.

/// Reject a request whose model name is not among the served model names.

/// `model` is the resolved model name the caller extracted from its own
/// request shape.
pub fn check_model_served(model: &str, served_model_names: &[String]) -> Result<(), ApiError> {
    if !served_model_names.iter().any(|n| n == model) {
        return Err(ApiError::model_not_found(model.to_string()));
    }
    Ok(())
}

/// Reject `stream_options` supplied without `stream=true`.
pub fn check_stream_options_requires_stream(
    stream_options_present: bool,
    stream: bool,
) -> Result<(), ApiError> {
    if stream_options_present && !stream {
        return Err(ApiError::invalid_request(
            "stream_options are only supported when stream=true.",
            Some("stream_options"),
        ));
    }
    Ok(())
}

/// Reject a `prompt_logprobs` value outside the supported range.

/// Valid values are any non-negative integer or the sentinel `-1` (full
/// vocabulary). `param` lets each surface attribute the error to its own field
/// (e.g. `prompt_logprobs` for OpenAI bodies, `sampling_params` for /generate).
pub fn check_prompt_logprobs_bound(
    prompt_logprobs: i32,
    param: &'static str,
) -> Result<(), ApiError> {
    if prompt_logprobs < 0 && prompt_logprobs != -1 {
        return Err(ApiError::invalid_request(
            "prompt_logprobs must be a non-negative value or -1.",
            Some(param),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResolvedRequestContext {
    pub request_id: String,
    pub data_parallel_rank: Option<u32>,
}

/// Merge `kv_transfer_params` into the engine extra-argument map.
pub fn merge_kv_transfer_params(
    mut xargs: Option<HashMap<String, Value>>,
    kv_transfer_params: Option<&HashMap<String, Value>>,
) -> Option<HashMap<String, Value>> {
    if let Some(kv_params) = kv_transfer_params {
        let map = xargs.get_or_insert_with(HashMap::new);
        map.insert(
            "kv_transfer_params".to_string(),
            Value::Object(
                kv_params
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            ),
        );
    }
    xargs
}

/// Convert OpenAI-style `logit_bias` with string token-ID keys into the
/// internal token-ID map.
pub fn convert_logit_bias(
    logit_bias: Option<HashMap<String, f32>>,
) -> Result<Option<HashMap<u32, f32>>, ApiError> {
    logit_bias
        .map(|bias| {
            bias.into_iter()
                .map(|(key, value)| {
                    key.parse().map(|k| (k, value)).map_err(|_| {
                        ApiError::invalid_request(
                            format!(
                                "Invalid key in 'logit_bias': '{key}' is not a valid token ID. \
                                 Token IDs must be non-negative integers."
                            ),
                            Some("logit_bias"),
                        )
                    })
                })
                .collect()
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn served(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn check_model_served_accepts_known_and_rejects_unknown() {
        assert!(check_model_served("m", &served(&["other", "m"])).is_ok());
        match check_model_served("m", &served(&["other"])) {
            Err(ApiError::ModelNotFound { model }) => assert_eq!(model, "m"),
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn check_stream_options_requires_stream_rejects_only_when_not_streaming() {
        assert!(check_stream_options_requires_stream(true, true).is_ok());
        assert!(check_stream_options_requires_stream(false, false).is_ok());
        match check_stream_options_requires_stream(true, false) {
            Err(ApiError::InvalidRequest { param, .. }) => {
                assert_eq!(param, Some("stream_options"))
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn check_prompt_logprobs_bound_accepts_non_negative_and_sentinel() {
        assert!(check_prompt_logprobs_bound(0, "prompt_logprobs").is_ok());
        assert!(check_prompt_logprobs_bound(5, "prompt_logprobs").is_ok());
        assert!(check_prompt_logprobs_bound(-1, "prompt_logprobs").is_ok());
    }

    #[test]
    fn check_prompt_logprobs_bound_rejects_other_negatives_with_caller_param() {
        match check_prompt_logprobs_bound(-2, "sampling_params") {
            Err(ApiError::InvalidRequest { param, .. }) => {
                assert_eq!(param, Some("sampling_params"))
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }
}
