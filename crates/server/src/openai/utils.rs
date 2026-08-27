use std::collections::HashMap;

use crate::openai::error::ApiError;

/// OpenAI completion usage includes every generated token, including tokens
/// retained internally for reasoning or model control.
pub(crate) fn completion_token_count(visible: u32, internal: u32) -> u32 {
    visible.saturating_add(internal)
}

// ---- Transport-agnostic compatibility predicates ----

/// Reject a request whose model name is not among the served model names.
///
/// `model` is the resolved model name the caller extracted from its own
/// request shape.
pub fn check_model_served(model: &str, served_model_name: &str) -> Result<(), ApiError> {
    if model != served_model_name {
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
///
/// Valid values are any non-negative integer or the sentinel `-1` (full
/// vocabulary). `param` lets each surface attribute the error to its own field
/// (for example, `prompt_logprobs` in an OpenAI request body).
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
    pub trace_context: HashMap<String, String>,
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
