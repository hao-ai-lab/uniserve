//! Validation and accounting helpers shared by OpenAI request families.

use std::collections::HashMap;

use crate::openai::error::ApiError;

/// Counts every generated token included in OpenAI completion usage, including tokens
/// retained internally for reasoning or model control.
///
/// The sum saturates at `u32::MAX` instead of overflowing.
pub(crate) fn completion_token_count(visible: u32, internal: u32) -> u32 {
    visible.saturating_add(internal)
}

/// Rejects a request whose model name is not the served model name.
///
/// `model` is the resolved model name the caller extracted from its own
/// request shape. The comparison is exact, and a mismatch is reported as
/// [`ApiError::ModelNotFound`] (HTTP 404).
pub fn check_model_served(model: &str, served_model_name: &str) -> Result<(), ApiError> {
    if model != served_model_name {
        return Err(ApiError::model_not_found(model.to_string()));
    }
    Ok(())
}

/// Rejects `stream_options` supplied without `stream=true`.
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

/// Rejects a `prompt_logprobs` value outside the supported range.
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

/// Converts OpenAI-style `logit_bias` with string token-ID keys into the
/// internal token-ID map.
///
/// JSON object keys are always strings, so each key is parsed as a decimal
/// `u32` token ID; any key that does not parse fails the whole conversion
/// with an invalid-request error on `logit_bias`. Bias values pass through
/// unchanged, and an absent map stays `None`. Token IDs are not checked
/// against the vocabulary here.
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
