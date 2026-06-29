use uniserve_openai_api::utils::{
    check_model_served, check_prompt_logprobs_bound, check_stream_options_requires_stream,
};

use super::types::GenerateRequest;
use crate::error::{ApiError, bail_invalid_request};

/// Enforce the minimal compatibility contract for the Rust token generate
/// route.

/// The transport-agnostic compatibility rules (model-name membership,
/// `stream_options` requires `stream=true`, and the `prompt_logprobs` bound)
/// are shared with the OpenAI chat/completions validators via
/// [`uniserve_openai_api::utils`]; those helpers return the canonical OpenAI
/// `ApiError`, which `?` converts into this crate's `ApiError`. Only
/// request-shape-specific checks (token-ID non-emptiness) live locally.
pub(super) fn validate_request_compat(
    request: &GenerateRequest,
    served_model_names: &[String],
) -> Result<(), ApiError> {
    if let Some(model) = request.model.as_ref() {
        check_model_served(model, served_model_names)?;
    }

    check_stream_options_requires_stream(request.stream_options.is_some(), request.stream)?;

    if request.token_ids.is_empty() {
        bail_invalid_request!(
            param = "token_ids",
            "token_ids must contain at least one token ID."
        );
    }

    if request.sampling_params.max_tokens == Some(0) {
        bail_invalid_request!(
            param = "sampling_params",
            "max_tokens must be greater than 0."
        );
    }

    if let Some(prompt_logprobs) = request.sampling_params.prompt_logprobs {
        check_prompt_logprobs_bound(prompt_logprobs, "sampling_params")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::validate_request_compat;
    use crate::routes::inference::generate::types::GenerateRequest;

    fn base_request() -> GenerateRequest {
        serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {}
        }))
        .expect("parse request")
    }

    fn served(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn validate_request_compat_accepts_streaming() {
        let request = GenerateRequest {
            stream: true,
            ..base_request()
        };
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok());
    }

    #[test]
    fn validate_request_compat_rejects_stream_options_without_streaming() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": false,
            "stream_options": {"include_usage": true},
            "sampling_params": {}
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());
    }

    #[test]
    fn validate_request_compat_rejects_empty_token_ids() {
        let request = GenerateRequest {
            token_ids: Vec::new(),
            ..base_request()
        };
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());
    }
}
