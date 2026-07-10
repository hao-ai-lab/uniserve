use crate::openai::utils::{
    check_model_served, check_prompt_logprobs_bound, check_stream_options_requires_stream,
};

use super::types::GenerateRequest;
use crate::openai::ApiError;

/// Enforce the minimal compatibility contract for the Rust token generate
/// route.
///
/// The transport-agnostic compatibility rules (model-name membership,
/// `stream_options` requires `stream=true`, and the `prompt_logprobs` bound)
/// are shared with the OpenAI chat/completions validators via
/// [`uniserve_protocol_adapters::openai::utils`]; those helpers return the canonical OpenAI
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
        return Err(ApiError::invalid_request(
            "token_ids must contain at least one token ID.",
            Some("token_ids"),
        ));
    }

    if request.sampling_params.max_tokens == Some(0) {
        return Err(ApiError::invalid_request(
            "max_tokens must be greater than 0.",
            Some("sampling_params"),
        ));
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
    use crate::raw_generate::GenerateRequest;

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

    #[test]
    fn validate_request_compat_rejects_zero_max_tokens() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {"max_tokens": 0}
        }))
        .expect("parse request");
        let error = validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"]))
            .expect_err("max_tokens == 0 must be rejected");
        assert_eq!(
            error.to_error_response().error.param.as_deref(),
            Some("sampling_params")
        );
    }

    #[test]
    fn validate_request_compat_accepts_one_max_tokens() {
        // Boundary just above the rejected zero value.
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {"max_tokens": 1}
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok());
    }

    #[test]
    fn validate_request_compat_rejects_out_of_range_prompt_logprobs() {
        // Any negative value other than the -1 full-vocabulary sentinel is invalid.
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {"prompt_logprobs": -2}
        }))
        .expect("parse request");
        let error = validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"]))
            .expect_err("prompt_logprobs == -2 must be rejected");
        assert_eq!(
            error.to_error_response().error.param.as_deref(),
            Some("sampling_params")
        );
    }

    #[test]
    fn validate_request_compat_accepts_minus_one_prompt_logprobs() {
        // -1 is the full-vocabulary sentinel and sits at the boundary of the valid range.
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {"prompt_logprobs": -1}
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok());
    }
}
