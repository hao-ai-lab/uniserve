use uniserve_serving::text::{Prompt, TextDecodeOptions, TextRequest};

use super::types::GenerateRequest;
use super::validate;
use crate::openai::{ApiError, LoraModelResolution, ResolvedRequestContext};

/// Lowered generate request plus the response request ID.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedRequest {
    pub request_id: String,
    pub text_request: TextRequest,
    pub stream: bool,
    pub include_usage: bool,
    pub include_continuous_usage: bool,
    pub include_logprobs: bool,
    pub include_prompt_logprobs: bool,
}

/// Validate and lower one raw generate request into the internal
/// text-generation format.
pub fn prepare_generate_request(
    request: GenerateRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
) -> Result<PreparedRequest, ApiError> {
    validate::validate_request_compat(&request, &lora_resolution.model_names)?;
    if request
        .kv_transfer_params
        .as_ref()
        .is_some_and(|value| !value.is_empty())
    {
        return Err(ApiError::InvalidRequest {
            message: "`kv_transfer_params` is not supported by this runtime.".to_string(),
            param: Some("kv_transfer_params"),
        });
    }
    if !request.other.is_empty() {
        let key = request.other.keys().next().cloned().unwrap_or_default();
        return Err(ApiError::InvalidRequest {
            message: format!("unsupported generate request field `{key}`"),
            param: None,
        });
    }

    let stream = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let include_continuous_usage = include_usage
        && request
            .stream_options
            .as_ref()
            .and_then(|options| options.continuous_usage_stats)
            .unwrap_or(false);
    let include_logprobs = request.sampling_params.logprobs.is_some();
    let include_prompt_logprobs = request.sampling_params.prompt_logprobs.is_some();
    let sampling_params = request.sampling_params;

    let text_request = TextRequest {
        request_id: ctx.request_id.clone(),
        prompt: Prompt::TokenIds(request.token_ids),
        sampling_params,
        decode_options: TextDecodeOptions::default(),
        intermediate: false,
        priority: request.priority,
        cache_salt: request.cache_salt,
        add_special_tokens: false,
        data_parallel_rank: ctx.data_parallel_rank,
        trace_context: ctx.trace_context,
        adapter: lora_resolution.adapter.clone(),
    };

    Ok(PreparedRequest {
        request_id: ctx.request_id,
        text_request,
        stream,
        include_usage,
        include_continuous_usage,
        include_logprobs,
        include_prompt_logprobs,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use uniserve_serving::text::Prompt;

    use super::prepare_generate_request;
    use crate::openai::{LoraModelResolution, ResolvedRequestContext};
    use crate::raw_generate::GenerateRequest;

    fn served(names: &[&str]) -> LoraModelResolution {
        LoraModelResolution {
            model_names: names.iter().map(|s| s.to_string()).collect(),
            adapter: uniserve_serving::AdapterSelection::Base,
        }
    }

    #[test]
    fn prepare_generate_request_maps_token_prompt_and_sampling_params() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22, 33],
            "priority": -3,
            "cache_salt": "salt",
            "sampling_params": {
                "max_tokens": 7,
                "logprobs": 2,
                "prompt_logprobs": 1,
                "ignore_eos": true
            }
        }))
        .expect("parse request");

        let prepared = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .expect("prepare");

        assert_eq!(
            prepared.text_request.prompt,
            Prompt::TokenIds(vec![11, 22, 33])
        );
        assert_eq!(prepared.text_request.sampling_params.max_tokens, Some(7));
        assert_eq!(prepared.text_request.sampling_params.logprobs, Some(2));
        assert_eq!(
            prepared.text_request.sampling_params.prompt_logprobs,
            Some(1)
        );
        assert!(prepared.text_request.sampling_params.ignore_eos);
        assert_eq!(prepared.text_request.priority, -3);
        assert_eq!(prepared.text_request.cache_salt.as_deref(), Some("salt"));
    }

    #[test]
    fn prepare_generate_request_rejects_kv_transfer_params() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22, 33],
            "sampling_params": {},
            "kv_transfer_params": {"connector": "x"}
        }))
        .expect("parse request");

        let error = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .expect_err("unsupported transfer parameters must be rejected");

        assert!(matches!(
            error,
            crate::openai::ApiError::InvalidRequest {
                param: Some("kv_transfer_params"),
                ..
            }
        ));
    }

    #[test]
    fn prepare_generate_request_gates_continuous_usage_on_include_usage() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": true,
            "stream_options": {
                "continuous_usage_stats": true
            },
            "sampling_params": {}
        }))
        .expect("parse request");

        let prepared = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .expect("prepare");

        assert!(!prepared.include_usage);
        assert!(!prepared.include_continuous_usage);
    }
}
