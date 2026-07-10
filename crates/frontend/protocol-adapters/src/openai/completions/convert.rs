use uniserve_openai_types::{CompletionRequest, Prompt as OpenAiPrompt};
use uniserve_serving::text::{
    Prompt as TextPrompt, SamplingParams, TextDecodeOptions, TextRequest,
};
use uniserve_serving::{RequestMetadata, ServeRequest};

use super::validate;
use crate::openai::error::ApiError;
use crate::openai::lora::LoraModelResolution;
use crate::openai::structured_outputs::convert_from_response_format_value;
use crate::openai::utils::{ResolvedRequestContext, convert_logit_bias};

/// Lowered completion request plus the public response metadata carried by
/// every SSE chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedRequest {
    /// Stable OpenAI-style request ID, reused as the external text request ID.
    pub request_id: String,
    /// Public model ID echoed back to the client.
    pub response_model: String,
    /// Whether the caller asked for the final streamed usage chunk.
    pub include_usage: bool,
    /// Canonical semantic request submitted to the serving runtime.
    pub serve_request: ServeRequest,
    /// Original text prompt that should be echoed back northbound when
    /// `echo=true`.
    pub echo: Option<String>,
    /// Whether to include token IDs alongside generated text.
    pub return_token_ids: bool,
    /// Whether to format logprob tokens as `token_id:{id}`.
    pub return_tokens_as_token_ids: bool,
}

/// Validate and lower one OpenAI completions request into the internal
/// text-generation format.
///
/// `lora_resolution.model_names` must be non-empty; the first entry is used as
/// the base `model` field in responses when no LoRA adapter is selected.
pub fn prepare_completion_request(
    request: CompletionRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
) -> Result<PreparedRequest, ApiError> {
    validate::validate_request_compat(&request, &lora_resolution.model_names)?;
    if request
        .kv_transfer_params
        .as_ref()
        .is_some_and(|value| !value.is_empty())
    {
        return Err(ApiError::invalid_request(
            "`kv_transfer_params` is not supported by this runtime.".to_string(),
            Some("kv_transfer_params"),
        ));
    }
    if request
        .uniserve_xargs
        .as_ref()
        .is_some_and(|value| !value.is_empty())
    {
        return Err(ApiError::invalid_request(
            "`uniserve_xargs` is not supported; use typed request fields.".to_string(),
            Some("uniserve_xargs"),
        ));
    }

    let request_id = format!("cmpl-{}", ctx.request_id);
    let response_model = lora_resolution.response_model();

    let logprobs = match request.logprobs {
        Some(logprobs) => Some(i32::try_from(logprobs).map_err(|_| {
            ApiError::invalid_request(
                "`logprobs` must fit within a signed 32-bit integer.".to_string(),
                Some("logprobs"),
            )
        })?),
        None => None,
    };
    let prompt_logprobs = request
        .prompt_logprobs
        .or(if request.echo && !request.stream {
            logprobs
        } else {
            None
        });
    let include_usage = (request.stream_options.as_ref())
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let echo = request
        .echo
        .then(|| request.prompt.as_text().cloned())
        .flatten();
    let prompt = convert_prompt(request.prompt);

    let structured_outputs =
        convert_from_response_format_value(&request.response_format, &request.structured_outputs)?;

    let text_request = TextRequest {
        request_id: request_id.clone(),
        prompt,
        sampling_params: SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            seed: request.seed,
            max_tokens: request.max_tokens,
            min_tokens: request.min_tokens,
            logprobs,
            prompt_logprobs,
            min_p: request.min_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            repetition_penalty: request.repetition_penalty,
            stop_token_ids: request.stop_token_ids,
            ignore_eos: request.ignore_eos,
            logit_bias: convert_logit_bias(request.logit_bias)?,
            allowed_token_ids: request.allowed_token_ids,
            bad_words: None,
            logprob_token_ids: None,
            structured_outputs,
            skip_reading_prefix_cache: None,
            write_prefix_cache: None,
        },
        decode_options: TextDecodeOptions {
            skip_special_tokens: request.skip_special_tokens,
            include_stop_str_in_output: request.include_stop_str_in_output,
            stop_strings: request.stop.map(|stop| stop.into_vec()),
            min_tokens: request.min_tokens.unwrap_or(0),
        },
        intermediate: request.stream,
        priority: request.priority.unwrap_or(0),
        cache_salt: request.cache_salt,
        add_special_tokens: request.add_special_tokens,
        data_parallel_rank: ctx.data_parallel_rank,
        trace_context: ctx.trace_context,
        adapter: lora_resolution.adapter.clone(),
    };

    let serve_request = ServeRequest::from_text_request(
        text_request,
        RequestMetadata {
            protocol_adapter: Some("openai_completions".to_string()),
            route: Some("/v1/completions".to_string()),
            ..RequestMetadata::default()
        },
    )
    .map_err(|error| ApiError::invalid_request(error.to_string(), None))?;

    Ok(PreparedRequest {
        request_id,
        response_model,
        include_usage,
        serve_request,
        echo,
        return_token_ids: request.return_token_ids.unwrap_or(false),
        return_tokens_as_token_ids: request.return_tokens_as_token_ids.unwrap_or(false),
    })
}

fn convert_prompt(prompt: OpenAiPrompt) -> TextPrompt {
    match prompt {
        OpenAiPrompt::Text(text) => TextPrompt::Text(text),
        OpenAiPrompt::TokenIds(token_ids) => TextPrompt::TokenIds(token_ids),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use uniserve_openai_types::{CompletionRequest, Prompt as OpenAiPrompt};

    use super::prepare_completion_request;
    use crate::openai::lora::LoraModelResolution;
    use crate::openai::utils::ResolvedRequestContext;

    fn served(names: &[&str]) -> LoraModelResolution {
        LoraModelResolution {
            model_names: names.iter().map(|s| s.to_string()).collect(),
            adapter: uniserve_serving::AdapterSelection::Base,
        }
    }

    fn base_request_json() -> serde_json::Value {
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": true
        })
    }

    #[test]
    fn completion_http_request_deserializes_text_prompt() {
        let request: CompletionRequest =
            serde_json::from_value(base_request_json()).expect("parse request");

        assert_eq!(request.prompt, OpenAiPrompt::Text("hello".to_string()));
        assert_eq!(request.model, "Qwen/Qwen1.5-0.5B-Chat");
    }

    #[test]
    fn completion_http_request_deserializes_token_id_prompt() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": [11, 22, 33],
            "stream": true,
            "ignore_eos": true,
            "max_tokens": 7
        }))
        .expect("parse request");

        assert_eq!(request.prompt, OpenAiPrompt::TokenIds(vec![11, 22, 33]));
        assert_eq!(request.max_tokens, Some(7));
        assert!(request.ignore_eos);
    }

    #[test]
    fn prepare_completion_request_maps_sampling_fields() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": [11, 22, 33],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 7,
            "logprobs": 2,
            "top_p": 0.9,
            "top_k": 42,
            "min_p": 0.1,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.3,
            "repetition_penalty": 1.1,
            "ignore_eos": true,
            "skip_special_tokens": false
        }))
        .expect("parse request");

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .expect("prepare");

        assert!(prepared.include_usage);
        assert_eq!(
            prepared.serve_request.model_context,
            uniserve_serving::ModelContext::TokenIds {
                token_ids: vec![11, 22, 33],
                tokenizer: uniserve_serving::TokenizerReference::RuntimeProfile,
            }
        );
        assert_eq!(prepared.serve_request.generation.max_tokens, Some(7));
        assert_eq!(prepared.serve_request.generation.logprobs, Some(2));
        assert_eq!(prepared.serve_request.generation.top_p, Some(0.9));
        assert_eq!(prepared.serve_request.generation.top_k, Some(42));
        assert_eq!(prepared.serve_request.generation.min_p, Some(0.1));
        assert_eq!(
            prepared.serve_request.generation.frequency_penalty,
            Some(0.2)
        );
        assert_eq!(
            prepared.serve_request.generation.presence_penalty,
            Some(0.3)
        );
        assert_eq!(
            prepared.serve_request.generation.repetition_penalty,
            Some(1.1)
        );
        assert!(prepared.serve_request.generation.ignore_eos);
        assert!(!prepared.serve_request.generation.skip_special_tokens);
    }

    #[test]
    fn prepare_completion_request_accepts_text_echo() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": true,
            "echo": true,
            "max_tokens": 7
        }))
        .expect("parse request");

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .expect("prepare");

        assert_eq!(prepared.echo, Some("hello".to_string()));
        assert_eq!(prepared.serve_request.generation.max_tokens, Some(7));
    }

    #[test]
    fn prepare_completion_request_enables_prompt_logprobs_for_non_stream_echo() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "echo": true,
            "stream": false,
            "logprobs": 3
        }))
        .expect("parse request");

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .expect("prepare");

        assert_eq!(prepared.serve_request.generation.logprobs, Some(3));
        assert_eq!(prepared.serve_request.generation.prompt_logprobs, Some(3));
    }

    #[test]
    fn prepare_completion_request_rejects_token_id_prompt_echo() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": [11, 22, 33],
            "stream": true,
            "echo": true
        }))
        .expect("parse request");

        assert!(
            prepare_completion_request(
                request,
                &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
                ResolvedRequestContext::default(),
            )
            .is_err()
        );
    }

    #[test]
    fn prepare_completion_request_accepts_logprobs_fields() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": false,
            "logprobs": 1,
            "prompt_logprobs": 2
        }))
        .expect("parse request");

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .expect("prepare");
        assert_eq!(prepared.serve_request.generation.logprobs, Some(1));
        assert_eq!(prepared.serve_request.generation.prompt_logprobs, Some(2));
    }

    #[test]
    fn prepare_completion_request_threads_data_parallel_rank() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": false,
        }))
        .expect("parse request");

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext {
                request_id: "req".to_string(),
                data_parallel_rank: Some(3),
                ..ResolvedRequestContext::default()
            },
        )
        .expect("prepare");
        assert_eq!(
            prepared.serve_request.scheduling.data_parallel_rank,
            Some(3)
        );
    }

    #[test]
    fn prepare_completion_request_leaves_data_parallel_rank_none_when_absent() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": false,
        }))
        .expect("parse request");

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .expect("prepare");
        assert_eq!(prepared.serve_request.scheduling.data_parallel_rank, None);
    }
}
