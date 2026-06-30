use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::Validate;

use crate::common::{
    LogProbs, Normalizable, Prompt, StreamOptions, StringOrArray, Usage, default_true,
    validate_stop,
};

/// Serde default for `CompletionRequest::max_tokens`, matching the reference
/// / OpenAI default.
fn default_completion_max_tokens() -> Option<u32> {
    Some(16)
}

/// Request type for the Completions API.

/// Mirrors the `CompletionRequest` class with UniServe extension fields.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub struct CompletionRequest {
    // -------- Standard OpenAI API Parameters --------
    /// ID of the model to use
    pub model: String,

    /// The prompt(s) to generate completions for.

    /// Token-ID input bypasses tokenizer work after conversion to the text
    /// facade.
    pub prompt: Prompt,

    /// Echo back the prompt in addition to the completion
    #[serde(default)]
    pub echo: bool,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based
    /// on their existing frequency in the text so far
    pub frequency_penalty: Option<f32>,

    /// Modify the likelihood of specified tokens appearing in the completion
    pub logit_bias: Option<HashMap<String, f32>>,

    /// Include the log probabilities on the logprobs most likely tokens
    pub logprobs: Option<u32>,

    /// The maximum number of tokens to generate (defaults to 16 when absent,
    /// matching the reference / OpenAI API convention)
    #[serde(default = "default_completion_max_tokens")]
    pub max_tokens: Option<u32>,

    /// How many completions to generate for each prompt
    pub n: Option<u32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based
    /// on whether they appear in the text so far
    pub presence_penalty: Option<f32>,

    /// If specified, our system will make a best effort to sample
    /// deterministically
    pub seed: Option<i64>,

    /// Up to 4 sequences where the API will stop generating further tokens
    #[validate(custom(function = "validate_stop"))]
    pub stop: Option<StringOrArray>,

    /// Whether to stream back partial progress
    #[serde(default)]
    pub stream: bool,

    /// The suffix that comes after a completion of inserted text
    pub suffix: Option<String>,

    /// What sampling temperature to use, between 0 and 2
    pub temperature: Option<f32>,

    /// An alternative to sampling with temperature (nucleus sampling)
    pub top_p: Option<f32>,

    /// A unique identifier representing your end-user
    pub user: Option<String>,

    // -------- Sampling Parameters --------
    /// Options for streaming response
    pub stream_options: Option<StreamOptions>,

    /// Use beam search instead of sampling
    #[serde(default)]
    pub use_beam_search: bool,

    /// Top-k sampling parameter
    pub top_k: Option<u32>,

    /// Min-p nucleus sampling parameter
    pub min_p: Option<f32>,

    /// Repetition penalty for reducing repetitive text
    pub repetition_penalty: Option<f32>,

    /// Length penalty for beam search
    pub length_penalty: Option<f32>,

    /// Specific token IDs to use as stop conditions
    pub stop_token_ids: Option<Vec<u32>>,

    /// Include stop string in output
    #[serde(default)]
    pub include_stop_str_in_output: bool,

    /// Ignore end-of-sequence tokens during generation
    #[serde(default)]
    pub ignore_eos: bool,

    /// Minimum number of tokens to generate
    pub min_tokens: Option<u32>,

    /// Skip special tokens during detokenization
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    /// Add spaces between special tokens during detokenization
    #[serde(default = "default_true")]
    pub spaces_between_special_tokens: bool,

    /// Truncate prompt tokens to this length
    pub truncate_prompt_tokens: Option<i64>,

    /// Restrict output to these token IDs only
    pub allowed_token_ids: Option<Vec<u32>>,

    /// Number of prompt logprobs to return
    pub prompt_logprobs: Option<i32>,

    // -------- Extra Parameters --------
    /// Whether to add special tokens (e.g. BOS) to the prompt
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,

    /// Format specification for structured output (JSON mode, JSON schema,
    /// etc.)
    pub response_format: Option<Value>,

    /// Additional kwargs for structured outputs
    pub structured_outputs: Option<Value>,

    /// Request scheduling priority (lower means earlier; default 0)
    pub priority: Option<i32>,

    /// External request ID used for response correlation.
    pub request_id: Option<String>,

    /// Tokens represented as strings of the form 'token_id:{token_id}' in
    /// logprobs
    pub return_tokens_as_token_ids: Option<bool>,

    /// Include token IDs alongside generated text
    pub return_token_ids: Option<bool>,

    /// Salt for prefix cache isolation in multi-user environments
    pub cache_salt: Option<String>,

    /// KV transfer parameters for disaggregated serving
    pub kv_transfer_params: Option<HashMap<String, Value>>,

    /// Additional request parameters with string or numeric values for custom
    /// extensions
    pub uniserve_xargs: Option<HashMap<String, Value>>,

    /// Additional fields
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Normalizable for CompletionRequest {}

/// Mirrors the `CompletionResponse` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Option<Usage>,
    pub system_fingerprint: Option<String>,
    pub kv_transfer_params: Option<Value>,
}

/// Mirrors the `CompletionResponseChoice` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<Value>,
    pub prompt_logprobs: Option<Vec<Option<HashMap<String, f32>>>>,
    pub token_ids: Option<Vec<u32>>,
    pub prompt_token_ids: Option<Vec<u32>>,
}

/// Mirrors the `CompletionStreamResponse` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct CompletionStreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionStreamChoice>,
    pub usage: Option<Usage>,
}

impl CompletionStreamResponse {
    /// Create a stream response with the standard envelope fields pre-filled.
    pub fn new(id: &str, model: &str, created: u64) -> Self {
        Self {
            id: id.to_string(),
            object: "text_completion".to_string(),
            created,
            model: model.to_string(),
            choices: Vec::new(),
            usage: None,
        }
    }
}

/// Mirrors the `CompletionResponseStreamChoice` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize)]
pub struct CompletionStreamChoice {
    pub index: u32,
    pub text: String,
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<Value>,
    pub token_ids: Option<Vec<u32>>,
    pub prompt_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum CompletionSseChunk {
    /// Ordinary OpenAI completions delta/final chunk.
    Chunk(CompletionStreamResponse),
    /// Final usage chunk emitted before `[DONE]` when `include_usage=true`.
    Usage(CompletionStreamResponse),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unknown top-level fields are preserved in the `other` catch-all rather
    /// than dropped, mirroring the `ChatCompletionRequest::other` behavior.
    #[test]
    fn unknown_fields_are_captured_in_other() {
        let json = serde_json::json!({
            "model": "test-model",
            "prompt": "hello",
            "temperature": 0.5,
            "some_future_field": 42,
            "vendor_extension": {"nested": true},
        });

        let req: CompletionRequest = serde_json::from_value(json).unwrap();

        // Known fields still deserialize normally.
        assert_eq!(req.model, "test-model");
        assert_eq!(req.temperature, Some(0.5));
        assert_eq!(req.prompt, Prompt::Text("hello".to_string()));

        // Unknown fields land in the catch-all, not silently discarded.
        assert_eq!(req.other.get("some_future_field"), Some(&Value::from(42)));
        assert_eq!(
            req.other.get("vendor_extension"),
            Some(&serde_json::json!({"nested": true}))
        );
        // Known fields are not duplicated into the catch-all.
        assert!(!req.other.contains_key("model"));
        assert!(!req.other.contains_key("temperature"));
    }

    /// A minimal request relies on serde defaults: `max_tokens` defaults to 16
    /// and the `default_true` flags default to true.
    #[test]
    fn minimal_request_applies_serde_defaults() {
        let json = serde_json::json!({
            "model": "m",
            "prompt": "p",
        });

        let req: CompletionRequest = serde_json::from_value(json).unwrap();

        assert_eq!(req.max_tokens, Some(16));
        assert!(!req.echo);
        assert!(!req.stream);
        assert!(req.skip_special_tokens);
        assert!(req.spaces_between_special_tokens);
        assert!(req.add_special_tokens);
        assert!(req.other.is_empty());
    }

    /// A token-id prompt deserializes into `Prompt::TokenIds` and round-trips
    /// back to the same JSON array.
    #[test]
    fn token_id_prompt_round_trips_through_serde() {
        let json = serde_json::json!({
            "model": "m",
            "prompt": [1, 2, 3],
        });

        let req: CompletionRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.prompt, Prompt::TokenIds(vec![1, 2, 3]));

        let reserialized = serde_json::to_value(&req).unwrap();
        assert_eq!(reserialized["prompt"], serde_json::json!([1, 2, 3]));
    }

    /// `skip_serializing_none` drops absent optional fields from the serialized
    /// request, so e.g. `seed` and `top_k` do not appear when unset.
    #[test]
    fn unset_optional_fields_are_omitted_on_serialize() {
        let json = serde_json::json!({
            "model": "m",
            "prompt": "p",
        });
        let req: CompletionRequest = serde_json::from_value(json).unwrap();

        let value = serde_json::to_value(&req).unwrap();
        let map = value.as_object().unwrap();
        assert!(!map.contains_key("seed"));
        assert!(!map.contains_key("top_k"));
        assert!(!map.contains_key("logprobs"));
        // Present-with-default scalar fields are still serialized.
        assert_eq!(map.get("max_tokens"), Some(&Value::from(16)));
    }

    /// A populated response serializes with the choice fields it carries and
    /// omits unset optional envelope fields (`skip_serializing_none`).
    #[test]
    fn response_serializes_choice_and_omits_unset_fields() {
        let response = CompletionResponse {
            id: "cmpl-1".to_string(),
            object: "text_completion".to_string(),
            created: 99,
            model: "m".to_string(),
            choices: vec![CompletionChoice {
                index: 0,
                text: "world".to_string(),
                logprobs: None,
                finish_reason: Some("stop".to_string()),
                stop_reason: None,
                prompt_logprobs: None,
                token_ids: None,
                prompt_token_ids: None,
            }],
            usage: None,
            system_fingerprint: None,
            kv_transfer_params: None,
        };

        let value = serde_json::to_value(&response).unwrap();
        let map = value.as_object().unwrap();
        assert_eq!(map.get("id"), Some(&Value::from("cmpl-1")));
        assert_eq!(value["choices"][0]["text"], Value::from("world"));
        assert_eq!(value["choices"][0]["finish_reason"], Value::from("stop"));
        // Unset response-envelope optionals are omitted.
        assert!(!map.contains_key("usage"));
        assert!(!map.contains_key("system_fingerprint"));
        // Unset choice optionals are omitted too.
        let choice = value["choices"][0].as_object().unwrap();
        assert!(!choice.contains_key("logprobs"));
        assert!(!choice.contains_key("token_ids"));
    }

    /// `CompletionStreamResponse::new` pre-fills the standard envelope.
    #[test]
    fn stream_response_new_prefills_envelope() {
        let stream = CompletionStreamResponse::new("id-1", "model-x", 7);

        assert_eq!(stream.id, "id-1");
        assert_eq!(stream.object, "text_completion");
        assert_eq!(stream.model, "model-x");
        assert_eq!(stream.created, 7);
        assert!(stream.choices.is_empty());
        assert!(stream.usage.is_none());
    }
}
