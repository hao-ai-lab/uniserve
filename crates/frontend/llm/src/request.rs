use std::collections::BTreeMap;

use uniserve_core::now_unix_secs;
use uniserve_engine_client::protocol::lora::LoraRequest;
use uniserve_engine_client::protocol::multimodal::MmFeatures;
use uniserve_engine_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};
use uuid::Uuid;

use crate::error::{Error, Result};

/// Tokenized decoder-only generate request accepted by [`crate::Llm`].

/// This is the first-stage Rust subset of the inputs that eventually flow into
/// Python `AsyncLLM.generate`. The boundary is intentionally above
/// [`EngineCoreRequest`], but below higher-level text and multimodal
/// preprocessing.

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    /// Unique ID of the request.
    pub request_id: String,
    /// Token IDs of the prompt.
    pub prompt_token_ids: Vec<u32>,
    /// Sampling parameters forwarded to the engine.
    pub sampling_params: EngineCoreSamplingParams,
    /// Optional multimodal features already prepared by `chat`.
    pub mm_features: Option<MmFeatures>,

    // Fields below are currently likely unused by callers.
    pub arrival_time: Option<f64>,
    pub cache_salt: Option<String>,
    pub trace_headers: Option<BTreeMap<String, String>>,
    pub priority: i32,
    pub data_parallel_rank: Option<u32>,
    pub reasoning_ended: Option<bool>,
    pub lora_request: Option<LoraRequest>,
}

#[derive(Debug)]
pub(crate) struct PreparedGenerateRequest {
    pub engine_request: EngineCoreRequest,
}

impl GenerateRequest {
    /// Validate and lower this request into the raw engine request format.
    pub(crate) fn prepare(self, randomize_request_id: bool) -> Result<PreparedGenerateRequest> {
        if self.prompt_token_ids.is_empty() {
            return Err(Error::EmptyPromptTokenIds {
                request_id: self.request_id,
            });
        }
        let GenerateRequest {
            request_id,
            prompt_token_ids,
            sampling_params,
            mm_features,
            arrival_time,
            cache_salt,
            trace_headers,
            priority,
            data_parallel_rank,
            reasoning_ended,
            lora_request,
        } = self;

        let external_request_id = request_id;
        let engine_request_id = if randomize_request_id {
            // Use the full UUID (128 bits) as the suffix rather than truncating
            // to 8 hex chars (32 bits): the truncated form has a non-negligible
            // birthday-bound collision probability among requests sharing the
            // same external request id.
            let random_suffix = Uuid::new_v4().simple().to_string();
            format!("{external_request_id}-{random_suffix}")
        } else {
            external_request_id.clone()
        };

        Ok(PreparedGenerateRequest {
            engine_request: EngineCoreRequest {
                request_id: engine_request_id,
                prompt_token_ids: Some(prompt_token_ids),
                mm_features,
                sampling_params: Some(sampling_params),
                pooling_params: None,
                arrival_time: arrival_time.unwrap_or_else(now_unix_secs),
                lora_request,
                cache_salt,
                data_parallel_rank,
                prompt_embeds: None,
                prompt_is_token_ids: None,
                client_index: 0,
                current_wave: 0,
                priority,
                trace_headers,
                resumable: false,
                external_req_id: Some(external_request_id),
                reasoning_ended,
                reasoning_parser_kwargs: None,
                abort_immediately: false,
                native: None,
            },
        })
    }
}

impl PreparedGenerateRequest {
    /// Return the original prompt token IDs copied into the raw engine request.
    pub(crate) fn prompt_token_ids(&self) -> &[u32] {
        self.engine_request
            .prompt_token_ids
            .as_deref()
            .unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use uniserve_engine_client::protocol::EngineCoreSamplingParams;

    use super::GenerateRequest;
    use crate::error::Error;

    fn sample_request() -> GenerateRequest {
        GenerateRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: vec![11, 22, 33],
            sampling_params: EngineCoreSamplingParams::for_test(),
            mm_features: None,
            arrival_time: Some(42.5),
            cache_salt: Some("salt".to_string()),
            trace_headers: Some(BTreeMap::from([(
                "x-trace-id".to_string(),
                "abc".to_string(),
            )])),
            priority: 3,
            data_parallel_rank: Some(2),
            reasoning_ended: Some(true),
            lora_request: None,
        }
    }

    #[test]
    fn prepare_builds_engine_request() {
        let prepared = sample_request().prepare(true).unwrap();

        assert_eq!(prepared.prompt_token_ids(), &[11, 22, 33]);

        let request = prepared.engine_request;
        assert_eq!(request.external_req_id.as_deref(), Some("req-1"));
        assert!(request.request_id.starts_with("req-1-"));
        assert_ne!(request.request_id, "req-1");
        assert_eq!(request.prompt_token_ids.as_deref(), Some(&[11, 22, 33][..]));
        assert_eq!(request.arrival_time, 42.5);
        assert_eq!(request.cache_salt.as_deref(), Some("salt"));
        assert_eq!(request.data_parallel_rank, Some(2));
        assert_eq!(
            request.trace_headers,
            Some(BTreeMap::from([(
                "x-trace-id".to_string(),
                "abc".to_string(),
            )]))
        );
        assert_eq!(request.reasoning_ended, Some(true));
    }

    #[test]
    fn prepare_rejects_empty_prompt_tokens() {
        let mut request = sample_request();
        request.prompt_token_ids.clear();

        let error = request.prepare(true).unwrap_err();
        assert!(matches!(
            error,
            Error::EmptyPromptTokenIds { request_id } if request_id == "req-1"
        ));
    }

    #[test]
    fn prepare_uses_full_uuid_suffix() {
        let prepared = sample_request().prepare(true).unwrap();
        let request_id = prepared.engine_request.request_id;

        let suffix = request_id
            .strip_prefix("req-1-")
            .expect("randomized id keeps the external prefix");
        // A simple-format UUID is 32 hex chars; the previous implementation
        // truncated it to 8, weakening collision resistance.
        assert_eq!(suffix.len(), 32);
        assert!(suffix.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn prepare_can_preserve_external_request_id() {
        let prepared = sample_request().prepare(false).unwrap();

        let request = prepared.engine_request;
        assert_eq!(request.external_req_id.as_deref(), Some("req-1"));
        assert_eq!(request.request_id, "req-1");
    }

    #[test]
    fn prepare_forwards_multimodal_features() {
        let mut request = sample_request();
        request.mm_features = Some(Vec::new());

        let prepared = request.prepare(false).unwrap();

        assert_eq!(prepared.engine_request.mm_features, Some(Vec::new()));
    }
}
