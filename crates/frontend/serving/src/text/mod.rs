//! Shared text-generation support used by chat and future raw completions.
//!
//! This crate intentionally stays below chat semantics:
//! prompt text handling, tokenizer/model loading, incremental detokenization,
//! and the thin generate-facing backend interface live here.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::mem::take;

pub use backend::{DynTextBackend, SamplingHints, TextBackend};
pub use error::{Error, Result};
use futures::Stream;
pub use lower::resolve_max_tokens;
pub(crate) use lower::{PreparedTextRequest, lower_text_request};
pub use output::{
    CollectedTextOutput, DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs,
    DecodedTextEvent, DecodedTokenLogprob, FinishReason, Finished, StopReason, TextDecodeOptions,
    TextOutputStreamExt,
};
pub use request::{Prompt, SamplingParams, TextRequest};
use trait_set::trait_set;
use uniserve_engine_gateway::EngineGateway;
use uniserve_engine_gateway::generation::GenerationEventStream;
use uniserve_model_profile::tokenizer::DynTokenizer;

pub mod backend;
mod error;
mod lower;
pub mod output;
mod request;
pub(crate) mod structured_output;
pub use uniserve_model_profile::tokenizer;

trait_set! {
 /// Shared streamed text output type used by raw completions and other text-only northbound paths.
    pub trait TextOutputStream = Stream<Item = Result<DecodedTextEvent>> + Send + 'static;
}

/// Text compilation and decoding implementation used by [`crate::ServingRuntime`].
///
/// This layer stays below chat semantics: prompt text or prompt token IDs flow
/// in, decoded text deltas and terminal metadata flow out.
#[derive(Clone)]
pub(crate) struct TextRuntime {
    /// Tokenizer/model metadata backend responsible for prompt encode/decode
    /// and sampling hints.
    backend: DynTextBackend,
    /// Context window size reported by the engine startup handshake, with
    /// optional override from config.
    max_model_len: u32,
}

impl TextRuntime {
    /// Create text runtime state from the resolved backend and gateway limits.
    pub(crate) fn new(gateway: &EngineGateway, backend: DynTextBackend) -> Self {
        // Prefer the engine-reported max_model_len because it reflects the
        // post-profiling, auto-fitted KV cache limit rather than static
        // frontend metadata.
        let max_model_len = gateway.snapshot().max_model_len;

        Self {
            backend,
            max_model_len,
        }
    }

    /// Override the maximum model context length explicitly.
    ///
    /// This takes priority over both the engine-reported default and any
    /// tokenizer/model metadata exposed by the backend.
    pub(crate) fn with_max_model_len(mut self, max_model_len: u32) -> Self {
        self.max_model_len = max_model_len;
        self
    }

    /// Return the tokenizer used by this text backend.
    pub(crate) fn tokenizer(&self) -> DynTokenizer {
        self.backend.tokenizer()
    }

    /// Compile one text request into a tokenized, engine-ready request without
    /// submitting it.
    pub(crate) fn compile(&self, mut request: TextRequest) -> Result<PreparedTextRequest> {
        request.validate()?;

        let tokenizer = self.backend.tokenizer();
        let prompt_token_ids = match take(&mut request.prompt) {
            Prompt::Text(text) => tokenizer.encode(&text, request.add_special_tokens)?,
            // Pre-tokenized prompts are the main completions-side escape hatch that lets benchmark
            // and infra workloads bypass chat rendering and tokenizer overhead entirely.
            Prompt::TokenIds(token_ids) => token_ids,
        };

        let mut sampling_hints = self.backend.sampling_hints()?;
        sampling_hints.max_model_len = Some(self.max_model_len);
        lower_text_request(request, prompt_token_ids, sampling_hints, &*tokenizer)
    }

    /// Submit a previously compiled request and return the raw token stream.
    pub(crate) async fn generate_raw_prepared(
        &self,
        gateway: &EngineGateway,
        prepared: PreparedTextRequest,
    ) -> Result<GenerationEventStream> {
        let raw_stream = gateway.submit_generation(prepared.submission).await?;
        Ok(raw_stream)
    }

    /// Submit a previously compiled request and stream incrementally decoded
    /// text.
    pub(crate) async fn generate_prepared(
        &self,
        gateway: &EngineGateway,
        prepared: PreparedTextRequest,
    ) -> Result<impl TextOutputStream> {
        let text_request = prepared.text_request.clone();
        let prompt_token_ids = prepared.submission.request.prompt_token_ids();
        let prompt_logprobs_requested = prepared
            .submission
            .request
            .sampling
            .prompt_logprobs_requested();
        let generated_logprobs_requested = prepared
            .submission
            .request
            .sampling
            .generated_logprobs_requested();
        let raw_stream = self.generate_raw_prepared(gateway, prepared).await?;
        let tokenizer = self.backend.tokenizer();
        let decoded_stream = output::decoded_text_event_stream(
            text_request.request_id,
            tokenizer,
            prompt_token_ids,
            prompt_logprobs_requested,
            generated_logprobs_requested,
            raw_stream,
            text_request.decode_options,
            text_request.intermediate,
        );

        Ok(decoded_stream)
    }
}
