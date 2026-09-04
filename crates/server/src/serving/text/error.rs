//! Text tokenization, decoding, and stream errors.

use crate::engine_client::Error as GatewayError;
use thiserror::Error;

#[derive(Debug, Error)]
/// Text input, tokenization, decoding, or engine-stream failure.
pub enum Error {
    /// The tokenizer cannot encode or decode model text.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    /// Sampling controls violate the engine sampling contract.
    #[error("invalid sampling parameters: {0}")]
    InvalidSampling(#[from] uniserve_core::SamplingParamsError),
    /// A requested log-probability count falls outside the accepted range.
    #[error("{field} must be non-negative or -1, got {value}")]
    InvalidLogprobCount {
        /// Invalid request field.
        field: &'static str,
        /// Rejected field value.
        value: i32,
    },
    /// The minimum generation length exceeds the maximum generation length.
    #[error("min_tokens ({min_tokens}) exceeds max_tokens ({max_tokens})")]
    MinTokensExceedsMaximum {
        /// Requested minimum generated-token count.
        min_tokens: u32,
        /// Effective maximum generated-token count.
        max_tokens: u32,
    },
    /// A tokenized text request contains an empty prompt.
    #[error("text request `{request_id}` must contain at least one prompt token ID")]
    EmptyPromptTokenIds {
        /// Identifier of the rejected request.
        request_id: String,
    },
    /// The lowered request violates the canonical engine request contract.
    #[error("invalid canonical generation request: {0}")]
    InvalidGenerationRequest(String),
    /// The tokenized prompt fills or exceeds the model context window.
    #[error(
        "this model's maximum context length is {max_model_len} tokens, \
         but the prompt contains {prompt_len} input tokens"
    )]
    PromptTooLong {
        /// Maximum token count accepted by the model context.
        max_model_len: u32,
        /// Tokenized prompt length.
        prompt_len: u32,
    },
    /// The engine stream closes without terminal output.
    #[error("text request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput {
        /// Identifier of the incomplete request.
        request_id: String,
    },
    /// An engine event cannot be represented as valid decoded text output.
    #[error("text request `{request_id}` received malformed output: {message}")]
    MalformedOutput {
        /// Identifier of the affected request.
        request_id: String,
        /// Description of the violated output invariant.
        message: String,
    },
    /// Model profile assets cannot be resolved.
    #[error(transparent)]
    ModelAssets(#[from] crate::profile::assets::Error),
    /// The engine gateway rejects the request or fails its stream.
    #[error(transparent)]
    Gateway(#[from] GatewayError),
}

/// Result type returned by text serving operations.
pub type Result<T> = std::result::Result<T, Error>;

impl From<crate::profile::tokenizer::TokenizerError> for Error {
    /// Converts the source value into this type.
    fn from(error: crate::profile::tokenizer::TokenizerError) -> Self {
        Self::Tokenizer(error.0)
    }
}
