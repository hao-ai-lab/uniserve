use crate::engine_client::Error as GatewayError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("invalid sampling parameters: {0}")]
    InvalidSampling(#[from] uniserve_core::SamplingParamsError),
    #[error("{field} must be non-negative or -1, got {value}")]
    InvalidLogprobCount { field: &'static str, value: i32 },
    #[error("min_tokens ({min_tokens}) exceeds max_tokens ({max_tokens})")]
    MinTokensExceedsMaximum { min_tokens: u32, max_tokens: u32 },
    #[error("text request `{request_id}` must contain at least one prompt token ID")]
    EmptyPromptTokenIds { request_id: String },
    #[error("invalid canonical generation request: {0}")]
    InvalidGenerationRequest(String),
    #[error(
        "this model's maximum context length is {max_model_len} tokens, \
         but the prompt contains {prompt_len} input tokens"
    )]
    PromptTooLong { max_model_len: u32, prompt_len: u32 },
    #[error("text request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput { request_id: String },
    #[error("text request `{request_id}` received malformed output: {message}")]
    MalformedOutput { request_id: String, message: String },
    #[error(transparent)]
    ModelAssets(#[from] crate::profile::assets::Error),
    #[error(transparent)]
    Gateway(#[from] GatewayError),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<crate::profile::tokenizer::TokenizerError> for Error {
    fn from(error: crate::profile::tokenizer::TokenizerError) -> Self {
        Self::Tokenizer(error.0)
    }
}
