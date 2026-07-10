use thiserror::Error;
use uniserve_engine_gateway::Error as GatewayError;

#[derive(Debug, Error)]
pub enum Error {
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("invalid structured-output constraint: {0}")]
    StructuredOutput(String),
    #[error("invalid sampling parameters: {0}")]
    InvalidSampling(#[from] uniserve_core::SamplingParamsError),
    #[error("{field} must be non-negative or -1, got {value}")]
    InvalidLogprobCount { field: &'static str, value: i32 },
    #[error("min_tokens ({min_tokens}) exceeds max_tokens ({max_tokens})")]
    MinTokensExceedsMaximum { min_tokens: u32, max_tokens: u32 },
    #[error("text request `{request_id}` must contain at least one prompt token ID")]
    EmptyPromptTokenIds { request_id: String },
    #[error("text request `{request_id}` has an adapter id outside the scheduler id space")]
    AdapterIdOutOfRange { request_id: String },
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
    ModelAssets(#[from] uniserve_model_profile::assets::Error),
    #[error(transparent)]
    Gateway(#[from] GatewayError),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<uniserve_model_profile::tokenizer::TokenizerError> for Error {
    fn from(error: uniserve_model_profile::tokenizer::TokenizerError) -> Self {
        Self::Tokenizer(error.0)
    }
}
