use thiserror::Error;
use uniserve_engine_client::Error as EngineCoreError;
use uniserve_llm::Error as LlmError;

#[derive(Debug, Error)]
pub enum Error {
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("text request `{request_id}` must contain at least one prompt token ID")]
    EmptyPromptTokenIds { request_id: String },
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
    ModelAssets(#[from] uniserve_model_assets::Error),
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error(transparent)]
    EngineCore(#[from] EngineCoreError),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<uniserve_tokenizer::TokenizerError> for Error {
    fn from(error: uniserve_tokenizer::TokenizerError) -> Self {
        Self::Tokenizer(error.0)
    }
}
