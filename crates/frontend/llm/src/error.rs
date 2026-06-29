use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the Rust `llm` facade.
#[derive(Debug, Error)]
pub enum Error {
    #[error("generate request `{request_id}` has an empty prompt_token_ids")]
    EmptyPromptTokenIds { request_id: String },
    #[error("engine client error")]
    EngineCoreClient(#[from] uniserve_engine_client::Error),
    #[error("malformed engine output: {message}")]
    MalformedEngineOutput { message: String },
}
