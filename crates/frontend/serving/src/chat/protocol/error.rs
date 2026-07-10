use thiserror::Error;

/// Errors owned by chat request/event protocol helpers.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("chat request must contain at least one message")]
    EmptyMessages,
    #[error("cannot continue the final message when the last message is not from the assistant")]
    ContinueFinalAssistantWithoutFinalAssistant,
    #[error("chat template error: {0}")]
    ChatTemplate(String),
    #[error("unsupported multimodal content: {0}")]
    UnsupportedMultimodalContent(&'static str),
    #[error("chat request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput { request_id: String },
}

pub type Result<T> = std::result::Result<T, Error>;
