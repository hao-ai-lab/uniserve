use thiserror::Error;
use thiserror_ext::Macro;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Error, Macro)]
#[thiserror_ext(macro(path = "crate::chat::error"))]
pub enum Error {
    #[error("chat request must contain at least one message")]
    EmptyMessages,
    #[error("cannot continue the final message when the last message is not from the assistant")]
    ContinueFinalAssistantWithoutFinalAssistant,
    #[error("chat template is required but none was configured")]
    MissingChatTemplate,
    #[error("chat template error: {0}")]
    ChatTemplate(String),
    #[error("multimodal input is not supported by this chat renderer")]
    UnsupportedMultimodalRenderer,
    #[error("unsupported multimodal content: {0}")]
    UnsupportedMultimodalContent(&'static str),
    #[error("multimodal preprocessing error: {0}")]
    Multimodal(#[message] String),
    #[error("failed to initialize {kind} parser `{name}`")]
    ParserInitialization {
        kind: &'static str,
        name: String,
        #[source]
        error: BoxedError,
    },
    #[error(
        "this model's maximum context length is {max_model_len} tokens, \
         but the prompt contains {prompt_len} input tokens"
    )]
    PromptTooLong { max_model_len: u32, prompt_len: u32 },
    #[error("chat request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput { request_id: String },
    #[error("tool call stream state is inconsistent: {message}")]
    ToolCallStreamInvariant { message: String },
    #[error(transparent)]
    ModelAssets(#[from] uniserve_model_profile::assets::Error),
    #[error(transparent)]
    Text(#[from] crate::text::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<crate::chat::protocol::Error> for Error {
    fn from(error: crate::chat::protocol::Error) -> Self {
        match error {
            crate::chat::protocol::Error::EmptyMessages => Self::EmptyMessages,
            crate::chat::protocol::Error::ContinueFinalAssistantWithoutFinalAssistant => {
                Self::ContinueFinalAssistantWithoutFinalAssistant
            }
            crate::chat::protocol::Error::ChatTemplate(message) => Self::ChatTemplate(message),
            crate::chat::protocol::Error::UnsupportedMultimodalContent(kind) => {
                Self::UnsupportedMultimodalContent(kind)
            }
            crate::chat::protocol::Error::StreamClosedBeforeTerminalOutput { request_id } => {
                Self::StreamClosedBeforeTerminalOutput { request_id }
            }
        }
    }
}

impl From<crate::chat::template::Error> for Error {
    fn from(error: crate::chat::template::Error) -> Self {
        match error {
            crate::chat::template::Error::MissingChatTemplate => Self::MissingChatTemplate,
            crate::chat::template::Error::ChatTemplate(message) => Self::ChatTemplate(message),
            crate::chat::template::Error::UnsupportedMultimodalRenderer => {
                Self::UnsupportedMultimodalRenderer
            }
            crate::chat::template::Error::UnsupportedMultimodalContent(kind) => {
                Self::UnsupportedMultimodalContent(kind)
            }
            crate::chat::template::Error::ModelAssets(error) => Self::ModelAssets(error),
            crate::chat::template::Error::Protocol(error) => error.into(),
            crate::chat::template::Error::Text(error) => error.into(),
        }
    }
}

impl From<crate::chat::output::Error> for Error {
    fn from(error: crate::chat::output::Error) -> Self {
        match error {
            crate::chat::output::Error::ParserInitialization { kind, name, error } => {
                Self::ParserInitialization { kind, name, error }
            }
            crate::chat::output::Error::ToolCallStreamInvariant { message } => {
                Self::ToolCallStreamInvariant { message }
            }
            crate::chat::output::Error::StreamClosedBeforeTerminalOutput { request_id } => {
                Self::StreamClosedBeforeTerminalOutput { request_id }
            }
            crate::chat::output::Error::Protocol(error) => error.into(),
            crate::chat::output::Error::Text(error) => error.into(),
        }
    }
}
