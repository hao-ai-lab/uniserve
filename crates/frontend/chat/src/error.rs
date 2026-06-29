use thiserror::Error;
use thiserror_ext::Macro;
use uniserve_chat_output::error::available_parser_hint;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Error, Macro)]
#[thiserror_ext(macro(path = "crate::error"))]
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
    #[error("{kind} parsing is not available for model `{model_id}`")]
    ParserUnavailableForModel {
        kind: &'static str,
        model_id: String,
    },
    #[error("{kind} parsing is disabled by frontend configuration")]
    ParserDisabled { kind: &'static str },
    #[error(
        "{kind} parser `{name}` is not registered{}",
        available_parser_hint(.available_names)
    )]
    ParserUnavailableByName {
        kind: &'static str,
        name: String,
        available_names: Vec<String>,
    },
    #[error("failed to initialize {kind} parser `{name}`")]
    ParserInitialization {
        kind: &'static str,
        name: String,
        #[source]
        error: BoxedError,
    },
    #[error(
        "gpt_oss uses native Harmony output parsing; generic {kind} parser override `{selection}` is not supported"
    )]
    HarmonyParserOverrideUnsupported {
        kind: &'static str,
        selection: String,
    },
    #[error("harmony output parsing failed")]
    HarmonyOutputParsing {
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
    ModelAssets(#[from] uniserve_model_assets::Error),
    #[error(transparent)]
    Text(#[from] uniserve_text::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<uniserve_chat_protocol::Error> for Error {
    fn from(error: uniserve_chat_protocol::Error) -> Self {
        match error {
            uniserve_chat_protocol::Error::EmptyMessages => Self::EmptyMessages,
            uniserve_chat_protocol::Error::ContinueFinalAssistantWithoutFinalAssistant => {
                Self::ContinueFinalAssistantWithoutFinalAssistant
            }
            uniserve_chat_protocol::Error::ChatTemplate(message) => Self::ChatTemplate(message),
            uniserve_chat_protocol::Error::UnsupportedMultimodalContent(kind) => {
                Self::UnsupportedMultimodalContent(kind)
            }
            uniserve_chat_protocol::Error::StreamClosedBeforeTerminalOutput { request_id } => {
                Self::StreamClosedBeforeTerminalOutput { request_id }
            }
        }
    }
}

impl From<uniserve_chat_template::Error> for Error {
    fn from(error: uniserve_chat_template::Error) -> Self {
        match error {
            uniserve_chat_template::Error::MissingChatTemplate => Self::MissingChatTemplate,
            uniserve_chat_template::Error::ChatTemplate(message) => Self::ChatTemplate(message),
            uniserve_chat_template::Error::UnsupportedMultimodalRenderer => {
                Self::UnsupportedMultimodalRenderer
            }
            uniserve_chat_template::Error::UnsupportedMultimodalContent(kind) => {
                Self::UnsupportedMultimodalContent(kind)
            }
            uniserve_chat_template::Error::ModelAssets(error) => Self::ModelAssets(error),
            uniserve_chat_template::Error::Protocol(error) => error.into(),
            uniserve_chat_template::Error::Text(error) => error.into(),
        }
    }
}

impl From<uniserve_chat_output::Error> for Error {
    fn from(error: uniserve_chat_output::Error) -> Self {
        match error {
            uniserve_chat_output::Error::ParserUnavailableForModel { kind, model_id } => {
                Self::ParserUnavailableForModel { kind, model_id }
            }
            uniserve_chat_output::Error::ParserDisabled { kind } => Self::ParserDisabled { kind },
            uniserve_chat_output::Error::ParserUnavailableByName {
                kind,
                name,
                available_names,
            } => Self::ParserUnavailableByName {
                kind,
                name,
                available_names,
            },
            uniserve_chat_output::Error::ParserInitialization { kind, name, error } => {
                Self::ParserInitialization { kind, name, error }
            }
            uniserve_chat_output::Error::HarmonyParserOverrideUnsupported { kind, selection } => {
                Self::HarmonyParserOverrideUnsupported { kind, selection }
            }
            uniserve_chat_output::Error::HarmonyOutputParsing { error } => {
                Self::HarmonyOutputParsing { error }
            }
            uniserve_chat_output::Error::ToolCallStreamInvariant { message } => {
                Self::ToolCallStreamInvariant { message }
            }
            uniserve_chat_output::Error::StreamClosedBeforeTerminalOutput { request_id } => {
                Self::StreamClosedBeforeTerminalOutput { request_id }
            }
            uniserve_chat_output::Error::Protocol(error) => error.into(),
            uniserve_chat_output::Error::Text(error) => error.into(),
        }
    }
}
