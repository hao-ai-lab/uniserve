use thiserror::Error as ThisError;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, ThisError)]
pub enum Error {
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
    #[error("tool call stream state is inconsistent: {message}")]
    ToolCallStreamInvariant { message: String },
    #[error("chat request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput { request_id: String },
    #[error(transparent)]
    Protocol(#[from] uniserve_chat_protocol::Error),
    #[error(transparent)]
    Text(#[from] uniserve_text::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Format the available-parser suffix used in user-facing error messages.

/// Shared with downstream crates (e.g. `uniserve-chat`) that re-wrap these
/// errors so the "choose from" hint wording stays identical across layers.
pub fn available_parser_hint(available_names: &[String]) -> String {
    if available_names.is_empty() {
        String::new()
    } else {
        format!(" (choose from: {})", available_names.join(", "))
    }
}
