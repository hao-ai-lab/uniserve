//! Chat rendering, parsing, and output-processing errors.
//!
//! Request validation, template rendering, and the Qwen3 output-processing
//! stages all report this one error type. `assemble_chat_event_stream` wraps
//! output-time errors in `ServeError::OutputProcessing`, except an engine
//! rejection carried in [`Error::Text`], which it turns into a `Rejected`
//! serving event.

use thiserror::Error;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Error)]
/// Error produced by chat request rendering or output processing.
pub enum Error {
    /// The request contains no messages to render.
    #[error("chat request must contain at least one message")]
    EmptyMessages,
    /// Continuation mode is selected for history that does not end with an assistant message.
    #[error("cannot continue the final message when the last message is not from the assistant")]
    ContinueFinalAssistantWithoutFinalAssistant,
    /// Prompt rendering requires a template but no template is configured.
    #[error("chat template is required but none was configured")]
    MissingChatTemplate,
    /// The chat template cannot be loaded, compiled, or rendered, or assistant
    /// history cannot be converted into template values (for example, tool-call
    /// arguments that are not valid JSON).
    #[error("chat template error: {0}")]
    ChatTemplate(String),
    /// The selected renderer cannot represent multimodal content.
    #[error("multimodal input is not supported by this chat renderer")]
    UnsupportedMultimodalRenderer,
    /// Multimodal content reached a text-only conversion or a renderer
    /// configured without multimodal support.
    #[error("unsupported multimodal content: {0}")]
    UnsupportedMultimodalContent(&'static str),
    /// Multimodal content cannot be converted into model inputs.
    #[error("multimodal preprocessing error: {0}")]
    Multimodal(String),
    /// A reasoning or tool-call output parser cannot be initialized.
    #[error("failed to initialize {kind} parser `{name}`")]
    ParserInitialization {
        /// Semantic parser category.
        kind: &'static str,
        /// Configured parser name.
        name: String,
        /// Parser-specific initialization error.
        #[source]
        error: BoxedError,
    },
    /// The rendered prompt exceeds the model context limit.
    #[error(
        "this model's maximum context length is {max_model_len} tokens, \
         but the prompt contains {prompt_len} input tokens"
    )]
    PromptTooLong {
        /// Maximum prompt length accepted by the model.
        max_model_len: u32,
        /// Rendered prompt length in tokens.
        prompt_len: u32,
    },
    /// The engine stream ends without a terminal event.
    #[error("chat request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput {
        /// Identifier of the incomplete request.
        request_id: String,
    },
    /// Incremental tool-call events violate the assembler state contract:
    /// arguments arrive with no open tool call, or for a tool index other than
    /// the open one.
    #[error("tool call stream state is inconsistent: {message}")]
    ToolCallStreamInvariant {
        /// Description of the violated stream invariant.
        message: String,
    },
    /// Model profile assets cannot be loaded or validated.
    #[error(transparent)]
    ModelAssets(#[from] crate::profile::assets::Error),
    /// The text layer fails, for example in detokenization or engine output
    /// validation, or reports an engine rejection.
    #[error(transparent)]
    Text(#[from] crate::serving::text::Error),
}

/// Result type returned by chat calls.
pub type Result<T> = std::result::Result<T, Error>;
