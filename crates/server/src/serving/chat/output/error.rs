use thiserror::Error as ThisError;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("failed to initialize {kind} parser `{name}`")]
    ParserInitialization {
        kind: &'static str,
        name: String,
        #[source]
        error: BoxedError,
    },
    #[error("tool call stream state is inconsistent: {message}")]
    ToolCallStreamInvariant { message: String },
    #[error("chat request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput { request_id: String },
    #[error(transparent)]
    Protocol(#[from] crate::serving::chat::protocol::Error),
    #[error(transparent)]
    Text(#[from] crate::serving::text::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
