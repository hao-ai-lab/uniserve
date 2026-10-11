//! Errors returned by the server-facing engine client.

use thiserror::Error;
use thiserror_ext::Macro;

/// Result type returned by engine-client calls.
pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the engine client.
#[derive(Debug, Error, Macro)]
pub enum Error {
    /// A live request already owns the supplied identifier
    /// (`EngineClient::register_request`), or the registered request was
    /// already submitted to the engine (`EngineClient::submit_generation`,
    /// `EngineClient::submit_media`).
    #[error("request `{request_id}` is already in flight")]
    DuplicateRequestId {
        /// Conflicting external request identifier.
        request_id: String,
    },
    /// Submission names an identifier that is not registered, or whose
    /// registration was released or replaced before the submission claimed
    /// it.
    #[error("request `{request_id}` has no matching registration")]
    UnknownRequestId {
        /// External identity supplied by the caller.
        request_id: String,
    },
    /// The engine rejected request submission.
    #[error(transparent)]
    Submit(#[from] uniserve_engine::SubmitError),
    /// The client cannot serve the call: the engine failed to start or to
    /// complete its shutdown, or the loaded runtime family does not serve the
    /// submitted request kind.
    #[error("engine client is closed: {message}")]
    ClientClosed {
        /// Human-readable closure context.
        message: String,
    },
}
