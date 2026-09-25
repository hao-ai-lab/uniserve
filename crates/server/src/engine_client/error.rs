//! Errors returned by the server-facing engine client.

use std::sync::Arc;

use thiserror::Error;
use thiserror_ext::Macro;

/// Result type returned by engine-client calls.
pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the engine client.
#[derive(Debug, Error, Macro)]
pub enum Error {
    /// An operating-system or transport call failed.
    #[error("io error")]
    Io(#[from] std::io::Error),
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
    /// registration does not hold the engine `RequestId` the submitted request
    /// carries (a different incarnation, or one whose engine side was already
    /// released).
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
    /// Status cannot be read after client closure.
    #[error("engine status is unavailable because the engine client is closed")]
    StatusUnavailable,

    /// Shared ownership of an error returned to multiple observers.
    #[error(transparent)]
    Shared(Arc<Self>),
}
