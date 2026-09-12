//! Errors returned by the server-facing engine client.

use std::sync::Arc;

use thiserror::Error;
use thiserror_ext::Macro;

/// Result type returned by engine-client operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the engine client.
#[derive(Debug, Error, Macro)]
pub enum Error {
    #[error("io error")]
    /// An operating-system or transport operation failed.
    Io(#[from] std::io::Error),
    #[error("request `{request_id}` is already in flight")]
    /// A live request already owns the supplied identifier.
    DuplicateRequestId {
        /// Conflicting external request identifier.
        request_id: String,
    },
    /// Submission does not match the reserved request incarnation.
    #[error("request `{request_id}` has no matching registration")]
    UnknownRequestId {
        /// External identity supplied by the caller.
        request_id: String,
    },
    #[error(transparent)]
    /// The engine rejected request submission.
    Submit(#[from] uniserve_engine::SubmitError),
    #[error("engine client is closed: {message}")]
    /// The client closed before completing the requested operation.
    ClientClosed {
        /// Human-readable closure context.
        message: String,
    },
    #[error("engine status is unavailable because the engine client is closed")]
    /// Status cannot be read after client closure.
    StatusUnavailable,

    /// Shared ownership of an error returned to multiple observers.
    #[error(transparent)]
    Shared(Arc<Self>),
}
