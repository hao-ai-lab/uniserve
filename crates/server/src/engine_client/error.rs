use std::sync::Arc;

use thiserror::Error;
use thiserror_ext::Macro;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the engine client.
#[derive(Debug, Error, Macro)]
pub enum Error {
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("request `{request_id}` is already in flight")]
    DuplicateRequestId { request_id: String },
    #[error(transparent)]
    Submit(#[from] uniserve_engine::SubmitError),
    #[error("engine client is closed: {message}")]
    ClientClosed { message: String },
    #[error("engine status is unavailable because the engine client is closed")]
    StatusUnavailable,

    /// A special variant to allow cloning the same error.
    #[error(transparent)]
    Shared(Arc<Self>),
}
