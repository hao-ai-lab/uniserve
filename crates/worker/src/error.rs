//! Errors from request execution and resource ownership.

/// Invalid submissions and operations on unavailable worker state.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Cuda(String),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    State(&'static str),
    #[error("{0}")]
    Resource(&'static str),
    #[error("{0}")]
    Unsupported(String),
    #[error("asynchronous transfer ticket capacity is exhausted")]
    ReadBackpressure { returns: u64 },
    #[error("{0}")]
    Invariant(String),
}

pub type Result<T> = std::result::Result<T, Error>;
