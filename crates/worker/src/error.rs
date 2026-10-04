//! Errors from request execution and resource ownership.

/// Invalid submissions and operations on unavailable worker state.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    State(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;
