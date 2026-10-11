//! The worker's failure classes as a Python enum.
//!
//! [`PyErrorCode`] is the extension's `ErrorCode` class: the worker raises
//! and reports failures with its members, and `convert` reads them back as
//! [`ErrorCode`]. The conversion is an exhaustive `match`, so a class added to
//! the schema's enum does not compile until it has a Python member.

use pyo3::prelude::*;
use uniserve_worker_ipc::ErrorCode;

/// A failure class of the worker, spelled `ErrorCode.<MEMBER>` in Python;
/// `str` gives the snake_case name the engine logs.
#[pyclass(
    name = "ErrorCode",
    module = "uniserve_worker._uniserve_ipc",
    eq,
    eq_int,
    hash,
    frozen,
    str,
    from_py_object,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum PyErrorCode {
    UnsupportedCall,
    InvalidDescriptor,
    UnsupportedSetup,
    InputError,
    ComputeError,
    ResourceError,
    InvariantViolation,
    FatalWorkerFailure,
    InvalidRequest,
}

impl PyErrorCode {
    /// Returns the schema's class for this member.
    fn code(&self) -> ErrorCode {
        match self {
            Self::UnsupportedCall => ErrorCode::UnsupportedCall,
            Self::InvalidDescriptor => ErrorCode::InvalidDescriptor,
            Self::UnsupportedSetup => ErrorCode::UnsupportedSetup,
            Self::InputError => ErrorCode::InputError,
            Self::ComputeError => ErrorCode::ComputeError,
            Self::ResourceError => ErrorCode::ResourceError,
            Self::InvariantViolation => ErrorCode::InvariantViolation,
            Self::FatalWorkerFailure => ErrorCode::FatalWorkerFailure,
            Self::InvalidRequest => ErrorCode::InvalidRequest,
        }
    }
}

impl From<PyErrorCode> for ErrorCode {
    fn from(code: PyErrorCode) -> Self {
        code.code()
    }
}

impl std::fmt::Display for PyErrorCode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.code(), formatter)
    }
}
