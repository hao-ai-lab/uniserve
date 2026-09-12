//! Versioned scheduler-to-worker protocol and shared-memory transport.
//!
//! [`NewRequest`] admits static state, [`ScheduleBatch`] submits planned operations, and
//! [`BatchOutput`] returns completions and products. [`WorkerInfo`] describes a
//! loaded worker before execution begins. Numeric request, operation, point,
//! and generation identities remain stable across serialization.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use uniserve_core::{
    BlockId, ImageParams, KvCacheDtype, KvCacheGroup, RequestId, SamplingParams, TokenLogprob,
};
pub use uniserve_core::{ComputationId, ForwardStats};

/// Result type for semantic worker-message validation.
pub type ValidationResult<T> = std::result::Result<T, ValidationError>;

/// Describes a worker message that violates the protocol contract.
#[derive(Debug, thiserror::Error)]
#[error("invalid worker message: {0}")]
pub struct ValidationError(String);

impl ValidationError {
    /// Constructs a validation error from contextual text.
    pub(crate) fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl From<uniserve_core::SamplingParamsError> for ValidationError {
    /// Converts sampling validation failures into worker-protocol failures.
    fn from(error: uniserve_core::SamplingParamsError) -> Self {
        Self::message(error.to_string())
    }
}

impl From<uniserve_core::ImageParamsError> for ValidationError {
    /// Converts image validation failures into worker-protocol failures.
    fn from(error: uniserve_core::ImageParamsError) -> Self {
        Self::message(error.to_string())
    }
}

macro_rules! invalid_message {
    ($($arg:tt)*) => {
        ValidationError::message(format!($($arg)*))
    };
}

macro_rules! bail_invalid {
    ($($arg:tt)*) => {
        return Err(invalid_message!($($arg)*))
    };
}

macro_rules! ensure_valid {
    ($condition:expr, $($arg:tt)*) => {
        if !$condition {
            bail_invalid!($($arg)*);
        }
    };
}

/// FlatBuffers encoding and decoding for protocol messages.
pub mod codec;
/// Shared-memory request-response endpoints and wake events.
pub mod iceoryx;
#[allow(missing_docs, warnings)]
/// FlatBuffers bindings generated from the worker protocol schema.
pub mod schema {
    include!(concat!(env!("OUT_DIR"), "/flatbuffers/mod.rs"));
}

pub use iceoryx::{
    ClientEndpoint, DEFAULT_SERVICE_PREFIX, EVT_COMMAND, EVT_COMPLETION, EVT_DEATH, EVT_REQUEST,
    EVT_RESULT, Frame, Header, IPC_VERSION, IpcError, IpcResult, Pending, ServerEndpoint,
    WakeEvents, WakeSender, header_for_request, header_for_response, is_supported_ipc_version,
    service_name,
};

mod info;
mod operation;
mod request;
mod tensor;

pub use info::*;
pub use operation::*;
pub use request::*;
pub use tensor::*;
#[cfg(test)]
mod tests;
