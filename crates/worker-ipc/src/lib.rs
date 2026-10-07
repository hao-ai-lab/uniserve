//! Versioned scheduler-to-worker protocol and its rank transports.
//!
//! [`NewRequest`] admits static state, [`Batch`] submits planned calls, and
//! [`BatchOutput`] returns completions and products. [`WorkerInfo`] describes a
//! loaded worker before execution begins. Numeric request, call, point,
//! and generation identities remain stable across serialization.
//!
//! [`WorkerRequest`] and [`WorkerResponse`] are the envelopes exchanged per
//! rank, encoded by [`codec`] as FlatBuffers payloads behind a fixed
//! [`Header`]. The engine reaches a rank on the head's host through
//! [`iceoryx`] shared storage and a rank on another host through a [`socket`]
//! stream; both carry the same frames, and [`channel`] presents either one as
//! a [`RankChannel`] (engine side) or [`RankServer`] (rank side). The Python
//! worker binds [`RankServer`] through the `worker-ipc-py` extension, so the
//! engine and that extension must be built with the same [`IPC_VERSION`].

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
pub use uniserve_core::{CallId, ForwardStats};
use uniserve_core::{
    ImageParams, KvCacheDtype, KvCacheGroup, KvGroupKind, RequestId, SamplingParams, TokenLogprob,
    UnitId,
};

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

// The validation macros below are textually scoped: they are visible only in
// modules declared after them, which is how `call`, `info` and `tensor` use
// them. Keep them above those `mod` declarations.
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

/// Both ends of a rank channel over either transport.
pub mod channel;
/// FlatBuffers encoding and decoding for protocol messages.
pub mod codec;
/// Shared-storage rank transport for ranks on the head's host.
pub mod iceoryx;
/// Stream-socket rank transport for ranks on other hosts.
pub mod socket;
#[allow(missing_docs, warnings)]
/// FlatBuffers bindings generated from the worker protocol schema.
pub mod schema {
    include!(concat!(env!("OUT_DIR"), "/flatbuffers/mod.rs"));
}

pub use channel::{
    Outstanding, RankChannel, RankReport, RankServer, SHARED_STORAGE_CHANNEL, SOCKET_CHANNEL, Wake,
};
pub use iceoryx::{
    ClientEndpoint, DEFAULT_SERVICE_PREFIX, EVT_COMPLETION, EVT_DEATH, EVT_REQUEST, EVT_RESULT,
    Frame, Header, IPC_VERSION, IpcError, IpcResult, Pending, ServerEndpoint, WakeEvents,
    WakeSender, header_for_request, header_for_response, is_supported_ipc_version, service_name,
};
pub use socket::{SocketClient, SocketServer};

// Message types are split by concern and re-exported flat at the crate root;
// each module takes the shared imports through `use super::*` and the
// validation macros through textual scope.
mod call;
mod config;
mod info;
mod request;
mod tensor;

pub use call::*;
pub use config::LaneConfig;
pub use info::*;
pub use request::*;
pub use tensor::*;
#[cfg(test)]
mod tests;
