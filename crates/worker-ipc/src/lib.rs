//! Versioned scheduler-to-worker protocol and FlatBuffers transport types.
//!
//! A [`NewRequest`] carries static request state once, [`Batch`] carries planned
//! operations, and [`CompletionReport`] returns resolved outputs. [`WorkerInfo`]
//! is the post-load handshake. Runtime identity is numeric and stable across
//! serialization: request epoch, operation id, point index, and generation.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use uniserve_core::{
    BlockId, GenerationRuntimeCapabilities, ImageParams, KvCacheDtype, KvCacheGroup, ModelDtype,
    RankInfo, RequestId, SamplingParams,
};
pub use uniserve_core::{OpId, WorkerForwardStats};

pub type ProtocolResult<T> = std::result::Result<T, WireError>;

#[derive(Debug, thiserror::Error)]
#[error("worker protocol violation: {0}")]
pub struct WireError(String);

impl WireError {
    pub(crate) fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl From<uniserve_core::SamplingParamsError> for WireError {
    fn from(error: uniserve_core::SamplingParamsError) -> Self {
        Self::message(error.to_string())
    }
}

impl From<uniserve_core::ImageParamsError> for WireError {
    fn from(error: uniserve_core::ImageParamsError) -> Self {
        Self::message(error.to_string())
    }
}

macro_rules! wire_error {
    ($($arg:tt)*) => {
        WireError::message(format!($($arg)*))
    };
}

macro_rules! wire_bail {
    ($($arg:tt)*) => {
        return Err(wire_error!($($arg)*))
    };
}

macro_rules! wire_ensure {
    ($condition:expr, $($arg:tt)*) => {
        if !$condition {
            wire_bail!($($arg)*);
        }
    };
}

pub mod codec;
pub mod iceoryx;
mod resources;
#[allow(warnings)]
pub mod schema {
    include!(concat!(env!("OUT_DIR"), "/flatbuffers/mod.rs"));
}

pub use iceoryx::{
    ClientEndpoint, DEFAULT_SERVICE_PREFIX, EVT_COMMAND, EVT_COMPLETION, EVT_DEATH, EVT_REQUEST,
    EVT_RESULT, Frame, Header, IpcError, IpcResult, Pending, ServerEndpoint, WIRE_VERSION,
    WakeEvents, WakeSender, header_for_request, header_for_response, is_supported_wire_version,
    service_name,
};
pub use resources::{ResourceClass, ResourcePressure};

mod capabilities;
mod operation;
mod product;
mod request;

pub use capabilities::*;
pub use operation::*;
pub use product::*;
pub use request::*;
#[cfg(test)]
mod tests;
