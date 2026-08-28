//! Versioned scheduler-to-worker IPC and FlatBuffers transport types.
//!
//! A [`NewRequest`] carries static request state once, [`Batch`] carries planned
//! operations, and [`CompletionReport`] returns resolved outputs. [`WorkerInfo`]
//! is the post-load handshake. Runtime identity is numeric and stable across
//! serialization: request epoch, operation id, point index, and generation.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use uniserve_core::{
    BlockId, GenerationLimits, ImageParams, KvCacheDtype, KvCacheGroup, ModelDtype, RankInfo,
    RequestId, SamplingParams,
};
pub use uniserve_core::{OpId, WorkerForwardStats};

pub type ValidationResult<T> = std::result::Result<T, ValidationError>;

#[derive(Debug, thiserror::Error)]
#[error("invalid worker message: {0}")]
pub struct ValidationError(String);

impl ValidationError {
    pub(crate) fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl From<uniserve_core::SamplingParamsError> for ValidationError {
    fn from(error: uniserve_core::SamplingParamsError) -> Self {
        Self::message(error.to_string())
    }
}

impl From<uniserve_core::ImageParamsError> for ValidationError {
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

pub mod codec;
pub mod iceoryx;
mod resources;
#[allow(warnings)]
pub mod schema {
    include!(concat!(env!("OUT_DIR"), "/flatbuffers/mod.rs"));
}

pub use iceoryx::{
    ClientEndpoint, DEFAULT_SERVICE_PREFIX, EVT_COMMAND, EVT_COMPLETION, EVT_DEATH, EVT_REQUEST,
    EVT_RESULT, Frame, Header, IPC_VERSION, IpcError, IpcResult, Pending, ServerEndpoint,
    WakeEvents, WakeSender, header_for_request, header_for_response, is_supported_ipc_version,
    service_name,
};
pub use resources::{ResourceClass, ResourcePressure};

mod info;
mod operation;
mod product;
mod request;

pub use info::*;
pub use operation::*;
pub use product::*;
pub use request::*;
#[cfg(test)]
mod tests;
