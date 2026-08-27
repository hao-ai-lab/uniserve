//! Worker execution protocol.
//!
//! The scheduler and worker exchange four cross-layer records — [`Operation`],
//! [`VersionRef`], [`ProductRef`], and [`ModelOutput`] — plus a request
//! [`Control`] command. Every operation names one closed [`ForwardMode`] variant, one
//! exact parent version, and its declared input and output products. The worker
//! returns exactly one [`ModelOutput`] per operation. Two host-computed
//! digests fix identity: an operation [`Operation::plan_digest`] over immutable
//! registration fields, and a [`ModelOutput::compute_semantic_digest`] over
//! the selected result.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use uniserve_core::{
    BlockId, GenerationRuntimeCapabilities, ImageParams, KvCacheDtype, KvCacheGroupSpec,
    ModelDtype, RankInfo, RequestId, SamplingParams,
};
pub use uniserve_core::{Digest, OpId, WorkerForwardStats};

pub type ProtocolResult<T> = std::result::Result<T, ProtocolError>;

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("worker protocol violation: {0}")]
    Violation(String),
    #[error(transparent)]
    Sampling(#[from] uniserve_core::SamplingParamsError),
    #[error(transparent)]
    Image(#[from] uniserve_core::ImageParamsError),
}

impl ProtocolError {
    pub(crate) fn message(message: impl Into<String>) -> Self {
        Self::Violation(message.into())
    }
}

macro_rules! protocol_error {
    ($($arg:tt)*) => {
        ProtocolError::message(format!($($arg)*))
    };
}

macro_rules! protocol_bail {
    ($($arg:tt)*) => {
        return Err(protocol_error!($($arg)*))
    };
}

macro_rules! protocol_ensure {
    ($condition:expr, $($arg:tt)*) => {
        if !$condition {
            protocol_bail!($($arg)*);
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
mod digest;
mod operation;
mod product;
mod request;

pub use capabilities::*;
pub(crate) use digest::{CanonicalDigest, is_digest};
pub use operation::*;
pub use product::*;
pub use request::*;
#[cfg(test)]
mod tests;
