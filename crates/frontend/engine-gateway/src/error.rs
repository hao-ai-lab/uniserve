use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;
use thiserror_ext::Macro;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the engine client.
///
/// Protocol-shape errors live in [`uniserve_engine_wire::Error`] and arrive here through the
/// transparent `Proto` variant; this enum owns transport- and client-lifecycle
/// failures.
#[derive(Debug, Error, Macro)]
pub enum Error {
    #[error(transparent)]
    Proto(#[from] uniserve_engine_wire::Error),
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("zmq transport error")]
    Transport(#[from] zeromq::ZmqError),
    #[error("engine core reported fatal failure")]
    EngineCoreDead,
    #[error("startup handshake timed out while waiting for {stage} after {timeout:?}")]
    HandshakeTimeout {
        stage: &'static str,
        timeout: Duration,
    },
    #[error("engine input registration timed out after {timeout:?}")]
    InputRegistrationTimeout { timeout: Duration },
    #[error("unexpected engine id in startup handshake: expected {expected:?}, got {actual:?}")]
    UnexpectedHandshakeIdentity { expected: Vec<u8>, actual: Vec<u8> },
    #[error("unexpected startup handshake message: {message}")]
    UnexpectedHandshakeMessage { message: String },
    #[error("unexpected output on main dispatcher path: {message}")]
    UnexpectedDispatcherOutput { message: String },
    #[error("engine control channel closed unexpectedly: {message}")]
    ControlClosed { message: String },
    #[error("request `{request_id}` is already in flight")]
    DuplicateRequestId { request_id: String },
    #[error("data parallel rank {rank} is out of range for {num_engines} engine(s)")]
    InvalidDataParallelRank { rank: u32, num_engines: u32 },
    #[error("engine event dispatcher closed: {message}")]
    DispatcherClosed { message: String },
    #[error("engine client is closed: {message}")]
    ClientClosed { message: String },
    #[error("engine status is unavailable because the execution gateway is closed")]
    StatusUnavailable,

    /// A special variant to allow cloning the same error.
    #[error(transparent)]
    Shared(Arc<Self>),
}
