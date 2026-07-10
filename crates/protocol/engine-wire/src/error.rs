use thiserror::Error;
use thiserror_ext::Macro;

use crate::utility::UtilityCallId;

pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced while encoding/decoding the engine wire protocol.
///
/// Transport- and client-lifecycle errors (handshake timeouts, dead engines,
/// closed registries) live with the code that owns sockets and registries; this
/// crate only reports protocol-shape problems.
#[derive(Debug, Error, Macro)]
pub enum Error {
    #[error("messagepack encode failed for {target_type}: {message}")]
    Encode {
        target_type: &'static str,
        message: String,
    },
    #[error("messagepack decode failed for {target_type}: {message}")]
    Decode {
        target_type: &'static str,
        message: String,
    },
    #[error("messagepack value decode failed")]
    ValueDecode(#[from] rmpv::decode::Error),
    #[error("messagepack ext value decode failed: {message}")]
    ExtValueDecode { message: String },
    #[error("unsupported auxiliary frame(s): expected 1 frame, got {frame_count}")]
    UnsupportedAuxFrames { frame_count: usize },
    #[error("invalid canonical generation request: {message}")]
    InvalidGenerationRequest { message: String },
    #[error("utility call `{method}` (id {call_id}) failed: {message}")]
    UtilityCallFailed {
        method: String,
        call_id: UtilityCallId,
        message: String,
    },
    #[error("utility call `{method}` (id {call_id}) result decode failed: {message}")]
    UtilityResultDecode {
        method: String,
        call_id: UtilityCallId,
        message: String,
    },
}
