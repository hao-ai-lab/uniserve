use thiserror::Error;
use thiserror_ext::Macro;

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
    #[error("invalid canonical generation request: {message}")]
    InvalidGenerationRequest { message: String },
}
