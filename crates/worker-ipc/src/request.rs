//! Worker request and response envelopes.

use super::*;

/// Kind discriminator carried by each worker request frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    /// Queries worker identity and capabilities.
    Info,
    /// Submits a batch.
    Submit,
    /// Requests orderly worker shutdown.
    Close,
}

impl RequestKind {
    /// Request kinds accepted by the worker endpoint.
    pub const ALL: [Self; 3] = [Self::Info, Self::Submit, Self::Close];

    /// Returns the stable wire name for this request kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Submit => "submit",
            Self::Close => "close",
        }
    }
}

/// Host-to-worker request envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerRequest {
    /// Queries worker identity and capabilities.
    Info {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
    },
    /// Submits a batch.
    Submit {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// Fully lowered batch.
        batch: Batch,
    },
    /// Requests orderly worker shutdown.
    Close {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
    },
}

impl WorkerRequest {
    /// Returns this envelope's request-kind discriminator.
    pub const fn kind(&self) -> RequestKind {
        match self {
            Self::Info { .. } => RequestKind::Info,
            Self::Submit { .. } => RequestKind::Submit,
            Self::Close { .. } => RequestKind::Close,
        }
    }

    /// Returns the optional call correlation identifier.
    pub const fn message_id(&self) -> Option<u64> {
        match self {
            Self::Info { message_id }
            | Self::Submit { message_id, .. }
            | Self::Close { message_id } => *message_id,
        }
    }

    /// Replaces the call correlation identifier.
    pub fn set_call_id(&mut self, value: Option<u64>) {
        match self {
            Self::Info { message_id }
            | Self::Submit { message_id, .. }
            | Self::Close { message_id } => *message_id = value,
        }
    }

    /// Returns the batch carried by this request, if any.
    pub const fn batch(&self) -> Option<&Batch> {
        match self {
            Self::Submit { batch, .. } => Some(batch),
            _ => None,
        }
    }

    /// Constructs a worker-capability request.
    pub fn info() -> Self {
        Self::Info { message_id: None }
    }
    /// Constructs a batch-submission request.
    pub fn submit(batch: Batch) -> Self {
        Self::Submit {
            message_id: None,
            batch,
        }
    }
    /// Constructs a worker-shutdown request.
    pub fn close() -> Self {
        Self::Close { message_id: None }
    }
}

/// Kind discriminator carried by each worker response frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseKind {
    /// Worker capability response.
    Info,
    /// Partial or terminal run result.
    Result,
    /// Successful control acknowledgement.
    Ok,
    /// Structured worker failure.
    Error,
}

/// Request and call identity attached to a worker error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorCallIdentity {
    /// Request lineage owning the failed call.
    pub request_key: RequestKey,
    /// Failed call identity.
    pub call_id: CallId,
}

/// Structured worker failure with optional call identities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerResponseError {
    /// Human-readable failure description.
    pub message: String,
    /// Optional stable machine-readable error code.
    pub code: Option<String>,
    /// Whether the worker is unsafe to serve further requests.
    pub fatal: bool,
    /// Optional worker phase that failed.
    pub phase: Option<String>,
    /// Optional execution route that failed.
    pub route: Option<String>,
    /// Calls affected by the failure.
    pub calls: Vec<ErrorCallIdentity>,
}

/// Worker-to-host response envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerResponse {
    /// Returns worker identity and capabilities.
    Info {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// Loaded worker capabilities.
        info: WorkerInfo,
    },
    /// Returns partial or terminal run progress.
    Result {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// Physical run result.
        result: BatchOutput,
    },
    /// Acknowledges a successful control request.
    Ok {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
    },
    /// Returns a structured worker failure.
    Error {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// Structured failure details.
        error: WorkerResponseError,
    },
}

impl WorkerResponse {
    /// Returns this envelope's response-kind discriminator.
    pub const fn kind(&self) -> ResponseKind {
        match self {
            Self::Info { .. } => ResponseKind::Info,
            Self::Result { .. } => ResponseKind::Result,
            Self::Ok { .. } => ResponseKind::Ok,
            Self::Error { .. } => ResponseKind::Error,
        }
    }

    /// Returns the optional call correlation identifier.
    pub const fn message_id(&self) -> Option<u64> {
        match self {
            Self::Info { message_id, .. }
            | Self::Result { message_id, .. }
            | Self::Ok { message_id }
            | Self::Error { message_id, .. } => *message_id,
        }
    }

    /// Replaces the call correlation identifier.
    pub fn set_call_id(&mut self, value: Option<u64>) {
        match self {
            Self::Info { message_id, .. }
            | Self::Result { message_id, .. }
            | Self::Ok { message_id }
            | Self::Error { message_id, .. } => *message_id = value,
        }
    }

    /// Returns the run result carried by this response, if any.
    pub const fn report(&self) -> Option<&BatchOutput> {
        match self {
            Self::Result { result, .. } => Some(result),
            _ => None,
        }
    }

    /// Constructs a worker-capability response.
    pub fn info(info: WorkerInfo) -> Self {
        Self::Info {
            message_id: None,
            info,
        }
    }

    /// Constructs a run-result response.
    pub fn result(result: BatchOutput) -> Self {
        Self::Result {
            message_id: None,
            result,
        }
    }

    /// Constructs an acknowledgment response.
    pub fn ok() -> Self {
        Self::Ok { message_id: None }
    }

    /// Constructs a structured worker-error response.
    pub fn error(error: WorkerResponseError) -> Self {
        Self::Error {
            message_id: None,
            error,
        }
    }
}
