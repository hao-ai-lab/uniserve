//! Administrative request/response framing around batches and worker information.

use super::*;

// ---------------------------------------------------------------------------
// Administrative request and response framing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    GetInfo,
    Execute,
    PollCompletions,
    DropSession,
    Shutdown,
    ReleaseProducts,
    GetPressure,
}

impl RequestKind {
    pub const ALL: [Self; 7] = [
        Self::GetInfo,
        Self::Execute,
        Self::PollCompletions,
        Self::DropSession,
        Self::Shutdown,
        Self::ReleaseProducts,
        Self::GetPressure,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GetInfo => "get_info",
            Self::Execute => "execute",
            Self::PollCompletions => "poll_completions",
            Self::DropSession => "drop_session",
            Self::Shutdown => "shutdown",
            Self::ReleaseProducts => "release_products",
            Self::GetPressure => "get_pressure",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerRequest {
    GetInfo { call_id: Option<u64> },
    Execute { call_id: Option<u64>, batch: Batch },
    PollCompletions { call_id: Option<u64>, step_id: u64 },
    DropSession { session_id: RequestId },
    ReleaseProducts { product_handles: Vec<u64> },
    GetPressure { call_id: Option<u64> },
    Shutdown,
}

impl WorkerRequest {
    pub const fn kind(&self) -> RequestKind {
        match self {
            Self::GetInfo { .. } => RequestKind::GetInfo,
            Self::Execute { .. } => RequestKind::Execute,
            Self::PollCompletions { .. } => RequestKind::PollCompletions,
            Self::DropSession { .. } => RequestKind::DropSession,
            Self::ReleaseProducts { .. } => RequestKind::ReleaseProducts,
            Self::GetPressure { .. } => RequestKind::GetPressure,
            Self::Shutdown => RequestKind::Shutdown,
        }
    }

    pub const fn call_id(&self) -> Option<u64> {
        match self {
            Self::GetInfo { call_id }
            | Self::Execute { call_id, .. }
            | Self::PollCompletions { call_id, .. }
            | Self::GetPressure { call_id } => *call_id,
            Self::DropSession { .. } | Self::ReleaseProducts { .. } | Self::Shutdown => None,
        }
    }

    pub fn set_call_id(&mut self, value: Option<u64>) {
        match self {
            Self::GetInfo { call_id }
            | Self::Execute { call_id, .. }
            | Self::PollCompletions { call_id, .. }
            | Self::GetPressure { call_id } => *call_id = value,
            _ => {}
        }
    }

    pub const fn batch(&self) -> Option<&Batch> {
        match self {
            Self::Execute { batch, .. } => Some(batch),
            _ => None,
        }
    }

    pub fn get_info() -> Self {
        Self::GetInfo { call_id: None }
    }
    pub fn execute(batch: Batch) -> Self {
        Self::Execute {
            call_id: None,
            batch,
        }
    }
    pub fn poll_completions(step_id: u64) -> Self {
        Self::PollCompletions {
            call_id: None,
            step_id,
        }
    }
    pub fn drop_session(session_id: RequestId) -> Self {
        Self::DropSession { session_id }
    }
    pub fn shutdown() -> Self {
        Self::Shutdown
    }
    pub fn release_products(product_handles: Vec<u64>) -> Self {
        Self::ReleaseProducts { product_handles }
    }
    pub fn get_pressure() -> Self {
        Self::GetPressure { call_id: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseKind {
    Info,
    Result,
    Ok,
    Error,
    Pressure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorOperationIdentity {
    pub request_key: RequestKey,
    pub op_id: OpId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerResponseError {
    pub message: String,
    pub code: Option<String>,
    pub retryable: bool,
    pub fatal: bool,
    pub phase: Option<String>,
    pub route: Option<String>,
    pub operations: Vec<ErrorOperationIdentity>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerResponse {
    Info {
        call_id: Option<u64>,
        info: WorkerInfo,
    },
    Result {
        call_id: Option<u64>,
        completion_report: CompletionReport,
    },
    Ok {
        call_id: Option<u64>,
    },
    Error {
        call_id: Option<u64>,
        error: WorkerResponseError,
    },
    Pressure {
        call_id: Option<u64>,
        pressure: Vec<ResourcePressure>,
    },
}

impl WorkerResponse {
    pub const fn kind(&self) -> ResponseKind {
        match self {
            Self::Info { .. } => ResponseKind::Info,
            Self::Result { .. } => ResponseKind::Result,
            Self::Ok { .. } => ResponseKind::Ok,
            Self::Error { .. } => ResponseKind::Error,
            Self::Pressure { .. } => ResponseKind::Pressure,
        }
    }

    pub const fn call_id(&self) -> Option<u64> {
        match self {
            Self::Info { call_id, .. }
            | Self::Result { call_id, .. }
            | Self::Ok { call_id }
            | Self::Error { call_id, .. }
            | Self::Pressure { call_id, .. } => *call_id,
        }
    }

    pub fn set_call_id(&mut self, value: Option<u64>) {
        match self {
            Self::Info { call_id, .. }
            | Self::Result { call_id, .. }
            | Self::Ok { call_id }
            | Self::Error { call_id, .. }
            | Self::Pressure { call_id, .. } => *call_id = value,
        }
    }

    pub const fn report(&self) -> Option<&CompletionReport> {
        match self {
            Self::Result {
                completion_report, ..
            } => Some(completion_report),
            _ => None,
        }
    }

    pub fn info(info: WorkerInfo) -> Self {
        Self::Info {
            call_id: None,
            info,
        }
    }

    pub fn completion_report(completion_report: CompletionReport) -> Self {
        Self::Result {
            call_id: None,
            completion_report,
        }
    }

    pub fn ok() -> Self {
        Self::Ok { call_id: None }
    }

    pub fn error(error: WorkerResponseError) -> Self {
        Self::Error {
            call_id: None,
            error,
        }
    }
}
