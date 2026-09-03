//! Four-operation worker request/response framing.

use super::*;

// ---------------------------------------------------------------------------
// Worker request and response framing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    Info,
    Submit,
    Poll,
    Close,
}

impl RequestKind {
    pub const ALL: [Self; 4] = [Self::Info, Self::Submit, Self::Poll, Self::Close];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Submit => "submit",
            Self::Poll => "poll",
            Self::Close => "close",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerRequest {
    Info { call_id: Option<u64> },
    Submit { call_id: Option<u64>, run: Run },
    Poll { call_id: Option<u64>, run_id: u64 },
    Close { call_id: Option<u64> },
}

impl WorkerRequest {
    pub const fn kind(&self) -> RequestKind {
        match self {
            Self::Info { .. } => RequestKind::Info,
            Self::Submit { .. } => RequestKind::Submit,
            Self::Poll { .. } => RequestKind::Poll,
            Self::Close { .. } => RequestKind::Close,
        }
    }

    pub const fn call_id(&self) -> Option<u64> {
        match self {
            Self::Info { call_id }
            | Self::Submit { call_id, .. }
            | Self::Poll { call_id, .. }
            | Self::Close { call_id } => *call_id,
        }
    }

    pub fn set_call_id(&mut self, value: Option<u64>) {
        match self {
            Self::Info { call_id }
            | Self::Submit { call_id, .. }
            | Self::Poll { call_id, .. }
            | Self::Close { call_id } => *call_id = value,
        }
    }

    pub const fn run(&self) -> Option<&Run> {
        match self {
            Self::Submit { run, .. } => Some(run),
            _ => None,
        }
    }

    pub fn info() -> Self {
        Self::Info { call_id: None }
    }
    pub fn submit(run: Run) -> Self {
        Self::Submit { call_id: None, run }
    }
    pub fn poll(run_id: u64) -> Self {
        Self::Poll {
            call_id: None,
            run_id,
        }
    }
    pub fn close() -> Self {
        Self::Close { call_id: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseKind {
    Info,
    Result,
    Ok,
    Error,
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
        result: RunResult,
    },
    Ok {
        call_id: Option<u64>,
    },
    Error {
        call_id: Option<u64>,
        error: WorkerResponseError,
    },
}

impl WorkerResponse {
    pub const fn kind(&self) -> ResponseKind {
        match self {
            Self::Info { .. } => ResponseKind::Info,
            Self::Result { .. } => ResponseKind::Result,
            Self::Ok { .. } => ResponseKind::Ok,
            Self::Error { .. } => ResponseKind::Error,
        }
    }

    pub const fn call_id(&self) -> Option<u64> {
        match self {
            Self::Info { call_id, .. }
            | Self::Result { call_id, .. }
            | Self::Ok { call_id }
            | Self::Error { call_id, .. } => *call_id,
        }
    }

    pub fn set_call_id(&mut self, value: Option<u64>) {
        match self {
            Self::Info { call_id, .. }
            | Self::Result { call_id, .. }
            | Self::Ok { call_id }
            | Self::Error { call_id, .. } => *call_id = value,
        }
    }

    pub const fn report(&self) -> Option<&RunResult> {
        match self {
            Self::Result { result, .. } => Some(result),
            _ => None,
        }
    }

    pub fn info(info: WorkerInfo) -> Self {
        Self::Info {
            call_id: None,
            info,
        }
    }

    pub fn result(result: RunResult) -> Self {
        Self::Result {
            call_id: None,
            result,
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
