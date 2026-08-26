use std::sync::{Arc, Weak};

use super::in_process::EngineClient;
use crate::engine_client::error::{Error, Result};

/// The reason a request stream is being cancelled when its output stream is
/// dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamCancelCause {
    /// The consumer dropped the stream before the request reached a terminal
    /// engine output.
    #[default]
    DroppedStream,
    /// The frontend matched a stop string locally and intentionally stopped
    /// consuming the stream.
    StopStringMatched,
}

task_local::task_local! {
    static STREAM_CANCEL_CAUSE: StreamCancelCause;
}

impl StreamCancelCause {
    /// Return the cancellation cause currently associated with this task, or
    /// [`StreamCancelCause::DroppedStream`] by default.
    pub fn current() -> Self {
        STREAM_CANCEL_CAUSE.try_get().unwrap_or_default()
    }

    /// Drop one value while marking the drop as happening for this cancellation cause.
    pub fn drop_as<T>(self, value: T) {
        STREAM_CANCEL_CAUSE.sync_scope(self, move || drop(value));
    }
}

/// Request-scoped semantic control emitted by a frontend output stream.
#[derive(Debug, Clone)]
pub enum StreamControl {
    Cancel {
        cause: StreamCancelCause,
        output_token_count: usize,
    },
    Acknowledge {
        output_token_count: usize,
    },
}

/// Stream control work item sent to the active backend.
#[derive(Debug, Clone)]
pub struct StreamControlRequest {
    pub request_id: String,
    pub control: StreamControl,
}

/// Owned, typed health and build-provenance capability for an engine connection.
///
/// The weak reference keeps this capability independent of execution
/// ownership and deliberately exposes no generation submission capability.
#[derive(Clone)]
pub struct EngineStatus {
    client: Weak<EngineClient>,
}

impl EngineStatus {
    pub(crate) fn new(client: &Arc<EngineClient>) -> Self {
        Self {
            client: Arc::downgrade(client),
        }
    }

    fn client(&self) -> Result<Arc<EngineClient>> {
        self.client.upgrade().ok_or(Error::StatusUnavailable)
    }

    pub fn version(&self) -> Result<String> {
        Ok(self.client()?.uniserve_version().to_string())
    }

    pub fn is_healthy(&self) -> bool {
        self.client
            .upgrade()
            .is_some_and(|client| client.is_healthy())
    }

    pub fn health_error(&self) -> Option<Arc<Error>> {
        self.client
            .upgrade()
            .and_then(|client| client.health_error())
    }
}
