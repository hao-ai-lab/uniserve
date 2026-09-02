use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::serving::ServingRuntime;
use tokio::time::{Duration, Instant, sleep_until};
use tracing::warn;

const SHUTDOWN_REFCOUNT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Shared router state for one configured model and one generate funnel.
pub struct AppState {
    runtime: ServingRuntime,
    enable_log_requests: bool,
    enable_request_id_headers: bool,
    api_key: Option<String>,
    request_timeout: Option<Duration>,
    max_concurrent_requests: Option<u64>,
    server_load: AtomicU64,
}

impl AppState {
    pub fn new(runtime: ServingRuntime) -> Self {
        Self {
            runtime,
            enable_log_requests: false,
            enable_request_id_headers: false,
            api_key: None,
            request_timeout: None,
            max_concurrent_requests: None,
            server_load: AtomicU64::new(0),
        }
    }

    pub fn with_log_requests(mut self, enabled: bool) -> Self {
        self.enable_log_requests = enabled;
        self
    }

    pub fn with_request_id_headers(mut self, enabled: bool) -> Self {
        self.enable_request_id_headers = enabled;
        self
    }

    pub fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key;
        self
    }

    pub fn with_request_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.request_timeout = timeout;
        self
    }

    pub fn with_max_concurrent_requests(mut self, limit: Option<u64>) -> Self {
        self.max_concurrent_requests = limit;
        self
    }

    pub fn runtime(&self) -> &ServingRuntime {
        &self.runtime
    }

    pub fn enable_log_requests(&self) -> bool {
        self.enable_log_requests
    }

    pub fn enable_request_id_headers(&self) -> bool {
        self.enable_request_id_headers
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    pub fn request_timeout(&self) -> Option<Duration> {
        self.request_timeout
    }

    pub fn max_concurrent_requests(&self) -> Option<u64> {
        self.max_concurrent_requests
    }

    pub fn served_model_name(&self) -> &str {
        self.runtime.served_model_name()
    }

    pub fn engine(&self) -> &crate::engine_client::EngineClient {
        self.runtime.engine()
    }

    pub fn server_load(&self) -> u64 {
        self.server_load.load(Ordering::Relaxed)
    }

    pub fn increment_server_load(&self) {
        self.server_load.fetch_add(1, Ordering::Relaxed);
    }

    pub fn decrement_server_load(&self) {
        self.server_load.fetch_sub(1, Ordering::Relaxed);
    }

    /// Wait until request-owned references are dropped, then shut down the
    /// serving runtime and its engine client.
    pub async fn shutdown(mut self: Arc<Self>, deadline: Instant) -> anyhow::Result<()> {
        loop {
            match Arc::try_unwrap(self) {
                Ok(state) => {
                    state.runtime.shutdown().await?;
                    return Ok(());
                }
                Err(state) => self = state,
            }
            let ref_count = Arc::strong_count(&self);
            let now = Instant::now();
            if now >= deadline {
                warn!(
                    ref_count,
                    "shutdown deadline elapsed before app state became idle; skipping engine-client shutdown"
                );
                return Ok(());
            }
            sleep_until(std::cmp::min(
                deadline,
                now + SHUTDOWN_REFCOUNT_POLL_INTERVAL,
            ))
            .await;
        }
    }
}
