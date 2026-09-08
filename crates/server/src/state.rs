//! Shared application state and graceful engine shutdown coordination.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::serving::ServingRuntime;
use tokio::time::{Duration, Instant, sleep_until};
use tracing::warn;

const SHUTDOWN_REFCOUNT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Shared router state for one configured model and one generate funnel.
pub struct AppState {
    runtime: ServingRuntime,
    pub(crate) videos: Arc<crate::video_jobs::VideoJobs>,
    pub(crate) model_contract: Option<serde_json::Value>,
    enable_log_requests: bool,
    enable_request_id_headers: bool,
    api_key: Option<String>,
    request_timeout: Option<Duration>,
    max_concurrent_requests: Option<u64>,
    server_load: AtomicU64,
}

impl AppState {
    /// Creates application state with default HTTP policies.
    pub fn new(runtime: ServingRuntime) -> Self {
        Self {
            runtime,
            videos: Arc::default(),
            model_contract: None,
            enable_log_requests: false,
            enable_request_id_headers: false,
            api_key: None,
            request_timeout: None,
            max_concurrent_requests: None,
            server_load: AtomicU64::new(0),
        }
    }

    pub(crate) fn with_model_contract(mut self, contract: Option<serde_json::Value>) -> Self {
        self.model_contract = contract;
        self
    }

    /// Configures request lifecycle logging.
    pub fn with_log_requests(mut self, enabled: bool) -> Self {
        self.enable_log_requests = enabled;
        self
    }

    /// Configures request-ID response headers.
    pub fn with_request_id_headers(mut self, enabled: bool) -> Self {
        self.enable_request_id_headers = enabled;
        self
    }

    /// Configures bearer-token authentication.
    pub fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key;
        self
    }

    /// Configures the per-request HTTP timeout.
    pub fn with_request_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Configures the maximum number of concurrent HTTP requests.
    pub fn with_max_concurrent_requests(mut self, limit: Option<u64>) -> Self {
        self.max_concurrent_requests = limit;
        self
    }

    /// Returns the serving runtime.
    pub fn runtime(&self) -> &ServingRuntime {
        &self.runtime
    }

    /// Returns whether request lifecycle logging is enabled.
    pub fn enable_log_requests(&self) -> bool {
        self.enable_log_requests
    }

    /// Returns whether response request-ID headers are enabled.
    pub fn enable_request_id_headers(&self) -> bool {
        self.enable_request_id_headers
    }

    /// Returns the configured API key, if authentication is enabled.
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    /// Returns the configured request timeout.
    pub fn request_timeout(&self) -> Option<Duration> {
        self.request_timeout
    }

    /// Returns the configured concurrent-request limit.
    pub fn max_concurrent_requests(&self) -> Option<u64> {
        self.max_concurrent_requests
    }

    /// Returns the externally served model name.
    pub fn served_model_name(&self) -> &str {
        self.runtime.served_model_name()
    }

    /// Returns the engine client backing the serving runtime.
    pub fn engine(&self) -> &crate::engine_client::EngineClient {
        self.runtime.engine()
    }

    /// Returns the current number of active HTTP requests.
    pub fn server_load(&self) -> u64 {
        self.server_load.load(Ordering::Relaxed)
    }

    /// Increments the active HTTP request count.
    pub fn increment_server_load(&self) {
        self.server_load.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrements the active HTTP request count without underflow.
    pub fn decrement_server_load(&self) {
        self.server_load.fetch_sub(1, Ordering::Relaxed);
    }

    /// Waits until request-owned references are dropped, then shut down the
    /// serving runtime and its engine client.
    pub async fn shutdown(mut self: Arc<Self>, deadline: Instant) -> anyhow::Result<()> {
        self.videos.cancel_all();
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
