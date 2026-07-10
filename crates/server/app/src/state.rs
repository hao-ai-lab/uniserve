use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;
use tokio::time::{Duration, Instant, sleep_until};
use tracing::warn;
use uniserve_engine_gateway::EngineAppControl;
use uniserve_engine_gateway::transport::protocol::lora::LoraRequest;
use uniserve_serving::ServingRuntime;

use crate::lora::{LoadLoraError, LoraManager, LoraModelResolution, UnloadLoraError};
use uniserve_model_profile::dialect::GenerationDialectProfile;

use crate::server_info::{ServerInfoConfigFormat, ServerInfoSnapshot};

const SHUTDOWN_REFCOUNT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Shared router state for the minimal single-model OpenAI server.
pub struct AppState {
    /// All public model IDs served by this frontend. The first entry is the
    /// primary ID used in responses; all entries are valid in requests.
    served_model_names: Vec<String>,
    /// Canonical semantic serving runtime used by inference requests.
    runtime: ServingRuntime,
    /// Application-owned engine lifecycle and administration capability.
    engine_control: EngineAppControl,
    /// Whether to log a summary line for each completed request.
    enable_log_requests: bool,
    /// Whether to set X-Request-Id on every HTTP response.
    enable_request_id_headers: bool,
    /// Bearer token accepted by public serving routes.
    api_key: Option<String>,
    /// Bearer token accepted by sensitive management routes.
    admin_api_key: Option<String>,
    /// Optional per-request wall-clock timeout.
    request_timeout: Option<Duration>,
    /// Optional front-door HTTP admission limit for in-flight inference
    /// requests.
    max_concurrent_requests: Option<u64>,
    /// Whether development-only management routes are mounted.
    server_dev_mode: bool,
    /// Whether runtime LoRA management routes are mounted.
    runtime_lora_updating: bool,
    /// Absolute path prefixes allowed for runtime LoRA adapter loading.
    runtime_lora_allowed_path_prefixes: Vec<PathBuf>,
    /// Runtime server information returned by `/server_info`, when available.
    server_info: Option<ServerInfoSnapshot>,
    /// Number of in-flight inference requests currently owned by this frontend.
    server_load: AtomicU64,
    /// Dynamic LoRA adapter registry.
    lora_manager: LoraManager,
}

impl AppState {
    /// Construct one application state instance.
    ///
    /// `served_model_names` must be non-empty; the first entry is the primary
    /// model ID returned in API responses.
    ///
    /// # Panics
    ///
    /// Panics if `served_model_names` is empty.
    pub fn new(
        served_model_names: Vec<String>,
        runtime: ServingRuntime,
        engine_control: EngineAppControl,
    ) -> Self {
        assert!(
            !served_model_names.is_empty(),
            "served_model_names must not be empty"
        );
        Self {
            served_model_names,
            runtime,
            engine_control,
            enable_log_requests: false,
            enable_request_id_headers: false,
            api_key: None,
            admin_api_key: None,
            request_timeout: None,
            max_concurrent_requests: None,
            server_dev_mode: false,
            runtime_lora_updating: false,
            runtime_lora_allowed_path_prefixes: Vec::new(),
            server_info: None,
            server_load: AtomicU64::new(0),
            lora_manager: LoraManager::new(),
        }
    }

    /// Attach a generation dialect to an embedded/test runtime profile.
    pub fn with_generation_dialect(mut self, profile: GenerationDialectProfile) -> Self {
        self.runtime = self.runtime.with_generation_dialect(profile);
        self
    }

    /// Enable per-request completion logging.
    pub fn with_log_requests(mut self, enabled: bool) -> Self {
        self.enable_log_requests = enabled;
        self
    }

    /// Enable X-Request-Id response headers.
    pub fn with_request_id_headers(mut self, enabled: bool) -> Self {
        self.enable_request_id_headers = enabled;
        self
    }

    /// Configure the public serving API bearer token.
    pub fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key;
        self
    }

    /// Configure the sensitive management route bearer token.
    pub fn with_admin_api_key(mut self, admin_api_key: Option<String>) -> Self {
        self.admin_api_key = admin_api_key;
        self
    }

    /// Configure the per-request wall-clock timeout.
    pub fn with_request_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Configure the front-door HTTP admission limit.
    pub fn with_max_concurrent_requests(mut self, limit: Option<u64>) -> Self {
        self.max_concurrent_requests = limit;
        self
    }

    /// Configure development-only management route mounting.
    pub fn with_server_dev_mode(mut self, enabled: bool) -> Self {
        self.server_dev_mode = enabled;
        self
    }

    /// Configure runtime LoRA management route mounting.
    pub fn with_runtime_lora_updating(mut self, enabled: bool) -> Self {
        self.runtime_lora_updating = enabled;
        self
    }

    /// Configure allowed local path prefixes for runtime LoRA adapter loading.
    pub fn with_runtime_lora_allowed_path_prefixes(mut self, prefixes: Vec<PathBuf>) -> Self {
        self.runtime_lora_allowed_path_prefixes = prefixes;
        self
    }

    /// Attach the runtime server information snapshot used by `/server_info`.
    pub fn with_server_info(mut self, server_info: ServerInfoSnapshot) -> Self {
        self.server_info = Some(server_info);
        self
    }

    /// Canonical semantic runtime used by protocol adapters.
    pub fn runtime(&self) -> &ServingRuntime {
        &self.runtime
    }

    /// Whether request completion summaries should be logged.
    pub fn enable_log_requests(&self) -> bool {
        self.enable_log_requests
    }

    /// Whether HTTP responses should include X-Request-Id.
    pub fn enable_request_id_headers(&self) -> bool {
        self.enable_request_id_headers
    }

    /// Public serving API bearer token, when configured.
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    /// Sensitive management route bearer token, when configured.
    pub fn admin_api_key(&self) -> Option<&str> {
        self.admin_api_key.as_deref()
    }

    /// Per-request wall-clock timeout, when configured.
    pub fn request_timeout(&self) -> Option<Duration> {
        self.request_timeout
    }

    /// Front-door HTTP admission limit, when configured.
    pub fn max_concurrent_requests(&self) -> Option<u64> {
        self.max_concurrent_requests
    }

    /// Whether development-only management routes are mounted.
    pub fn server_dev_mode(&self) -> bool {
        self.server_dev_mode
    }

    /// Whether runtime LoRA management routes are mounted.
    pub fn runtime_lora_updating_enabled(&self) -> bool {
        self.runtime_lora_updating
    }

    /// Allowed local path prefixes for runtime LoRA adapter loading.
    pub fn runtime_lora_allowed_path_prefixes(&self) -> &[PathBuf] {
        &self.runtime_lora_allowed_path_prefixes
    }

    /// Build a `/server_info` response payload.
    pub fn server_info_response(&self, config_format: ServerInfoConfigFormat) -> Option<Value> {
        self.server_info
            .as_ref()
            .map(|server_info| server_info.response(config_format))
    }

    /// The primary model name echoed back in API responses (the first served
    /// name).
    pub fn primary_model_name(&self) -> &str {
        self.served_model_names
            .first()
            .map(String::as_str)
            .unwrap_or_default()
    }

    /// All model names served by this frontend.
    pub fn served_model_names(&self) -> &[String] {
        &self.served_model_names
    }

    /// Return base served model names plus dynamically loaded LoRA adapter
    /// names.
    pub async fn served_model_names_with_loras(&self) -> Vec<String> {
        self.lora_manager
            .served_model_names(&self.served_model_names)
            .await
    }

    /// Resolve the requested model against one dynamic LoRA registry snapshot.
    pub async fn resolve_model_with_loras(&self, model_name: Option<&str>) -> LoraModelResolution {
        self.lora_manager
            .resolve_model(&self.served_model_names, model_name)
            .await
    }

    /// Load one dynamic LoRA adapter and register it as a public model name.
    pub async fn load_lora(
        &self,
        lora_name: String,
        lora_path: String,
        load_inplace: bool,
        is_3d_lora_weight: bool,
    ) -> Result<LoraRequest, LoadLoraError> {
        self.lora_manager
            .load_lora(
                self.engine_control(),
                &self.served_model_names,
                lora_name,
                lora_path,
                load_inplace,
                is_3d_lora_weight,
            )
            .await
    }

    /// Remove one dynamic LoRA adapter from the engine and public model
    /// registry.
    pub async fn unload_lora(
        &self,
        lora_name: &str,
        lora_int_id: Option<u64>,
    ) -> Result<LoraRequest, UnloadLoraError> {
        self.lora_manager
            .unload_lora(self.engine_control(), lora_name, lora_int_id)
            .await
    }

    /// Typed engine lifecycle and administration capability.
    pub fn engine_control(&self) -> &EngineAppControl {
        &self.engine_control
    }

    /// Return the current in-flight inference request count for the `/load`
    /// endpoint.
    pub fn server_load(&self) -> u64 {
        self.server_load.load(Ordering::Relaxed)
    }

    /// Increment the in-flight inference request count, called by the load
    /// tracking middleware.
    pub fn increment_server_load(&self) {
        self.server_load.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the in-flight inference request count, called by the load
    /// tracking middleware.
    pub fn decrement_server_load(&self) {
        self.server_load.fetch_sub(1, Ordering::Relaxed);
    }

    /// Wait until all request-owned references are dropped, then shut down the
    /// engine client.
    ///
    /// If the deadline elapses while request/connection tasks still hold state
    /// references, skip the clean engine-client shutdown and let process
    /// teardown reclaim the remaining resources.
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
