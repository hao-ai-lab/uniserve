use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;
use tokio::time::{Duration, Instant, sleep_until};
use tracing::warn;
use uniserve_chat::ChatLlm;
use uniserve_engine_client::EngineCoreClient;
use uniserve_engine_client::protocol::lora::LoraRequest;

use crate::lora::{LoadLoraError, LoraManager, LoraModelResolution, UnloadLoraError};
use uniserve_native_api::NativeModelProfile;

use crate::server_info::{ServerInfoConfigFormat, ServerInfoSnapshot};

const SHUTDOWN_REFCOUNT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Shared router state for the minimal single-model OpenAI server.
pub struct AppState {
 /// All public model IDs served by this frontend. The first entry is the
 /// primary ID used in responses; all entries are valid in requests.
    served_model_names: Vec<String>,
 /// Shared chat facade used by all requests.
    chat: ChatLlm,
 /// Whether to log a summary line for each completed request.
    enable_log_requests: bool,
 /// Whether to set X-Request-Id on every HTTP response.
    enable_request_id_headers: bool,
 /// Runtime server information returned by `/server_info`, when available.
    server_info: Option<ServerInfoSnapshot>,
 /// Number of in-flight inference requests currently owned by this frontend.
    server_load: AtomicU64,
 /// Dynamic LoRA adapter registry.
    lora_manager: LoraManager,
 /// Model-family profile for the native image/interleave surface.
    native_profile: NativeModelProfile,
}

impl AppState {
 /// Construct one application state instance.

 /// `served_model_names` must be non-empty; the first entry is the primary
 /// model ID returned in API responses.

 /// # Panics

 /// Panics if `served_model_names` is empty.
    pub fn new(served_model_names: Vec<String>, chat: ChatLlm) -> Self {
        assert!(
            !served_model_names.is_empty(),
            "served_model_names must not be empty"
        );
        Self {
            served_model_names,
            chat,
            enable_log_requests: false,
            enable_request_id_headers: false,
            server_info: None,
            server_load: AtomicU64::new(0),
            lora_manager: LoraManager::new(),
            native_profile: NativeModelProfile::default(),
        }
    }

 /// Attach the native image/interleave profile resolved from the model
 /// tokenizer.
    pub fn with_native_profile(mut self, profile: NativeModelProfile) -> Self {
        self.native_profile = profile;
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

 /// Attach the runtime server information snapshot used by `/server_info`.
    pub fn with_server_info(mut self, server_info: ServerInfoSnapshot) -> Self {
        self.server_info = Some(server_info);
        self
    }

 /// Shared chat facade used by HTTP and gRPC requests.
    pub fn chat(&self) -> &ChatLlm {
        &self.chat
    }

 /// Return the model-family profile for the native image/interleave surface.
    pub fn native_profile(&self) -> &NativeModelProfile {
        &self.native_profile
    }

 /// Whether request completion summaries should be logged.
    pub fn enable_log_requests(&self) -> bool {
        self.enable_log_requests
    }

 /// Whether HTTP responses should include X-Request-Id.
    pub fn enable_request_id_headers(&self) -> bool {
        self.enable_request_id_headers
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
                self.uniserve_engine_client(),
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
            .unload_lora(self.uniserve_engine_client(), lora_name, lora_int_id)
            .await
    }

 /// Return a reference to the underlying engine core client for utility
 /// calls.
    pub fn uniserve_engine_client(&self) -> &EngineCoreClient {
        self.chat.uniserve_engine_client()
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

 /// If the deadline elapses while request/connection tasks still hold state
 /// references, skip the clean engine-client shutdown and let process
 /// teardown reclaim the remaining resources.
    pub async fn shutdown(mut self: Arc<Self>, deadline: Instant) -> anyhow::Result<()> {
        loop {
            match Arc::try_unwrap(self) {
                Ok(state) => {
                    state.chat.shutdown().await?;
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
