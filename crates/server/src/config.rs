use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use crate::profile::ModelDescription;
use crate::serving::chat::ChatTemplateContentFormatOption;
use anyhow::Result;
use serde::Serialize;
use serde_json::Value;
use uniserve_engine::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS, SchedulingPolicy, TransportMap,
    WorkerProcessArgs, WorkerTopology,
};

/// How the HTTP server obtains its listening socket.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum HttpListenerMode {
    /// Bind a fresh TCP listener on the given host/port.
    BindTcp { host: String, port: u16 },
    /// Bind a fresh Unix domain listener on the given filesystem path.
    BindUnix { path: String },
    /// Adopt an already-open listening socket inherited from a supervisor
    /// process.
    InheritedFd { fd: i32 },
}

/// Which forward-only worker the UniServe Rust engine drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub enum EngineBackendKind {
    /// GPU-free CPU simulation engine — no Python, no GPU.
    Sim,
    /// The real GPU path: a Python forward-only worker over the shared-memory ring.
    #[default]
    Worker,
}

/// Configuration of the in-process UniServe engine. Rust owns scheduling and
/// engine execution; Python owns model forward execution.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EngineSettings {
    /// Which forward-only worker to drive.
    pub backend: EngineBackendKind,
    /// Maximum ops assembled into one forward batch.
    pub max_batch: usize,
    /// Per-step scheduling token budget (vLLM's `max_num_batched_tokens`).
    pub max_num_batched_tokens: usize,
    /// Maximum concurrently running requests (vLLM's `max_num_seqs`).
    pub max_num_seqs: usize,
    /// Per-request ceiling for one prefill chunk (SGLang's chunked prefill size).
    pub long_prefill_threshold: usize,
    /// Per-step budget of text prefill tokens allowed to join a decode batch
    /// as one mixed extend+decode forward. `0` disables mixing.
    pub mixed_prefill_tokens: usize,
    /// Waiting queue policy used by the scheduler.
    pub scheduler_policy: SchedulingPolicy,
    /// Maximum model context length reported to the frontend. `None` means
    /// "derive from the loaded model's real context length" (see
    /// [`EngineSettings::DEFAULT_MAX_MODEL_LEN`] for the final fallback when the
    /// model exposes no value).
    pub max_model_len: Option<u32>,
    /// Number of tensor-parallel worker rank processes (tp size of the single
    /// Full pool in the default topology).
    /// Staged-worker topology, e.g. `encoder:2,prefill:1:tp=4,decode:1:tp=4`.
    /// `None` selects one Full pool. A multi-stage layout
    /// composes pools behind a `StagedExecutor`.
    pub workers: WorkerTopology,
    /// Per-edge data-plane transfer backend (`--transfer`), e.g.
    /// `encoder->prefill=shm,prefill->decode=cuda_ipc`.
    pub transfer: TransportMap,
    /// Worker process arguments completed with resolved model assets before spawn.
    pub worker_process: WorkerProcessArgs,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            backend: EngineBackendKind::Worker,
            max_batch: DEFAULT_MAX_BATCH,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            long_prefill_threshold: DEFAULT_LONG_PREFILL_THRESHOLD,
            mixed_prefill_tokens: DEFAULT_MIXED_PREFILL_TOKENS,
            scheduler_policy: SchedulingPolicy::Fcfs,
            max_model_len: None,
            workers: WorkerTopology::single_full(1),
            transfer: TransportMap::default(),
            worker_process: WorkerProcessArgs {
                resp_slot_cap: EngineSettings::DEFAULT_RESP_SLOT_CAP,
                ..WorkerProcessArgs::default()
            },
        }
    }
}

/// Normalized runtime configuration for the minimal OpenAI-compatible server.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Config {
    /// In-process UniServe Rust engine settings (the southbound boundary).
    pub engine: EngineSettings,
    /// Backend model identifier used for engine loading.
    pub model: String,
    /// Closed description that owns model-specific serving behavior.
    pub model_description: ModelDescription,
    /// Single model name exposed to clients via the OpenAI API. When absent,
    /// the resolved model identifier is used.
    pub served_model_name: Option<String>,
    /// Shared filesystem namespace used for media outputs.
    pub media_spool: PathBuf,
    /// HTTP listener setup.
    pub listener_mode: HttpListenerMode,
    /// Server-default chat template override, as a file path or inline
    /// template.
    pub chat_template: Option<String>,
    /// Server-default keyword arguments merged into every chat-template render.
    pub default_chat_template_kwargs: Option<HashMap<String, Value>>,
    /// How to serialize `message.content` for chat-template rendering.
    pub chat_template_content_format: ChatTemplateContentFormatOption,
    /// Log a summary line for each completed request.
    pub enable_log_requests: bool,
    /// When `true`, set `X-Request-Id` on every HTTP response.
    pub enable_request_id_headers: bool,
    /// Whether to emit periodic stats logging (throughput, queue depth, cache usage).
    pub log_stats: bool,
    /// Bearer token accepted by the public serving API. Omitted from serialized
    /// config snapshots because it is a secret.
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
    /// Optional per-request wall-clock timeout.
    pub request_timeout: Option<Duration>,
    /// Optional front-door HTTP admission limit for in-flight inference
    /// requests.
    pub max_concurrent_requests: Option<u64>,
    /// Maximum time to wait for active HTTP requests to drain on shutdown.
    pub shutdown_timeout: Duration,
    /// Whether the model description's reasoning parser separates
    /// `reasoning_content` from `content`. When `false`, reasoning delimiter
    /// tokens stream verbatim as content text.
    pub reasoning_parsing: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            engine: EngineSettings::default(),
            model: String::new(),
            model_description: ModelDescription::Qwen3,
            served_model_name: None,
            media_spool: PathBuf::from("/tmp/uniserve-media"),
            listener_mode: HttpListenerMode::BindTcp {
                host: "127.0.0.1".to_string(),
                port: 8000,
            },
            chat_template: None,
            default_chat_template_kwargs: None,
            chat_template_content_format: ChatTemplateContentFormatOption::default(),
            enable_log_requests: false,
            enable_request_id_headers: false,
            log_stats: true,
            api_key: None,
            request_timeout: None,
            max_concurrent_requests: None,
            shutdown_timeout: Duration::from_secs(0),
            reasoning_parsing: true,
        }
    }
}

impl Config {
    /// Validate frontend configuration that can be checked before engine
    /// startup.
    pub fn validate(&self) -> Result<()> {
        self.validate_listener()?;
        self.engine.validate()?;
        anyhow::ensure!(
            self.media_spool.is_absolute(),
            "media spool path must be absolute"
        );
        Ok(())
    }

    /// Reject listener configurations that can never bind successfully. Port 0
    /// is intentionally allowed: it requests an OS-assigned ephemeral port.
    fn validate_listener(&self) -> Result<()> {
        match &self.listener_mode {
            HttpListenerMode::BindTcp { host, .. } => {
                anyhow::ensure!(!host.trim().is_empty(), "listener host must not be empty");
            }
            HttpListenerMode::BindUnix { path } => {
                anyhow::ensure!(
                    !path.trim().is_empty(),
                    "listener Unix socket path must not be empty"
                );
            }
            HttpListenerMode::InheritedFd { fd } => {
                anyhow::ensure!(*fd >= 0, "inherited listener fd must be non-negative");
            }
        }
        Ok(())
    }
}

impl EngineSettings {
    /// Final fallback context length used when neither the CLI nor the loaded
    /// model exposes a `max_model_len`.
    pub const DEFAULT_MAX_MODEL_LEN: u32 = 8192;

    /// Default response-ring slot capacity in bytes for the worker IPC
    /// transport (64 MiB).
    pub const DEFAULT_RESP_SLOT_CAP: usize = 64 << 20;

    /// H3 carries only compact descriptors and completion records over worker
    /// IPC; encoded media remains in the shared spool.
    pub const MEDIA_IPC_SLOT_CAP: usize = 64 << 10;

    /// Reject numeric engine settings that are structurally required to be
    /// positive (they index, divide, or bound scheduling). This catches a `0`
    /// override before it reaches the scheduler or KV sizing math.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.worker_process.block_size > 0,
            "block_size must be greater than 0"
        );
        anyhow::ensure!(
            self.worker_process.pipeline_depth > 0,
            "pipeline_depth must be greater than 0"
        );
        anyhow::ensure!(self.max_batch > 0, "max_batch must be greater than 0");
        anyhow::ensure!(
            self.max_num_batched_tokens > 0,
            "max_num_batched_tokens must be greater than 0"
        );
        anyhow::ensure!(self.max_num_seqs > 0, "max_num_seqs must be greater than 0");
        anyhow::ensure!(
            self.max_model_len.map(|len| len > 0).unwrap_or(true),
            "max_model_len must be greater than 0"
        );
        anyhow::ensure!(
            self.worker_process.resp_slot_cap > 0,
            "resp_slot_cap must be greater than 0"
        );
        anyhow::ensure!(
            !self.workers.pools.is_empty()
                && self
                    .workers
                    .pools
                    .iter()
                    .all(|pool| pool.count > 0 && pool.tp > 0),
            "workers must contain positive pool counts and tensor-parallel sizes"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        let config = Config::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.engine, EngineSettings::default());
    }

    #[test]
    fn zero_block_size_is_rejected() {
        let mut config = Config::default();
        config.engine.worker_process.block_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn unset_max_model_len_validates() {
        let mut config = Config::default();
        config.engine.max_model_len = None;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn zero_max_model_len_is_rejected() {
        let mut config = Config::default();
        config.engine.max_model_len = Some(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn zero_resp_slot_cap_is_rejected() {
        let mut config = Config::default();
        config.engine.worker_process.resp_slot_cap = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn ephemeral_tcp_port_is_allowed() {
        let config = Config {
            listener_mode: HttpListenerMode::BindTcp {
                host: "127.0.0.1".to_string(),
                port: 0,
            },
            ..Config::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn empty_listener_host_is_rejected() {
        let config = Config {
            listener_mode: HttpListenerMode::BindTcp {
                host: String::new(),
                port: 8000,
            },
            ..Config::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn empty_unix_path_is_rejected() {
        let config = Config {
            listener_mode: HttpListenerMode::BindUnix {
                path: String::new(),
            },
            ..Config::default()
        };
        assert!(config.validate().is_err());
    }
}
