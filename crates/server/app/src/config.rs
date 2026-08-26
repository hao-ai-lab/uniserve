use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;
use uniserve_engine_runtime::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS, SchedulingPolicy,
};
use uniserve_model_profile::ModelDescription;
use uniserve_serving::chat::ChatTemplateContentFormatOption;
use uniserve_worker_ipc::WorkerLaunchConfig;

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

/// How the server reaches its engine core(s).
///
/// `InProcess` is the deliberate single-node default: the engine
/// runs on a thread inside the server with no serialized hop. The socket
/// variants give UniServe vLLM's process topology — engines in separate
/// processes behind the handshake-negotiated wire protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub enum EngineConnection {
    /// The engine runs on a scheduler thread inside this process.
    #[default]
    InProcess,
    /// This server owns the startup handshake; engine processes (managed
    /// subprocesses or externally started `uniserve engine`s) dial in.
    Handshake {
        /// Endpoint engines dial (e.g. `tcp://127.0.0.1:5557`).
        handshake_address: String,
        /// Host engines use to connect back to the data-plane sockets.
        advertised_host: String,
        /// Total engines expected to join.
        engine_count: usize,
        /// Per-phase startup wait; must cover the engines' model load.
        ready_timeout: Duration,
    },
    /// An external supervisor fixed the transport addresses; bind them and
    /// wait for the engines' registration frames.
    Bootstrapped {
        input_address: String,
        output_address: String,
        engine_count: usize,
        ready_timeout: Duration,
    },
}

/// Configuration of the in-process UniServe engine. Rust owns scheduling and
/// engine execution; Python owns model forward execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EngineSettings {
    /// How the server reaches its engine core(s): in-process (default) or
    /// over the socket transport.
    pub connection: EngineConnection,
    /// Which forward-only worker to drive.
    pub backend: EngineBackendKind,
    /// Compute device for the worker.
    pub device: String,
    /// Attention backend preference forwarded to the Python worker.
    pub attention_backend: String,
    /// KV block size in tokens.
    pub block_size: u32,
    /// Op-batches kept in flight against the worker.
    pub pipeline_depth: usize,
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
    /// Optional KV token-capacity override for the worker.
    pub kv_token_capacity: Option<u64>,
    /// Response-ring slot capacity in bytes for the worker IPC transport.
    pub resp_slot_cap: usize,
    /// Python interpreter used to launch the worker.
    pub worker_python: String,
    /// Number of tensor-parallel worker rank processes (tp size of the single
    /// Full pool in the default topology).
    pub worker_ranks: usize,
    /// Staged-worker topology, e.g. `encoder:2,prefill:1:tp=4,decode:1:tp=4`.
    /// `None` selects one Full pool. A multi-stage spec
    /// composes pools behind a `StageRouter`.
    pub workers: Option<String>,
    /// Per-edge data-plane transfer backend (`--transfer`), e.g.
    /// `encoder->prefill=shm,prefill->decode=cuda_ipc`.
    pub transfer: Option<String>,
    /// Explicit Python worker launch/runtime configuration.
    pub worker_launch: WorkerLaunchConfig,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            connection: EngineConnection::InProcess,
            backend: EngineBackendKind::Worker,
            device: "cuda".to_string(),
            attention_backend: "auto".to_string(),
            block_size: 64,
            pipeline_depth: 2,
            max_batch: DEFAULT_MAX_BATCH,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            long_prefill_threshold: DEFAULT_LONG_PREFILL_THRESHOLD,
            mixed_prefill_tokens: DEFAULT_MIXED_PREFILL_TOKENS,
            scheduler_policy: SchedulingPolicy::Fcfs,
            max_model_len: None,
            kv_token_capacity: None,
            resp_slot_cap: EngineSettings::DEFAULT_RESP_SLOT_CAP,
            worker_python: "python3".to_string(),
            worker_ranks: 1,
            workers: None,
            transfer: None,
            worker_launch: WorkerLaunchConfig::default(),
        }
    }
}

/// Normalized runtime configuration for the minimal OpenAI-compatible server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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
    /// When `true`, suppress periodic stats logging (throughput, queue depth,
    /// cache usage).
    pub disable_log_stats: bool,
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
            disable_log_stats: false,
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
        if self.model_description == ModelDescription::MiniMaxH3 {
            anyhow::ensure!(
                !matches!(
                    self.engine.connection,
                    EngineConnection::Bootstrapped { .. }
                ),
                "MiniMax H3 requires an in-process or handshake-managed engine so the media spool can be verified"
            );
        }

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
        anyhow::ensure!(self.block_size > 0, "block_size must be greater than 0");
        anyhow::ensure!(
            self.pipeline_depth > 0,
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
            self.resp_slot_cap > 0,
            "resp_slot_cap must be greater than 0"
        );
        anyhow::ensure!(self.worker_ranks > 0, "worker_ranks must be greater than 0");
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
        config.engine.block_size = 0;
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
        config.engine.resp_slot_cap = 0;
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
