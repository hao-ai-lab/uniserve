//! Server, engine, worker, model, and listener configuration values.
//!
//! `Config` is the normalized server input that the `uniserve` CLI and the
//! Dynamo worker binary build from their own arguments. `serve` validates it
//! before building state; a caller of `build_state` alone, such as the Dynamo
//! worker, calls `Config::validate` itself. `build_state` resolves model
//! assets, derives the worker and engine configurations from it, and starts
//! the engine.

use std::collections::HashMap;
use std::time::Duration;

use crate::serving::chat::ChatTemplateContentFormatOption;
use crate::serving::media::ImageFetchPolicy;
use crate::serving::systemone::ReadoutOptions;
use anyhow::Result;
use serde::Serialize;
use serde_json::Value;
use uniserve_engine::{
    DEFAULT_COMPONENT, DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH,
    DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS,
    SchedulingPolicy, TransferConfig, WorkerConfig, WorkerProcessArgs,
};

/// How the HTTP server obtains its listening socket.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum HttpListenerMode {
    /// Bind a fresh TCP listener on the given host/port.
    BindTcp {
        /// Host name or address passed to the TCP binder.
        host: String,
        /// TCP port, with zero requesting an OS-assigned ephemeral port.
        port: u16,
    },
    /// Bind a fresh Unix domain listener on the given filesystem path,
    /// replacing a stale socket file no server listens on. The socket file is
    /// removed when the listener is dropped at shutdown.
    BindUnix {
        /// Filesystem path for the Unix domain socket.
        path: String,
    },
    /// Adopt an already-open listening socket inherited from a supervisor
    /// process.
    InheritedFd {
        /// Nonnegative descriptor for the inherited listening socket.
        fd: i32,
    },
}

/// Configuration of the in-process UniServe engine. Rust owns scheduling and
/// engine execution; Python owns model forward execution.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EngineSettings {
    /// Maximum calls assembled into one forward batch.
    pub max_batch: usize,
    /// Maximum number of tokens scheduled in one engine step.
    pub max_num_batched_tokens: usize,
    /// Maximum number of concurrently running requests.
    pub max_num_seqs: usize,
    /// Per-request token ceiling for one prefill chunk.
    pub long_prefill_threshold: usize,
    /// Per-step budget of text prefill tokens a decode pass may co-schedule.
    /// The prefill tokens still travel as their own batch and numerical call.
    /// `0` disables co-scheduling.
    pub mixed_prefill_tokens: usize,
    /// Waiting queue policy used by the scheduler.
    pub scheduler_policy: SchedulingPolicy,
    /// Maximum model context length override. `None` derives the limit from
    /// the loaded model: its `max_position_embeddings` for a model with a root
    /// `config.json` (falling back to
    /// [`EngineSettings::DEFAULT_MAX_MODEL_LEN`]), or the default prompt limit
    /// of a diffusers pipeline.
    pub max_model_len: Option<u32>,
    /// Largest request duration, in seconds, resident media state is sized to
    /// serve. `InputProcessor::video_sampling` rejects longer video requests.
    pub max_video_seconds: f64,
    /// Static Worker configurations with ordered ranks and named computation components.
    pub workers: Vec<WorkerConfig>,
    /// Per-edge data-plane transfer backend (`--transfer`), e.g.
    /// `encoder->prefill=shm,prefill->decode=cuda_vmm`.
    pub transfer: TransferConfig,
    /// Worker process arguments completed with resolved model assets before spawn.
    pub worker_process: WorkerProcessArgs,
}

impl Default for EngineSettings {
    /// Returns the default value.
    fn default() -> Self {
        Self {
            max_batch: DEFAULT_MAX_BATCH,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            long_prefill_threshold: DEFAULT_LONG_PREFILL_THRESHOLD,
            mixed_prefill_tokens: DEFAULT_MIXED_PREFILL_TOKENS,
            scheduler_policy: SchedulingPolicy::Fcfs,
            max_model_len: None,
            max_video_seconds: 15.0,
            // One local CUDA rank running `DEFAULT_COMPONENT`. The `uniserve`
            // CLI and the Dynamo worker binary both replace this with a
            // placement built from their own arguments.
            workers: vec![WorkerConfig::placed(
                &["localhost".to_owned()],
                "cuda",
                1,
                2,
                WorkerConfig::single_component(DEFAULT_COMPONENT, 1),
            )],
            transfer: TransferConfig::default(),
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
    /// Backend model identifier used for engine loading: a Hugging Face
    /// repository ID or a local model directory. Empty by default; callers
    /// must set it.
    pub model: String,
    /// Single model name exposed to clients via the OpenAI API. When absent,
    /// the resolved model identifier is used.
    pub served_model_name: Option<String>,
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
    /// When `true`, set `X-Request-Id` on every HTTP response that passes the
    /// API-key check, carrying the ID the request ran under.
    pub enable_request_id_headers: bool,
    /// Whether to emit periodic stats logging (throughput, queue depth, cache usage).
    pub log_stats: bool,
    /// Bearer token accepted by the public serving API. Omitted from serialized
    /// config snapshots because it is a secret.
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
    /// Optional timeout until a request's response head is returned; streamed
    /// response bodies (chat SSE and video downloads) are not bounded by it.
    pub request_timeout: Option<Duration>,
    /// Optional HTTP admission limit for in-flight chat completion and image
    /// generation requests. Video requests are bounded by the video job slots
    /// instead.
    pub max_concurrent_requests: Option<u64>,
    /// Maximum time to wait for active HTTP requests to drain on shutdown.
    /// Zero aborts the server as soon as shutdown begins.
    pub shutdown_timeout: Duration,
    /// Whether the model description's reasoning parser separates
    /// `reasoning_content` from `content`. When `false`, reasoning delimiter
    /// tokens stream verbatim as content text.
    pub reasoning_parsing: bool,
    /// Time, size, and destination limits for request image references
    /// (`image_url` data and http(s) URLs).
    pub image_fetch: ImageFetchPolicy,
    /// How a DiffusionGemma server divides System One questions among
    /// readout prompts and canvases; other models ignore it.
    pub readout: ReadoutOptions,
}

impl Default for Config {
    /// Returns a configuration listening on TCP `127.0.0.1:8000` with an empty
    /// `model`, which callers must set before use.
    fn default() -> Self {
        Self {
            engine: EngineSettings::default(),
            model: String::new(),
            served_model_name: None,
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
            image_fetch: ImageFetchPolicy::default(),
            readout: ReadoutOptions::default(),
        }
    }
}

impl Config {
    /// Validates frontend configuration that can be checked before engine
    /// startup: the listener (`validate_listener`) and the engine settings
    /// (`EngineSettings::validate`). Returns the first violation found.
    pub fn validate(&self) -> Result<()> {
        self.validate_listener()?;
        self.engine.validate()?;
        Ok(())
    }

    /// Rejects listener configurations that cannot bind successfully. Port 0
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
    /// transport (64 MiB). `build_state` raises the configured capacity to the
    /// model's channel payload capacity when that is larger.
    pub const DEFAULT_RESP_SLOT_CAP: usize = 64 << 20;

    /// Rejects numeric engine settings that are structurally required to be
    /// positive (they index, divide, or bound scheduling). This catches a `0`
    /// override before it reaches the scheduler or KV sizing math. It also
    /// rejects a `max_video_seconds` outside the video API's [4, 15] second
    /// range (`validate_video_capacity`) and any
    /// worker placement `WorkerConfig::validate_all` refuses.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.worker_process.block_size > 0,
            "block_size must be greater than 0"
        );
        anyhow::ensure!(
            self.worker_process.queue_depth > 0,
            "queue_depth must be greater than 0"
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
        crate::serving::validate_video_capacity(self.max_video_seconds)
            .map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            self.worker_process.resp_slot_cap > 0,
            "resp_slot_cap must be greater than 0"
        );
        WorkerConfig::validate_all(&self.workers)?;
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
    }

    #[test]
    fn zero_block_size_is_rejected() {
        let mut config = Config::default();
        config.engine.worker_process.block_size = 0;
        assert!(config.validate().is_err());
    }

    /// A video capacity must lie within the API's [4, 15] second range.
    #[test]
    fn max_video_seconds_outside_the_api_range_is_rejected() {
        for (seconds, valid) in [
            (4.0, true),
            (15.0, true),
            (3.9, false),
            (15.5, false),
            (f64::NAN, false),
        ] {
            let mut config = Config::default();
            config.engine.max_video_seconds = seconds;
            assert_eq!(config.validate().is_ok(), valid, "{seconds} s");
        }
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
