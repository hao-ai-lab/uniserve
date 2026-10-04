//! Server, engine, worker, model, and listener configuration values.
//!
//! `Config` is the normalized server input that the `uniserve` CLI and the
//! Dynamo worker binary build from their own arguments. `serve` validates it
//! before building state; a caller of `build_state` alone, such as the Dynamo
//! worker, calls `Config::validate` itself. `build_state` resolves model
//! assets, derives the worker and engine configurations from it, and starts
//! the engine.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use crate::serving::chat::ChatTemplateContentFormatOption;
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
    /// Most denoiser rows a video request's conditions may take
    /// (`--max-condition-rows`). Video workers provision their condition
    /// products for it, and the video service rejects requests above it.
    pub max_condition_rows: u32,
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
            max_condition_rows: EngineSettings::DEFAULT_MAX_CONDITION_ROWS,
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
    /// Local copy of the base checkpoint that a component export (such as
    /// FastH3 OmniRef) pins for its other components, verified against the
    /// pinned revision by its Hugging Face download records. Without it the
    /// server and the workers read the pinned revision from the Hugging Face
    /// cache. Only a component export takes a base.
    pub base_model: Option<PathBuf>,
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
    /// Where video-request condition media may come from, and how large it
    /// may be.
    pub video_media: VideoMediaSettings,
}

/// Sources and limits of the condition media of video requests.
///
/// `data:` URIs are always accepted. Media of one request, decoded from every
/// source, is bounded per condition type and in total; the HTTP body limit of
/// the video routes follows the total.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VideoMediaSettings {
    /// The directory `file://` URIs resolve under; `None` refuses them.
    pub media_directory: Option<PathBuf>,
    /// Whether `http(s)://` media is fetched.
    pub remote_media: bool,
    /// Bytes of media one request may carry across all of its conditions.
    pub max_request_bytes: u64,
    /// The `ffprobe` executable that probes video and audio conditions.
    pub ffprobe: PathBuf,
}

impl VideoMediaSettings {
    /// The default request media total, 256 MiB.
    pub const DEFAULT_MAX_REQUEST_BYTES: u64 = 256 << 20;

    /// The HTTP body limit of the video submission routes, in bytes.
    ///
    /// A `data:` URI carries its media base64 encoded, four bytes per three,
    /// so a body holding `max_request_bytes` of media plus the request's own
    /// fields fits; it never falls below the server-wide 64 MiB limit.
    pub fn body_limit(&self) -> usize {
        let encoded = self.max_request_bytes.div_ceil(3) * 4 + (1 << 20);
        usize::try_from(encoded)
            .unwrap_or(usize::MAX)
            .max(crate::http::BODY_LIMIT)
    }
}

impl Default for VideoMediaSettings {
    /// No media directory, remote media enabled, a 256 MiB request total and
    /// `ffprobe` from `PATH`.
    fn default() -> Self {
        Self {
            media_directory: None,
            remote_media: true,
            max_request_bytes: Self::DEFAULT_MAX_REQUEST_BYTES,
            ffprobe: PathBuf::from("ffprobe"),
        }
    }
}

impl Default for Config {
    /// Returns a configuration listening on TCP `127.0.0.1:8000` with an empty
    /// `model`, which callers must set before use.
    fn default() -> Self {
        Self {
            engine: EngineSettings::default(),
            model: String::new(),
            base_model: None,
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
            video_media: VideoMediaSettings::default(),
        }
    }
}

impl Config {
    /// Validates frontend configuration that can be checked before engine
    /// startup: the listener (`validate_listener`), the engine settings
    /// (`EngineSettings::validate`) and a positive video media total. Returns
    /// the first violation found.
    pub fn validate(&self) -> Result<()> {
        self.validate_listener()?;
        self.engine.validate()?;
        anyhow::ensure!(
            self.video_media.max_request_bytes > 0,
            "max_request_bytes must be greater than 0"
        );
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

    /// Default condition capacity in denoiser rows: two keyframes on the
    /// largest named canvas, 1008 rows each, so every `fl2va` request fits.
    /// The capacity sizes each request slot's retained conditioning and the
    /// denoiser's largest layout, so a deployment that also serves `ref2va`
    /// states the capacity its references need (a five-second reference
    /// video with its soundtrack takes about 38,000 rows). A checkpoint's own
    /// sequence capacity bounds requests independently.
    pub const DEFAULT_MAX_CONDITION_ROWS: u32 = 2048;

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
