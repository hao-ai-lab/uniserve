//! CLI argument definitions for the `uniserve` (UniServe) binary.
//!
//! UniServe owns the engine and scheduler in Rust; Python only runs the model
//! forward pass. There is a single `serve` command.

use std::collections::HashMap;
use std::time::Duration;

use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use educe::Educe;
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror_ext::AsReport as _;
use uniserve_core::{KvCacheDtype, ModelDtype};
use uniserve_engine::{
    AttentionBackend, DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH,
    DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS,
    FlashInferBackend, LaneConfig, TransportMap, WorkerProcessArgs, WorkerTopology,
};
use uniserve_server::{
    ChatTemplateContentFormatOption, Config, EngineBackendKind, EngineSettings, HttpListenerMode,
    ModelDescription, SchedulingPolicy,
};

const API_KEY_ENV: &str = "UNISERVE_API_KEY";

/// Top-level parser for the `uniserve` binary.
#[derive(Debug, Parser)]
#[command(
    name = "uniserve",
    about = "UniServe OpenAI-compatible server: Rust engine + scheduler, Python forwards only."
)]
pub(crate) struct Cli {
    /// Default log level for UniServe components.
    #[arg(long, global = true)]
    pub log_level: Option<String>,
    /// Log level for HTTP server components.
    #[arg(long, global = true)]
    pub log_level_http: Option<String>,
    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    pub(crate) fn parse() -> Self {
        <Self as Parser>::parse()
    }
}

/// Supported top-level CLI commands.
#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Run the UniServe OpenAI server: the Rust engine and scheduler run
    /// in-process, driving a forward-only worker.
    Serve(Box<ServeArgs>),
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum SchedulerPolicyArg {
    Fcfs,
    Priority,
}

impl From<SchedulerPolicyArg> for SchedulingPolicy {
    fn from(value: SchedulerPolicyArg) -> Self {
        match value {
            SchedulerPolicyArg::Fcfs => SchedulingPolicy::Fcfs,
            SchedulerPolicyArg::Priority => SchedulingPolicy::Priority,
        }
    }
}

/// Arguments for the `serve` command.
#[derive(Educe, Clone, Args)]
#[educe(Debug)]
#[command(override_usage = "uniserve serve <MODEL> [OPTIONS]")]
pub(crate) struct ServeArgs {
    /// HTTP bind host for the OpenAI-compatible server.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// HTTP bind port for the OpenAI-compatible server.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,
    /// Unix domain socket path. If set, host and port arguments are ignored.
    #[arg(long)]
    pub uds: Option<String>,

    /// Shared runtime arguments.
    #[command(flatten)]
    pub runtime: SharedRuntimeArgs,
}

impl ServeArgs {
    /// Build the UniServe-native server config, binding the HTTP listener
    /// directly.
    pub(crate) fn to_uniserve_config(&self) -> Config {
        let listener_mode = match &self.uds {
            Some(path) => HttpListenerMode::BindUnix { path: path.clone() },
            None => HttpListenerMode::BindTcp {
                host: self.host.clone(),
                port: self.port,
            },
        };
        self.runtime.clone().into_config(listener_mode)
    }
}

/// Runtime arguments shared by every serve invocation.
#[derive(Educe, Clone, Args)]
#[educe(Debug)]
pub(crate) struct SharedRuntimeArgs {
    /// Model identifier or local model directory used for backend loading and
    /// public model ID.
    #[arg(value_name = "MODEL")]
    pub model: String,

    /// Closed model description that owns configured preprocessing and output behavior.
    #[arg(long)]
    pub model_description: ModelDescription,

    /// Override the maximum model context length. When unset, the model's real
    /// context length (`max_position_embeddings`) is used.
    #[arg(long = "max-model-len")]
    pub max_model_len: Option<u32>,
    /// Maximum request duration provisioned by a media deployment.
    #[arg(long = "max-video-seconds", default_value_t = 15.0)]
    pub max_video_seconds: f64,
    /// Optional explicit KV token capacity override for the worker.
    #[arg(long = "max-total-tokens")]
    pub kv_token_capacity: Option<u64>,
    /// Response-ring slot capacity in bytes for the worker IPC transport.
    #[arg(long, default_value_t = EngineSettings::DEFAULT_RESP_SLOT_CAP, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..), hide = true)]
    pub resp_slot_cap: usize,

    /// Run the GPU-free CPU simulation engine instead of spawning the real
    /// forward-only worker. No Python and no GPU are required.
    #[arg(long, hide = true)]
    pub sim: bool,
    /// Compute device for the forward-only worker.
    #[arg(long, default_value = "cuda")]
    pub device: String,
    /// Attention backend preference forwarded to the Python worker.
    #[arg(long, default_value = "auto")]
    pub attention_backend: AttentionBackend,
    /// Python interpreter used to launch the forward-only worker.
    #[arg(long, default_value_os_t = default_worker_python(), hide = true)]
    pub worker_python: std::path::PathBuf,
    /// Number of tensor-parallel worker rank processes behind each engine.
    #[arg(long = "tp-size", default_value_t = 1, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub worker_ranks: usize,
    /// Staged-worker topology, e.g. `encoder:2,prefill:1:tp=4,decode:1:tp=4`.
    /// Unset = a single Full pool; a multi-stage layout composes local pools
    /// behind a StagedExecutor.
    #[arg(long, hide = true)]
    pub workers: Option<WorkerTopology>,
    /// Per-edge data-plane transfer backend, e.g.
    /// `encoder->prefill=shm,prefill->decode=cuda_ipc`.
    #[arg(long, hide = true)]
    pub transfer: Option<TransportMap>,
    /// KV block size in tokens (the page size).
    #[arg(long = "page-size", default_value_t = 64, value_parser = clap::builder::RangedU64ValueParser::<u32>::new().range(1..))]
    pub block_size: u32,
    /// Explicit Python worker launch/runtime arguments.
    #[command(flatten)]
    pub worker_process: WorkerProcessOptions,
    /// How many op-batches the scheduler keeps in flight against the worker.
    #[arg(long, default_value_t = 2, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..), hide = true)]
    pub pipeline_depth: usize,
    /// Maximum number of ops assembled into one forward batch.
    #[arg(long, default_value_t = DEFAULT_MAX_BATCH, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..), hide = true)]
    pub max_batch: usize,
    /// Per-step scheduling token budget (vLLM's max_num_batched_tokens).
    #[arg(long, default_value_t = DEFAULT_MAX_NUM_BATCHED_TOKENS, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub max_num_batched_tokens: usize,
    /// Maximum concurrently running requests (vLLM's max_num_seqs).
    #[arg(long = "max-running-requests", default_value_t = DEFAULT_MAX_NUM_SEQS, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub max_num_seqs: usize,
    /// Per-request ceiling for one prefill chunk (SGLang's chunked prefill size).
    #[arg(long = "chunked-prefill-size", default_value_t = DEFAULT_LONG_PREFILL_THRESHOLD, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub long_prefill_threshold: usize,
    /// Per-step budget of text prefill tokens allowed to join a decode batch
    /// as one mixed extend+decode forward (0 disables mixing).
    #[arg(long, default_value_t = DEFAULT_MIXED_PREFILL_TOKENS, hide = true)]
    pub mixed_prefill_tokens: usize,
    /// Waiting queue policy used by the scheduler.
    #[arg(long = "schedule-policy", value_enum, default_value_t = SchedulerPolicyArg::Fcfs)]
    pub scheduler_policy: SchedulerPolicyArg,
    /// Maximum seconds to wait for active requests to drain during shutdown.
    /// `0` disables graceful drain (terminate immediately).
    #[arg(long, default_value_t = 30)]
    pub shutdown_timeout: u64,

    /// The file path to the chat template, or the template in single-line form
    /// for the specified model.
    #[arg(long)]
    pub chat_template: Option<String>,

    /// Default keyword arguments bound into the configured chat template.
    #[arg(long, value_parser = parse_json::<HashMap<String, Value>>, value_name = "JSON")]
    pub default_chat_template_kwargs: Option<HashMap<String, Value>>,

    /// The format to render message content within a chat template (`auto`,
    /// `string`, or `openai`).
    #[arg(long, default_value_t)]
    pub chat_template_content_format: ChatTemplateContentFormatOption,

    /// Reasoning parser applied to model output. `auto` uses the model
    /// description's parser to split `reasoning_content` from `content`;
    /// `none` streams reasoning delimiter tokens verbatim as content.
    #[arg(long, default_value = "auto", value_parser = ["auto", "none"])]
    pub reasoning_parser: String,

    /// Log a summary line for each completed request.
    #[arg(long = "log-requests")]
    pub enable_log_requests: bool,

    /// If specified, API server will add an X-Request-Id header to responses.
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    pub enable_request_id_headers: bool,

    /// Bearer token accepted by public serving API routes.
    #[arg(long = "api-key")]
    pub api_key: Option<String>,
    /// Per-request wall-clock timeout, in seconds.
    #[arg(long = "request-timeout", value_parser = clap::builder::RangedU64ValueParser::<u64>::new().range(1..))]
    pub request_timeout: Option<u64>,
    /// Front-door HTTP admission limit for in-flight inference requests.
    #[arg(long = "max-concurrent-requests", value_parser = clap::builder::RangedU64ValueParser::<u64>::new().range(1..))]
    pub max_concurrent_requests: Option<u64>,
    /// Enable or disable periodic logging of engine statistics.
    #[arg(long = "log-stats", action = ArgAction::Set, default_value_t = true)]
    pub log_stats: bool,

    /// The single model name used in the API. Defaults to the resolved model ID.
    #[arg(long)]
    pub served_model_name: Option<String>,
}

impl SharedRuntimeArgs {
    pub(crate) fn resolved_model(&self) -> String {
        self.model.clone()
    }

    fn configured_api_key(&self) -> Option<String> {
        non_empty_secret(self.api_key.as_deref()).or_else(|| {
            std::env::var(API_KEY_ENV)
                .ok()
                .and_then(|value| non_empty_secret(Some(&value)))
        })
    }

    /// Build the UniServe Rust-engine settings from these CLI arguments.
    pub(crate) fn engine_settings(&self) -> EngineSettings {
        let is_media = self.model_description == ModelDescription::MiniMaxH3;
        let mut worker_process = self.worker_process.to_args();
        worker_process.python = self.worker_python.clone();
        worker_process.model = self.model.clone();
        worker_process.device = self.device.clone();
        worker_process.world_size = self.worker_ranks;
        worker_process.pipeline_depth = self.pipeline_depth;
        worker_process.resp_slot_cap = if is_media {
            EngineSettings::MEDIA_IPC_SLOT_CAP
        } else {
            self.resp_slot_cap
        };
        worker_process.kv_token_capacity = self.kv_token_capacity;
        worker_process.block_size = self.block_size;
        worker_process.attention_backend = self.attention_backend.clone();
        EngineSettings {
            backend: if self.sim {
                EngineBackendKind::Sim
            } else {
                EngineBackendKind::Worker
            },
            max_batch: self.max_batch,
            max_num_batched_tokens: self.max_num_batched_tokens,
            max_num_seqs: self.max_num_seqs,
            long_prefill_threshold: self.long_prefill_threshold,
            mixed_prefill_tokens: self.mixed_prefill_tokens,
            scheduler_policy: self.scheduler_policy.into(),
            // `None` lets `build_state` derive the model's real context length;
            // an explicit `--max-model-len` overrides it.
            max_model_len: self.max_model_len,
            max_video_seconds: self.max_video_seconds,
            workers: self
                .workers
                .clone()
                .unwrap_or_else(|| WorkerTopology::single_full(self.worker_ranks)),
            transfer: self.transfer.clone().unwrap_or_default(),
            worker_process,
        }
    }

    /// Build the OpenAI-server config for the in-process UniServe engine.
    fn into_config(self, listener_mode: HttpListenerMode) -> Config {
        let engine = self.engine_settings();
        let model = self.resolved_model();
        let api_key = self.configured_api_key();
        let request_timeout = self.request_timeout.map(Duration::from_secs);
        Config {
            engine,
            model,
            model_description: self.model_description,
            served_model_name: self.served_model_name,
            listener_mode,
            chat_template: self.chat_template,
            default_chat_template_kwargs: self.default_chat_template_kwargs,
            chat_template_content_format: self.chat_template_content_format,
            enable_log_requests: self.enable_log_requests,
            enable_request_id_headers: self.enable_request_id_headers,
            log_stats: self.log_stats,
            api_key,
            request_timeout,
            max_concurrent_requests: self.max_concurrent_requests,
            shutdown_timeout: Duration::from_secs(self.shutdown_timeout),
            reasoning_parsing: self.reasoning_parser != "none",
        }
    }
}

/// Worker-runtime arguments forwarded to the Python worker process.
#[derive(Educe, Clone, Args)]
#[educe(Debug)]
pub(crate) struct WorkerProcessOptions {
    #[arg(long, hide = true)]
    pub worker_stub: bool,
    /// Checkpoint loader format used by every model worker.
    #[arg(long, default_value = "auto", value_parser = ["auto", "safetensors", "pt", "dummy", "sharded_state", "layered"])]
    pub load_format: String,
    /// Hugging Face cache root for repository model paths.
    #[arg(long)]
    pub download_dir: Option<std::path::PathBuf>,
    /// Concurrent checkpoint file readers.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub load_threads: Option<u32>,
    /// JSON map of checkpoint-relative paths to SHA-256 checksums.
    #[arg(long)]
    pub checksum_manifest: Option<std::path::PathBuf>,
    #[arg(long = "dtype", default_value = "bfloat16")]
    pub model_dtype: ModelDtype,
    /// JSON quantization policy, for example {"mode":"balanced"}.
    /// An empty object selects the model-owned default.
    #[arg(
        long,
        default_value = r#"{}"#,
        value_parser = parse_json_object,
        value_name = "JSON"
    )]
    pub quantization_config: serde_json::Value,
    #[arg(long)]
    pub kv_cache_dtype: Option<KvCacheDtype>,
    #[arg(long = "mem-fraction-static", default_value = "0.70")]
    pub kv_memory_fraction: f64,
    /// Parallelism mesh forwarded to the Python worker, e.g.
    /// `tower=text:cuda:0;gen:cuda:1,tower-kv-capacity=65536`.
    #[arg(long, hide = true)]
    pub worker_mesh: Option<String>,
    #[arg(long, hide = true)]
    pub tp_backend: Option<String>,
    /// Repeatable JSON descriptor for a deployment-static execution lane.
    #[arg(long = "lane")]
    pub lanes: Vec<LaneConfig>,
    #[arg(long, action = ArgAction::Set, default_value_t = true, hide = true)]
    pub cuda_graph: bool,
    #[arg(long, hide = true)]
    pub decode_graph_batch_sizes: Option<String>,
    #[arg(long, action = ArgAction::Set, default_value_t = false, hide = true)]
    pub prefill_cuda_graph: bool,
    #[arg(long, hide = true)]
    pub prefill_graph_token_sizes: Option<String>,
    #[arg(long, hide = true)]
    pub flow_graph_batch_sizes: Option<String>,
    #[arg(long, hide = true)]
    pub flow_graph_shapes: Option<String>,
    #[arg(long, default_value_t = 512 * 1024 * 1024, hide = true)]
    pub flashinfer_workspace_size: u64,
    #[arg(long, hide = true)]
    pub flashinfer_use_tensor_core: Option<String>,
    #[arg(long, default_value = "fa2", hide = true)]
    pub flashinfer_decode_backend: FlashInferBackend,
    #[arg(long, default_value = "auto", hide = true)]
    pub flashinfer_prefill_backend: FlashInferBackend,
    #[arg(long, hide = true)]
    pub flashinfer_decode_split_tile_size: Option<u32>,
    #[arg(long, hide = true)]
    pub flashinfer_prefill_split_tile_size: Option<u32>,
    #[arg(long, hide = true)]
    pub flashinfer_disable_split_kv: bool,
    #[arg(long, action = ArgAction::Set, default_value_t = true, hide = true)]
    pub flashinfer_fast_decode_plan: bool,
}

impl WorkerProcessOptions {
    fn to_args(&self) -> WorkerProcessArgs {
        WorkerProcessArgs {
            stub: self.worker_stub,
            load_format: self.load_format.clone(),
            download_dir: self.download_dir.clone(),
            load_threads: self.load_threads,
            checksum_manifest: self.checksum_manifest.clone(),
            model_dtype: self.model_dtype.clone(),
            quantization_config: self.quantization_config.clone(),
            kv_cache_dtype: self.kv_cache_dtype.clone(),
            kv_memory_fraction: self.kv_memory_fraction.clone(),
            mesh: self.worker_mesh.clone(),
            tp_backend: self.tp_backend.clone(),
            lanes: self.lanes.clone(),
            cuda_graph: self.cuda_graph,
            decode_graph_batch_sizes: self.decode_graph_batch_sizes.clone(),
            prefill_cuda_graph: self.prefill_cuda_graph,
            prefill_graph_token_sizes: self.prefill_graph_token_sizes.clone(),
            flow_graph_batch_sizes: self.flow_graph_batch_sizes.clone(),
            flow_graph_shapes: self.flow_graph_shapes.clone(),
            flashinfer_workspace_size: self.flashinfer_workspace_size,
            flashinfer_use_tensor_core: self.flashinfer_use_tensor_core.clone(),
            flashinfer_decode_backend: self.flashinfer_decode_backend.clone(),
            flashinfer_prefill_backend: self.flashinfer_prefill_backend.clone(),
            flashinfer_decode_split_tile_size: self.flashinfer_decode_split_tile_size,
            flashinfer_prefill_split_tile_size: self.flashinfer_prefill_split_tile_size,
            flashinfer_disable_split_kv: self.flashinfer_disable_split_kv,
            flashinfer_fast_decode_plan: self.flashinfer_fast_decode_plan,
            ..WorkerProcessArgs::default()
        }
    }
}

fn parse_json<T: DeserializeOwned>(value: &str) -> Result<T, String> {
    serde_json::from_str(value).map_err(|e| format!("invalid JSON object: {}", e.as_report()))
}

fn parse_json_object(value: &str) -> Result<Value, String> {
    let parsed = parse_json::<Value>(value)?;
    if parsed.is_object() {
        Ok(parsed)
    } else {
        Err("expected a JSON object".to_string())
    }
}

fn non_empty_secret(value: Option<&str>) -> Option<String> {
    let trimmed = value?.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Default worker interpreter: a `python3`/`python` next to the running binary
/// (the env `bin/` for a `pip install`), then `$VIRTUAL_ENV`, then `python3`.
fn default_worker_python() -> std::path::PathBuf {
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        for name in ["python3", "python"] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        let candidate = std::path::Path::new(&venv).join("bin").join("python");
        if candidate.is_file() {
            return candidate;
        }
    }
    "python3".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_requires_model_description() {
        let result = <Cli as clap::Parser>::try_parse_from(["uniserve", "serve", "model"]);
        assert!(result.is_err());
    }

    #[test]
    fn serve_rejects_zero_page_size() {
        let result = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "model",
            "--model-description",
            "qwen3",
            "--page-size",
            "0",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn serve_rejects_zero_checkpoint_readers() {
        let result = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "model",
            "--model-description",
            "qwen3",
            "--load-threads",
            "0",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn serve_accepts_runtime_configuration() {
        let parsed = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "model",
            "--model-description",
            "qwen3",
            "--device",
            "cpu",
            "--tp-size",
            "2",
            "--page-size",
            "128",
            "--pipeline-depth",
            "3",
            "--max-batch",
            "7",
            "--max-num-batched-tokens",
            "4096",
            "--max-running-requests",
            "33",
            "--max-total-tokens",
            "65536",
            "--chunked-prefill-size",
            "1234",
            "--schedule-policy",
            "priority",
            "--resp-slot-cap",
            "1048576",
            "--dtype",
            "float16",
            "--mem-fraction-static",
            "0.5",
            "--load-format",
            "safetensors",
            "--download-dir",
            "/models/cache",
            "--load-threads",
            "4",
            "--checksum-manifest",
            "/models/checksums.json",
            "--cuda-graph",
            "false",
            "--prefill-cuda-graph",
            "true",
            "--flashinfer-fast-decode-plan",
            "false",
            "--worker-mesh",
            "tower=text:cpu",
            "--lane",
            r#"{"lane_id":"decode","sm_budget":64,"domains":["decode"]}"#,
            "--workers",
            "prefill:1:tp=2,decode:1:tp=2",
            "--transfer",
            "prefill->decode=shm",
        ])
        .expect("configured serve invocation");
        assert!(matches!(parsed.command, Command::Serve(_)));
    }

    #[test]
    fn serve_accepts_component_quantization_config() {
        let parsed = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "model",
            "--model-description",
            "minimax-h3",
            "--quantization-config",
            r#"{"mode":"performance","components":{"transformer.attention":"fp8","transformer.mlp":"nvfp4","text_encoder":"bf16","video_vae":"bf16"}}"#,
        ])
        .expect("MiniMax H3 quantization config");
        let Command::Serve(args) = parsed.command;
        let worker = args.runtime.worker_process.to_args();
        assert_eq!(worker.quantization_config["mode"], "performance");
        assert_eq!(
            worker.quantization_config["components"]["transformer.mlp"],
            "nvfp4"
        );
    }

    #[test]
    fn serve_leaves_quantization_policy_to_model_by_default() {
        let parsed = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "model",
            "--model-description",
            "minimax-h3",
        ])
        .expect("MiniMax H3 default precision policy");
        let Command::Serve(args) = parsed.command;
        let worker = args.runtime.worker_process.to_args();
        assert_eq!(worker.quantization_config, serde_json::json!({}));
    }

    #[test]
    fn serve_rejects_non_object_quantization_config() {
        let error = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "model",
            "--model-description",
            "minimax-h3",
            "--quantization-config",
            r#"["fp8"]"#,
        ])
        .expect_err("quantization config must be an object");
        assert!(error.to_string().contains("expected a JSON object"));
    }
}
