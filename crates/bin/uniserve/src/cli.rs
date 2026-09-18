//! Command-line parsing and runtime configuration for the `uniserve` binary.
//!
//! The parser exposes the serving command and lowers its model, scheduler,
//! worker, and HTTP options into the typed server configuration.

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
    FlashInferBackend, LaneConfig, TransferConfig, WorkerConfig, WorkerProcessArgs,
};
use uniserve_server::{
    ChatTemplateContentFormatOption, Config, EngineSettings, HttpListenerMode, SchedulingPolicy,
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
    /// Parses arguments from the current process environment.
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

/// Scheduler ordering policy accepted by the command line.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum SchedulerPolicyArg {
    /// Admit requests in arrival order.
    Fcfs,
    /// Admit higher-priority requests first while preserving stable ties.
    Priority,
}

impl From<SchedulerPolicyArg> for SchedulingPolicy {
    /// Converts the source value into this type.
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
    /// Builds the UniServe-native server config, binding the HTTP listener
    /// directly.
    pub(crate) fn to_uniserve_config(&self, is_media: bool) -> Config {
        let listener_mode = match &self.uds {
            Some(path) => HttpListenerMode::BindUnix { path: path.clone() },
            None => HttpListenerMode::BindTcp {
                host: self.host.clone(),
                port: self.port,
            },
        };
        self.runtime.clone().into_config(listener_mode, is_media)
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

    /// Override the maximum model context length. When unset, the model's real
    /// context length (`max_position_embeddings`) is used.
    #[arg(long = "max-model-len")]
    pub max_model_len: Option<u32>,
    /// Maximum request duration provisioned by a media configuration.
    #[arg(long = "max-video-seconds", default_value_t = 15.0)]
    pub max_video_seconds: f64,
    /// Optional explicit KV token capacity override for the worker.
    #[arg(long = "max-total-tokens")]
    pub kv_token_capacity: Option<u64>,
    /// Response-ring slot capacity in bytes for the worker IPC transport.
    #[arg(long, default_value_t = EngineSettings::DEFAULT_RESP_SLOT_CAP, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..), hide = true)]
    pub resp_slot_cap: usize,
    /// Compute device for the forward-only worker.
    #[arg(long, default_value = "cuda")]
    pub device: String,
    /// Attention backend preference forwarded to the Python worker.
    #[arg(long, default_value = "auto")]
    pub attention_backend: AttentionBackend,
    /// Python interpreter used to launch the forward-only worker.
    #[arg(long, default_value_os_t = default_worker_python(), hide = true)]
    pub worker_python: std::path::PathBuf,
    /// Number of physical worker processes when configuration is omitted.
    #[arg(long = "worker-ranks", default_value_t = 1, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub worker_ranks: usize,
    /// JSON array of Worker configurations, including each replica's node/device ranks.
    #[arg(long, value_parser = parse_workers)]
    pub workers: Option<Box<[WorkerConfig]>>,
    /// Directed product bindings: source[:rank]->destination[:rank]=backend.
    #[arg(long)]
    pub transfer: Option<TransferConfig>,
    /// KV block size in tokens (the page size).
    #[arg(long = "page-size", default_value_t = 64, value_parser = clap::builder::RangedU64ValueParser::<u32>::new().range(1..))]
    pub block_size: u32,
    /// This instance's host identity. Ranks are placed on it by name, and the
    /// engine owns exactly the ranks whose placement node matches.
    #[arg(long = "host-identity", default_value = "localhost")]
    pub host_identity: String,
    /// Hosts the shorthand spreads `--worker-ranks` across, in order, starting
    /// with this instance's own. Ranks are assigned in blocks so the lowest
    /// ranks stay on the head's host, and each host numbers its devices from
    /// zero. Omitted, every rank is placed on this host.
    #[arg(long = "worker-hosts", value_delimiter = ',')]
    pub worker_hosts: Vec<String>,
    /// Explicit Python worker launch/runtime arguments.
    #[command(flatten)]
    pub worker_process: WorkerProcessOptions,
    /// How many op-batches the scheduler keeps in flight against the worker.
    #[arg(long, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..), hide = true)]
    pub queue_depth: Option<usize>,
    /// Maximum number of ops assembled into one forward batch.
    #[arg(long, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..), hide = true)]
    pub max_batch: Option<usize>,
    /// Maximum transformer-token work admitted in one scheduling step.
    #[arg(long, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub max_num_batched_tokens: Option<usize>,
    /// Maximum number of concurrently resident requests.
    #[arg(long = "max-running-requests", value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub max_num_seqs: Option<usize>,
    /// Maximum number of prompt tokens processed per request in one prefill step.
    #[arg(long = "chunked-prefill-size", value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub long_prefill_threshold: Option<usize>,
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
    /// Returns the normalized model identifier used by server configuration.
    pub(crate) fn resolved_model(&self) -> String {
        self.model.clone()
    }

    /// Returns the configured API key, if present.
    fn configured_api_key(&self) -> Option<String> {
        non_empty_secret(self.api_key.as_deref()).or_else(|| {
            std::env::var(API_KEY_ENV)
                .ok()
                .and_then(|value| non_empty_secret(Some(&value)))
        })
    }

    /// Builds the UniServe Rust-engine settings from these CLI arguments.
    ///
    /// `is_media` comes from the checkpoint itself; video deployments size their
    /// queue, batch and IPC slots differently from token deployments.
    /// Returns the hosts the shorthand places ranks on, head's host first.
    ///
    /// Section 7's eight-device configuration spans two hosts, and its muxer
    /// sits on rank zero of the head's host, so the head's own identity leads
    /// the list whether or not `--worker-hosts` repeats it.
    fn placement_hosts(&self) -> Vec<String> {
        let mut hosts = vec![self.host_identity.clone()];
        hosts.extend(
            self.worker_hosts
                .iter()
                .filter(|host| !host.is_empty() && **host != self.host_identity)
                .cloned(),
        );
        hosts
    }

    pub(crate) fn engine_settings(&self, is_media: bool) -> EngineSettings {
        let mut worker_process = self.worker_process.to_args();
        worker_process.host = self.host_identity.clone();
        worker_process.python = self.worker_python.clone();
        worker_process.model = self.model.clone();
        let queue_depth = self.queue_depth.unwrap_or(if is_media { 6 } else { 2 });
        worker_process.queue_depth = queue_depth;
        worker_process.resp_slot_cap = if is_media {
            EngineSettings::MEDIA_IPC_SLOT_CAP
        } else {
            self.resp_slot_cap
        };
        worker_process.kv_token_capacity = self.kv_token_capacity;
        worker_process.block_size = self.block_size;
        worker_process.attention_backend = self.attention_backend.clone();
        EngineSettings {
            max_batch: self
                .max_batch
                .unwrap_or(if is_media { 2 } else { DEFAULT_MAX_BATCH }),
            max_num_batched_tokens: self.max_num_batched_tokens.unwrap_or(if is_media {
                2
            } else {
                DEFAULT_MAX_NUM_BATCHED_TOKENS
            }),
            max_num_seqs: self.max_num_seqs.unwrap_or(if is_media {
                2
            } else {
                DEFAULT_MAX_NUM_SEQS
            }),
            long_prefill_threshold: self.long_prefill_threshold.unwrap_or(if is_media {
                1
            } else {
                DEFAULT_LONG_PREFILL_THRESHOLD
            }),
            mixed_prefill_tokens: self.mixed_prefill_tokens,
            scheduler_policy: self.scheduler_policy.into(),
            // `None` lets `build_state` derive the model's real context length;
            // an explicit `--max-model-len` overrides it.
            max_model_len: self.max_model_len,
            max_video_seconds: self.max_video_seconds,
            workers: self.workers.clone().map(Vec::from).unwrap_or_else(|| {
                let hosts = self.placement_hosts();
                vec![if is_media {
                    WorkerConfig::h3(&hosts, &self.device, self.worker_ranks, queue_depth)
                } else {
                    WorkerConfig::model(&hosts, &self.device, self.worker_ranks, queue_depth)
                }]
            }),
            transfer: self.transfer.clone().unwrap_or_default(),
            worker_process,
        }
    }

    /// Builds the OpenAI-server config for the in-process UniServe engine.
    fn into_config(self, listener_mode: HttpListenerMode, is_media: bool) -> Config {
        let engine = self.engine_settings(is_media);
        let model = self.resolved_model();
        let api_key = self.configured_api_key();
        let request_timeout = self.request_timeout.map(Duration::from_secs);
        Config {
            engine,
            model,
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
    /// Checkpoint loader format used by every model worker.
    #[arg(long, default_value = "auto", value_parser = ["auto", "safetensors", "pt", "dummy", "layered"])]
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
    /// `tower=text:cuda:0;gen:cuda:1`.
    #[arg(long, hide = true)]
    pub worker_mesh: Option<String>,
    #[arg(long, hide = true)]
    pub distributed_backend: Option<String>,
    /// Repeatable JSON descriptor for a configuration-static execution lane.
    #[arg(long = "lane")]
    pub lanes: Vec<LaneConfig>,
    /// GPU computation capture policy. Full rejects unavailable capture; off disables all graphs.
    #[arg(long, default_value = "auto", value_parser = ["off", "auto", "full"])]
    pub graph_policy: String,
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
    /// Video request shapes whose denoising ladders warmup captures,
    /// as `SECONDSxTOKENS` items, e.g. `5x1000,15x10000`.
    #[arg(long)]
    pub video_graph_shapes: Option<String>,
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
}

impl WorkerProcessOptions {
    /// Converts CLI worker options into the engine's process-launch contract.
    fn to_args(&self) -> WorkerProcessArgs {
        WorkerProcessArgs {
            load_format: self.load_format.clone(),
            download_dir: self.download_dir.clone(),
            load_threads: self.load_threads,
            checksum_manifest: self.checksum_manifest.clone(),
            model_dtype: self.model_dtype.clone(),
            quantization_config: self.quantization_config.clone(),
            kv_cache_dtype: self.kv_cache_dtype.clone(),
            kv_memory_fraction: self.kv_memory_fraction.clone(),
            mesh: self.worker_mesh.clone(),
            distributed_backend: self.distributed_backend.clone(),
            lanes: self.lanes.clone(),
            graph_policy: self.graph_policy.clone(),
            decode_graph_batch_sizes: self.decode_graph_batch_sizes.clone(),
            prefill_cuda_graph: self.prefill_cuda_graph,
            prefill_graph_token_sizes: self.prefill_graph_token_sizes.clone(),
            flow_graph_batch_sizes: self.flow_graph_batch_sizes.clone(),
            flow_graph_shapes: self.flow_graph_shapes.clone(),
            video_graph_shapes: self.video_graph_shapes.clone(),
            flashinfer_workspace_size: self.flashinfer_workspace_size,
            flashinfer_use_tensor_core: self.flashinfer_use_tensor_core.clone(),
            flashinfer_decode_backend: self.flashinfer_decode_backend.clone(),
            flashinfer_prefill_backend: self.flashinfer_prefill_backend.clone(),
            flashinfer_decode_split_tile_size: self.flashinfer_decode_split_tile_size,
            flashinfer_prefill_split_tile_size: self.flashinfer_prefill_split_tile_size,
            flashinfer_disable_split_kv: self.flashinfer_disable_split_kv,
            ..WorkerProcessArgs::default()
        }
    }
}

/// Parses the JSON.
fn parse_json<T: DeserializeOwned>(value: &str) -> Result<T, String> {
    serde_json::from_str(value).map_err(|e| format!("invalid JSON object: {}", e.as_report()))
}

/// Parses the JSON object.
fn parse_json_object(value: &str) -> Result<Value, String> {
    let parsed = parse_json::<Value>(value)?;
    if parsed.is_object() {
        Ok(parsed)
    } else {
        Err("expected a JSON object".to_string())
    }
}

/// Returns a secret only when it contains a nonempty value.
fn non_empty_secret(value: Option<&str>) -> Option<String> {
    let trimmed = value?.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Returns the default worker interpreter next to the running binary
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
    fn serve_accepts_automatic_model_detection() {
        let result = <Cli as clap::Parser>::try_parse_from(["uniserve", "serve", "model"]);
        assert!(result.is_ok());
    }

    #[test]
    fn serve_rejects_zero_page_size() {
        let result = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "model",
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
            "--device",
            "cpu",
            "--worker-ranks",
            "2",
            "--page-size",
            "128",
            "--queue-depth",
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
            "--graph-policy",
            "off",
            "--prefill-cuda-graph",
            "true",
            "--worker-mesh",
            "tower=text:cpu",
            "--lane",
            r#"{"lane_id":"decode","sm_budget":64,"domains":["decode"]}"#,
            "--workers",
            r#"[{"id":"text","ranks":[{"node":"localhost","device":"cpu"},{"node":"localhost","device":"cpu"}],"entries":{"model":{"ranks":[0,1],"parallel_config":{"tensor_parallel_size":2}}},"queue_depth":1}]"#,
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
        let parsed = <Cli as clap::Parser>::try_parse_from(["uniserve", "serve", "model"])
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
            "--quantization-config",
            r#"["fp8"]"#,
        ])
        .expect_err("quantization config must be an object");
        assert!(error.to_string().contains("expected a JSON object"));
    }
}

/// Parses the canonical worker list without a second configuration wrapper.
fn parse_workers(value: &str) -> Result<Box<[WorkerConfig]>, String> {
    let workers: Vec<WorkerConfig> =
        serde_json::from_str(value).map_err(|error| error.to_string())?;
    WorkerConfig::validate_all(&workers).map_err(|error| error.to_string())?;
    Ok(workers.into_boxed_slice())
}
