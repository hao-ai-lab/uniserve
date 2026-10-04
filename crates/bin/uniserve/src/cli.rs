//! Command-line parsing and runtime configuration for the `uniserve` binary.
//!
//! The parser exposes the serving command and lowers its model, scheduler,
//! worker, and HTTP options into the typed server configuration.
//!
//! The `///` docs on the fields and variants of the clap-derived types
//! (`Cli`, `Command`, `SchedulerPolicyArg`, `ServeArgs`, `SharedRuntimeArgs`,
//! and `WorkerProcessOptions`) are rendered as `--help` text, so developer
//! notes on those items belong in `//` comments.

use std::collections::HashMap;
use std::time::Duration;

use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use educe::Educe;
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror_ext::AsReport as _;
use uniserve_core::{KvCacheDtype, ModelDtype};
use uniserve_engine::{
    AttentionBackend, DEFAULT_COMPONENT, DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH,
    DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS,
    FlashInferBackend, LaneConfig, TransferConfig, WorkerConfig, WorkerProcessArgs,
};
use uniserve_server::profile::diffusion_gemma::DenoisingOverrides;
use uniserve_server::serving::systemone::{
    CandidateTokens, CanvasMode, ReadoutLayout, ReadoutOptions,
};
use uniserve_server::{
    ChatTemplateContentFormatOption, Config, EngineSettings, HttpListenerMode, ImageFetchPolicy,
    SchedulingPolicy, VideoMediaSettings,
};

const API_KEY_ENV: &str = "UNISERVE_API_KEY";

/// Call batches kept in flight against a worker when neither the command
/// line nor the deployment file sets its queue depth.
const DEFAULT_QUEUE_DEPTH: usize = 2;

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
    fn from(value: SchedulerPolicyArg) -> Self {
        match value {
            SchedulerPolicyArg::Fcfs => SchedulingPolicy::Fcfs,
            SchedulerPolicyArg::Priority => SchedulingPolicy::Priority,
        }
    }
}

/// Distributed expert access accepted by the command line.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum ExpertExchangeArg {
    /// FlashInfer's NVLink all-to-all around each rank's grouped experts.
    Alltoall,
    /// The fused CuTeDSL MegaMoE kernel (NVFP4 experts).
    Megamoe,
    /// Distributed weight data parallelism with asynchronous NVLink prefetch.
    Dwdp,
}

impl From<ExpertExchangeArg> for uniserve_engine::ExpertExchange {
    fn from(value: ExpertExchangeArg) -> Self {
        match value {
            ExpertExchangeArg::Alltoall => Self::AllToAll,
            ExpertExchangeArg::Megamoe => Self::MegaMoe,
            ExpertExchangeArg::Dwdp => Self::Dwdp,
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
    /// A stale socket file no server listens on is replaced, and the socket
    /// file is removed at shutdown.
    #[arg(long)]
    pub uds: Option<String>,

    /// Shared runtime arguments.
    #[command(flatten)]
    pub runtime: SharedRuntimeArgs,
}

impl ServeArgs {
    /// Builds the UniServe-native server config, binding the HTTP listener
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

    /// Local copy of the base checkpoint that a component export (such as
    /// FastH3 OmniRef) pins for its other components. The server and the
    /// workers verify its revision from its Hugging Face download records;
    /// without it they read the pinned revision from the Hugging Face cache.
    #[arg(long, value_name = "PATH")]
    pub base_model: Option<std::path::PathBuf>,

    /// Override the maximum model context length. When unset, the model's real
    /// context length (`max_position_embeddings`) is used.
    #[arg(long = "max-model-len")]
    pub max_model_len: Option<u32>,
    /// Longest video duration, in seconds, a media deployment provisions and
    /// admits: a capacity within the video API's [4, 15] second range,
    /// advertised as `max_seconds`.
    #[arg(long = "max-video-seconds", default_value_t = 15.0)]
    pub max_video_seconds: f64,
    /// Most denoiser rows the conditions of one video request may take: a
    /// capacity video workers provision their condition products, request
    /// slots and largest denoiser layout for, advertised as
    /// `max_condition_rows`. The default holds two keyframes on the largest
    /// canvas, every `fl2va` request; a `ref2va` deployment raises it to the
    /// rows its references take, about 38,000 for a five-second reference
    /// video with its soundtrack.
    #[arg(
        long = "max-condition-rows",
        default_value_t = uniserve_server::EngineSettings::DEFAULT_MAX_CONDITION_ROWS
    )]
    pub max_condition_rows: u32,
    /// Directory that `file://` condition media of video requests resolves
    /// under. Without it, `file://` media is refused.
    #[arg(long = "media-directory", value_name = "DIR")]
    pub media_directory: Option<std::path::PathBuf>,
    /// Whether the server fetches `http(s)://` condition media of video
    /// requests.
    #[arg(long = "remote-media", default_value_t = true, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub remote_media: bool,
    /// Bytes of condition media one video request may carry in total.
    #[arg(long = "max-request-bytes", default_value_t = VideoMediaSettings::DEFAULT_MAX_REQUEST_BYTES, value_parser = clap::value_parser!(u64).range(1..))]
    pub max_request_bytes: u64,
    /// `ffprobe` executable that probes video and audio condition media.
    #[arg(long, default_value = "ffprobe", value_name = "PATH")]
    pub ffprobe: std::path::PathBuf,
    /// Optional explicit KV token capacity override for the worker.
    #[arg(long = "max-total-tokens")]
    pub kv_token_capacity: Option<u64>,
    /// Recompute prompt prefixes instead of reusing or retaining cached KV pages.
    #[arg(long)]
    pub disable_prefix_cache: bool,
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
    /// Number of physical worker processes of each replica when configuration
    /// is omitted; they form one tensor-parallel group.
    #[arg(long = "worker-ranks", default_value_t = 1, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub worker_ranks: usize,
    /// Number of independent model replicas. Each replica runs its own
    /// scheduler, KV cache and worker ranks, and every request is served by
    /// the replica with the fewest requests in flight. Without `--workers`,
    /// the replicas take `--worker-ranks` ranks each, placed in blocks over
    /// `--worker-hosts`; a `--workers` file lists the replicas as equal
    /// consecutive blocks of groups.
    #[arg(long = "data-parallel-size", default_value_t = 1, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub data_parallel_size: usize,
    /// Shard the model's routed experts across the data-parallel replicas:
    /// each one-rank replica keeps its share of every expert layer and
    /// uses the selected expert exchange, while attention and every other
    /// layer stay data-parallel.
    #[arg(long = "expert-parallel", default_value_t = false)]
    pub expert_parallel: bool,
    /// Access distributed experts through NVLink all-to-all, fused MegaMoE,
    /// or DWDP's asynchronous weight prefetch with independent rank progress.
    #[arg(long = "expert-exchange", value_enum, default_value = "alltoall")]
    pub expert_exchange: ExpertExchangeArg,
    /// Path to a JSON deployment configuration: the Worker instances to serve,
    /// each one's node/device ranks, and the components placed on them.
    #[arg(long, value_name = "FILE", value_parser = read_workers)]
    pub workers: Option<Box<[WorkerConfig]>>,
    /// Directed product bindings: source[:rank]->destination[:rank]=mechanisms,
    /// where mechanisms names the edge's device mechanism, its host mechanism,
    /// or both joined by `+` with the device mechanism first (`cuda_vmm+shm`).
    #[arg(long)]
    pub transfer: Option<TransferConfig>,
    /// Tokens per KV page of the cache group with the widest token rows, a
    /// power of two; every other group's page holds as many tokens as fit
    /// the same unit plane. Unset, the worker chooses the largest size, at
    /// most 64, whose pages its attention kernels read in every cache group.
    #[arg(long = "page-size", value_parser = clap::builder::RangedU64ValueParser::<u32>::new().range(1..))]
    pub block_size: Option<u32>,
    /// This instance's host identity. Ranks are placed on it by name, and the
    /// engine owns exactly the ranks whose node matches it.
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
    /// How many call batches the scheduler keeps in flight against the worker.
    #[arg(long, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..), hide = true)]
    pub queue_depth: Option<usize>,
    /// Maximum number of calls assembled into one forward batch.
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

    /// If specified, API server will add an X-Request-Id header to responses,
    /// carrying the ID the request ran under: the client's X-Request-Id when
    /// accepted (1 to 128 visible ASCII characters without spaces), otherwise a
    /// generated one.
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    pub enable_request_id_headers: bool,

    /// Bearer token accepted by public serving API routes.
    #[arg(long = "api-key")]
    pub api_key: Option<String>,
    /// Timeout, in seconds, until a request's response head is sent; an
    /// expired request is answered with 504. Streamed bodies (chat completion
    /// SSE streams and video downloads) are not bounded by it.
    #[arg(long = "request-timeout", value_parser = clap::builder::RangedU64ValueParser::<u64>::new().range(1..))]
    pub request_timeout: Option<u64>,
    /// HTTP admission limit for in-flight chat completion and image generation
    /// requests, which are shed with 503 above it. Video requests are bounded
    /// by the video job slots instead.
    #[arg(long = "max-concurrent-requests", value_parser = clap::builder::RangedU64ValueParser::<u64>::new().range(1..))]
    pub max_concurrent_requests: Option<u64>,
    /// Enable or disable periodic logging of engine statistics.
    #[arg(long = "log-stats", action = ArgAction::Set, default_value_t = true)]
    pub log_stats: bool,

    /// Time limit, in seconds, for fetching one http(s) image URL, covering
    /// the connection, every redirect, and the complete response body.
    #[arg(
        long = "image-fetch-timeout",
        default_value_t = ImageFetchPolicy::DEFAULT_TIMEOUT.as_secs(),
        value_parser = clap::builder::RangedU64ValueParser::<u64>::new().range(1..)
    )]
    pub image_fetch_timeout: u64,
    /// Largest accepted input image in bytes, for fetched image URLs and
    /// decoded data URLs alike.
    #[arg(
        long = "image-fetch-max-bytes",
        default_value_t = ImageFetchPolicy::DEFAULT_MAX_BYTES,
        value_parser = clap::builder::RangedU64ValueParser::<u64>::new().range(1..)
    )]
    pub image_fetch_max_bytes: u64,
    /// Allow image URLs whose host is or resolves to a loopback, private,
    /// link-local, unique-local, or cloud metadata address. Scheme, redirect,
    /// time, size, and content-type limits still apply.
    #[arg(long = "allow-private-image-urls")]
    pub allow_private_image_urls: bool,

    /// The single model name used in the API. Defaults to the resolved model ID.
    #[arg(long)]
    pub served_model_name: Option<String>,

    /// How a DiffusionGemma server divides System One questions among
    /// readout prompts and canvases: `joint` packs questions in request order
    /// into shared canvases up to the canvas length; `independent` gives every
    /// question its own prompt and canvas.
    #[arg(long = "readout-layout", default_value = "joint")]
    pub readout_layout: ReadoutLayout,
    /// Length of every System One readout canvas on a DiffusionGemma server:
    /// `full` is the checkpoint's canvas length; `compact` the smallest
    /// multiple of 16 tokens that holds the answer scaffold; a positive
    /// multiple of 16 fixes the length, up to the checkpoint's canvas length.
    #[arg(long = "readout-canvas", default_value = "full")]
    pub readout_canvas: CanvasMode,
    /// Candidate token spellings used by System One readouts: `variants`
    /// sums supported spellings; `primary` uses only each candidate's
    /// space-prefixed token.
    #[arg(long = "readout-candidates", default_value = "variants")]
    pub readout_candidates: CandidateTokens,
    /// Block-diffusion sampling of every reply a DiffusionGemma server
    /// generates, as a JSON object that replaces any of the checkpoint's
    /// `generation_config.json` values `max_denoising_steps`,
    /// `entropy_bound`, `t_min`, `t_max`, `confidence_threshold`, and
    /// `stability_threshold`.
    #[arg(
        long = "diffusion-generation-config",
        value_parser = parse_json::<DenoisingOverrides>,
        value_name = "JSON"
    )]
    pub diffusion_generation_config: Option<DenoisingOverrides>,
}

impl SharedRuntimeArgs {
    /// Returns the positional `MODEL` argument unchanged; the server resolves
    /// model assets from it. When `--served-model-name` is absent,
    /// `async_main` defaults the served name to the same argument.
    pub(crate) fn resolved_model(&self) -> String {
        self.model.clone()
    }

    /// Returns the bearer token the server requires, if any.
    ///
    /// `--api-key` takes precedence over the `UNISERVE_API_KEY` environment
    /// variable. Each source is trimmed, and a blank value counts as unset, so
    /// a blank `--api-key` falls through to the environment variable.
    fn configured_api_key(&self) -> Option<String> {
        non_empty_secret(self.api_key.as_deref()).or_else(|| {
            std::env::var(API_KEY_ENV)
                .ok()
                .and_then(|value| non_empty_secret(Some(&value)))
        })
    }

    /// Returns the hosts the shorthand places ranks on, head's host first.
    ///
    /// `WorkerConfig::placed` assigns ranks to these hosts in contiguous blocks
    /// in list order, so leading with this instance's own identity keeps rank
    /// zero and the other lowest ranks on the head's host. Entries of
    /// `--worker-hosts` that are empty or repeat this instance's identity are
    /// dropped; other repeated names are kept as given.
    fn rank_hosts(&self) -> Vec<String> {
        let mut hosts = vec![self.host_identity.clone()];
        hosts.extend(
            self.worker_hosts
                .iter()
                .filter(|host| !host.is_empty() && **host != self.host_identity)
                .cloned(),
        );
        hosts
    }

    /// Builds the UniServe Rust-engine settings from these CLI arguments.
    ///
    /// The defaults are the same for every model. Each worker bounds what it
    /// can hold at load time, a media worker's request slots from its
    /// placement's queue depth, and the engine clamps resident requests and
    /// batch sizes to what the workers report. The IPC slot capacity a model's
    /// products require is applied when the model is resolved.
    pub(crate) fn engine_settings(&self) -> EngineSettings {
        // `build_state` in `uniserve_server` completes these launch arguments:
        // it sizes the IPC slots and the context length from the resolved
        // model and derives the worker's per-run batch bounds from the
        // settings built here.
        let mut worker_process = self.worker_process.to_args();
        worker_process.host = self.host_identity.clone();
        worker_process.python = self.worker_python.clone();
        worker_process.model = self.model.clone();
        worker_process.base_model = self.base_model.clone();
        let queue_depth = self.queue_depth.unwrap_or(DEFAULT_QUEUE_DEPTH);
        worker_process.queue_depth = queue_depth;
        worker_process.resp_slot_cap = self.resp_slot_cap;
        worker_process.kv_token_capacity = self.kv_token_capacity;
        worker_process.block_size = self.block_size;
        worker_process.attention_backend = self.attention_backend.clone();

        EngineSettings {
            max_batch: self.max_batch.unwrap_or(DEFAULT_MAX_BATCH),
            max_num_batched_tokens: self
                .max_num_batched_tokens
                .unwrap_or(DEFAULT_MAX_NUM_BATCHED_TOKENS),
            max_num_seqs: self.max_num_seqs.unwrap_or(DEFAULT_MAX_NUM_SEQS),
            long_prefill_threshold: self
                .long_prefill_threshold
                .unwrap_or(DEFAULT_LONG_PREFILL_THRESHOLD),
            mixed_prefill_tokens: self.mixed_prefill_tokens,
            prefix_cache: !self.disable_prefix_cache,
            scheduler_policy: self.scheduler_policy.into(),
            // `None` lets `build_state` derive the model's real context length;
            // an explicit `--max-model-len` overrides it.
            max_model_len: self.max_model_len,
            max_video_seconds: self.max_video_seconds,
            max_condition_rows: self.max_condition_rows,
            // Without a written configuration each replica serves one
            // component over its `--worker-ranks` ranks. A model whose
            // components are placed differently -- on disjoint ranks, or with
            // distinct partitions -- is served by writing that configuration,
            // which `--workers` parses into exactly the type this builds.
            workers: self.workers.clone().map(Vec::from).unwrap_or_else(|| {
                WorkerConfig::replicated(
                    &self.rank_hosts(),
                    &self.device,
                    self.data_parallel_size,
                    self.worker_ranks,
                    queue_depth,
                    WorkerConfig::single_component(DEFAULT_COMPONENT, self.worker_ranks),
                )
            }),
            transfer: self.transfer.clone().unwrap_or_default(),
            data_parallel_size: self.data_parallel_size,
            expert_parallel: self.expert_parallel.then_some(self.expert_exchange.into()),
            worker_process,
        }
    }

    /// Builds the OpenAI-server config for the in-process UniServe engine.
    ///
    /// Reads `UNISERVE_API_KEY` from the environment when `--api-key` is
    /// absent or blank (see `configured_api_key`).
    fn into_config(self, listener_mode: HttpListenerMode) -> Config {
        let engine = self.engine_settings();
        let model = self.resolved_model();
        let api_key = self.configured_api_key();
        let request_timeout = self.request_timeout.map(Duration::from_secs);
        Config {
            engine,
            model,
            base_model: self.base_model,
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
            video_media: VideoMediaSettings {
                media_directory: self.media_directory,
                remote_media: self.remote_media,
                max_request_bytes: self.max_request_bytes,
                ffprobe: self.ffprobe,
            },
            image_fetch: ImageFetchPolicy {
                timeout: Duration::from_secs(self.image_fetch_timeout),
                max_bytes: self.image_fetch_max_bytes,
                allow_private: self.allow_private_image_urls,
            },
            readout: ReadoutOptions {
                layout: self.readout_layout,
                canvas: self.readout_canvas,
                candidates: self.readout_candidates,
            },
            diffusion_generation: self.diffusion_generation_config.unwrap_or_default(),
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
    /// Each worker rank's share of its device's total storage, in (0, 1].
    /// A deployment file's `memory_fraction` overrides it for one worker.
    #[arg(
        long = "mem-fraction-static",
        default_value = "0.70",
        value_parser = parse_storage_fraction
    )]
    pub kv_storage_fraction: f64,
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
    /// Replay a startup-captured CUDA graph for every prefill call
    /// (default). `false` runs prefill eagerly, for debugging only.
    #[arg(long, action = ArgAction::Set, default_value_t = true, hide = true)]
    pub prefill_cuda_graph: bool,
    #[arg(long, hide = true)]
    pub prefill_graph_token_sizes: Option<String>,
    /// Capture image denoising calls at the configured flow shapes and
    /// replay those whose exact input signature was captured (default).
    /// `false` runs image denoising eagerly, for debugging only.
    #[arg(long, action = ArgAction::Set, default_value_t = true, hide = true)]
    pub flow_cuda_graph: bool,
    #[arg(long, hide = true)]
    pub flow_graph_batch_sizes: Option<String>,
    #[arg(long, hide = true)]
    pub flow_graph_shapes: Option<String>,
    /// Text capacities, in prompt tokens, of the video denoiser's layouts,
    /// as an increasing comma-separated list whose last entry holds
    /// `--max-model-len`, e.g. `1024,4096,16384`. Startup prepares every
    /// admitted duration at every capacity, and a request evaluates in the
    /// smallest capacity that holds its prompt; fewer capacities shorten
    /// startup and lower resident memory, finer ones pad less. Defaults to
    /// 1024 tokens, then steps of 2048 tokens.
    #[arg(long)]
    pub video_text_capacities: Option<String>,
    /// `ffmpeg` executable the media reader decodes reference videos with.
    /// The reference conditioning decodes with FFmpeg 8.1.2, whose LANCZOS
    /// scaler a reference video's pixels depend on.
    #[arg(long, default_value = "ffmpeg", value_name = "PATH")]
    pub ffmpeg: std::path::PathBuf,
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
            model_dtype: self.model_dtype,
            quantization_config: self.quantization_config.clone(),
            kv_cache_dtype: self.kv_cache_dtype,
            kv_storage_fraction: self.kv_storage_fraction,
            mesh: self.worker_mesh.clone(),
            distributed_backend: self.distributed_backend.clone(),
            lanes: self.lanes.clone(),
            graph_policy: self.graph_policy.clone(),
            decode_graph_batch_sizes: self.decode_graph_batch_sizes.clone(),
            prefill_cuda_graph: self.prefill_cuda_graph,
            prefill_graph_token_sizes: self.prefill_graph_token_sizes.clone(),
            flow_cuda_graph: self.flow_cuda_graph,
            flow_graph_batch_sizes: self.flow_graph_batch_sizes.clone(),
            flow_graph_shapes: self.flow_graph_shapes.clone(),
            video_text_capacities: self.video_text_capacities.clone(),
            ffmpeg: self.ffmpeg.clone(),
            flashinfer_workspace_size: self.flashinfer_workspace_size,
            flashinfer_use_tensor_core: self.flashinfer_use_tensor_core.clone(),
            flashinfer_decode_backend: self.flashinfer_decode_backend,
            flashinfer_prefill_backend: self.flashinfer_prefill_backend,
            flashinfer_decode_split_tile_size: self.flashinfer_decode_split_tile_size,
            flashinfer_prefill_split_tile_size: self.flashinfer_prefill_split_tile_size,
            flashinfer_disable_split_kv: self.flashinfer_disable_split_kv,
            ..WorkerProcessArgs::default()
        }
    }
}

/// Parses a JSON command-line value into `T` for clap.
///
/// The error text says "invalid JSON object" whatever `T` is; every caller
/// parses values that must be objects.
fn parse_json<T: DeserializeOwned>(value: &str) -> Result<T, String> {
    serde_json::from_str(value).map_err(|e| format!("invalid JSON object: {}", e.as_report()))
}

/// Parses a JSON value for clap and rejects anything but an object, such as
/// an array or a scalar.
fn parse_json_object(value: &str) -> Result<Value, String> {
    let parsed = parse_json::<Value>(value)?;
    if parsed.is_object() {
        Ok(parsed)
    } else {
        Err("expected a JSON object".to_string())
    }
}

/// Parses a device storage fraction for clap, refusing a value outside the
/// range `WorkerConfig::validate_storage_fraction` accepts.
fn parse_storage_fraction(value: &str) -> Result<f64, String> {
    let fraction = value
        .parse::<f64>()
        .map_err(|error| format!("invalid storage fraction {value:?}: {error}"))?;
    WorkerConfig::validate_storage_fraction(fraction).map_err(|error| error.to_string())?;
    Ok(fraction)
}

/// Returns the secret with surrounding whitespace trimmed, or `None` when it
/// is absent or blank.
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

/// Reads the deployment configuration a serve invocation was given.
///
/// The file holds a bare JSON array of `WorkerConfig` entries with no
/// enclosing object. The configuration is a file rather than an inline
/// argument because it states a whole deployment -- every rank's node and
/// device, and every component placed on them -- and is written once and
/// reused, not composed on a command line.
///
/// Runs as a clap value parser, so an unreadable file, malformed JSON, or a
/// configuration `WorkerConfig::validate_all` rejects fails argument parsing.
fn read_workers(path: &str) -> Result<Box<[WorkerConfig]>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("reading deployment configuration {path}: {error}"))?;
    let workers: Vec<WorkerConfig> = serde_json::from_str(&text)
        .map_err(|error| format!("parsing deployment configuration {path}: {error}"))?;
    WorkerConfig::validate_all(&workers).map_err(|error| error.to_string())?;
    Ok(workers.into_boxed_slice())
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

    /// The runtime options below parse together, and `--workers` reads and
    /// validates the deployment file at parse time.
    #[test]
    fn serve_accepts_runtime_configuration() {
        let configuration = written_configuration(
            r#"[{"id":"text","ranks":[{"node":"localhost","device":"cpu"},{"node":"localhost","device":"cpu"}],"components":{"model":{"ranks":[0,1],"parallel_config":{"tensor_parallel_size":2}}},"queue_depth":1,"memory_fraction":0.25}]"#,
        );
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
            configuration.to_str().expect("configuration path"),
        ])
        .expect("configured serve invocation");

        let Command::Serve(args) = parsed.command;
        let workers = args.runtime.workers.expect("the configuration was read");
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].id.0, "text");
        // The file's `memory_fraction` key deserializes into `storage_fraction`.
        assert_eq!(workers[0].storage_fraction, Some(0.25));
        assert_eq!(workers[0].components["model"].ranks, vec![0, 1]);
        assert_eq!(
            workers[0].components["model"]
                .parallel_config
                .tensor_parallel_size,
            2
        );
    }

    /// `--mem-fraction-static` is each rank's share of its device's total
    /// storage, so the parser accepts exactly the finite values in (0, 1]
    /// that the worker accepts, and refuses the rest before any rank starts.
    #[test]
    fn serve_accepts_only_a_storage_fraction_in_the_unit_interval() {
        let serve = |fraction: &str| {
            <Cli as clap::Parser>::try_parse_from([
                "uniserve",
                "serve",
                "model",
                "--mem-fraction-static",
                fraction,
            ])
        };
        for accepted in ["0.05", "1.0"] {
            assert!(serve(accepted).is_ok(), "{accepted} is a device share");
        }
        for refused in ["0", "-0.5", "1.5", "NaN", "inf"] {
            assert!(serve(refused).is_err(), "{refused} is not a device share");
        }
    }

    /// Writes a deployment configuration and returns the path naming it.
    ///
    /// The file name carries the process ID and the calling thread's ID, so
    /// concurrent callers in other test threads or processes write distinct
    /// files.
    fn written_configuration(body: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "uniserve-deployment-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, body).expect("write deployment configuration");
        path
    }

    #[test]
    fn serve_refuses_a_deployment_configuration_it_cannot_read() {
        let missing = std::env::temp_dir().join("uniserve-absent-deployment.json");
        let _ = std::fs::remove_file(&missing);
        let parsed = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "model",
            "--workers",
            missing.to_str().expect("configuration path"),
        ]);
        assert!(
            parsed.is_err(),
            "a missing configuration is not a deployment"
        );
    }

    #[test]
    fn serve_accepts_every_published_deployment_configuration() {
        // The repository's `configs/` tree holds the deployments the
        // documentation tells operators to pass to `--workers`.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../configs");
        let mut published = Vec::new();
        for model in std::fs::read_dir(&root).expect("configs directory") {
            for entry in std::fs::read_dir(model.expect("model directory").path())
                .expect("model configurations")
            {
                published.push(entry.expect("configuration entry").path());
            }
        }
        assert!(!published.is_empty(), "configs/ publishes deployments");

        for path in published {
            let parsed = <Cli as clap::Parser>::try_parse_from([
                "uniserve",
                "serve",
                "model",
                "--workers",
                path.to_str().expect("configuration path"),
            ]);
            assert!(
                parsed.is_ok(),
                "{} must be a valid deployment: {:?}",
                path.display(),
                parsed.err()
            );
        }
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
    fn serve_forwards_a_local_base_checkpoint() {
        let parsed = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "/models/FastH3-OmniRef",
            "--base-model",
            "/models/MiniMax-H3",
        ])
        .expect("component export with a local base");
        let Command::Serve(args) = parsed.command;
        let config = args.to_uniserve_config();
        let base = Some(std::path::PathBuf::from("/models/MiniMax-H3"));
        assert_eq!(config.base_model, base);
        assert_eq!(config.engine.worker_process.base_model, base);
    }

    #[test]
    fn serve_leaves_quantization_policy_to_model_by_default() {
        let parsed = <Cli as clap::Parser>::try_parse_from(["uniserve", "serve", "model"])
            .expect("MiniMax H3 default precision policy");
        let Command::Serve(args) = parsed.command;
        let worker = args.runtime.worker_process.to_args();
        assert_eq!(worker.quantization_config, serde_json::json!({}));
    }

    /// The image fetch flags lower into the server's image fetch policy, which
    /// defaults to a 20-second fetch, a 20 MB image, and public destinations
    /// only; a zero time or size limit is refused.
    #[test]
    fn serve_lowers_image_fetch_limits() {
        let serve = |flags: &[&str]| {
            <Cli as clap::Parser>::try_parse_from(
                ["uniserve", "serve", "model"]
                    .into_iter()
                    .chain(flags.iter().copied()),
            )
        };
        let policy = |flags: &[&str]| {
            let Command::Serve(args) = serve(flags).expect("serve invocation").command;
            args.to_uniserve_config().image_fetch
        };

        assert_eq!(
            policy(&[]),
            ImageFetchPolicy {
                timeout: Duration::from_secs(20),
                max_bytes: 20_000_000,
                allow_private: false,
            }
        );
        assert_eq!(
            policy(&[
                "--image-fetch-timeout",
                "5",
                "--image-fetch-max-bytes",
                "1024",
                "--allow-private-image-urls",
            ]),
            ImageFetchPolicy {
                timeout: Duration::from_secs(5),
                max_bytes: 1024,
                allow_private: true,
            }
        );
        assert!(serve(&["--image-fetch-timeout", "0"]).is_err());
        assert!(serve(&["--image-fetch-max-bytes", "0"]).is_err());
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
