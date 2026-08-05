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
use uniserve_engine_runtime::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS,
};
use uniserve_server::{
    ChatTemplateContentFormatOption, Config, EngineBackendKind, EngineSettings, HttpListenerMode,
    SchedulingPolicy,
};
use uniserve_worker_ipc::WorkerLaunchConfig;

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
    /// Run the UniServe OpenAI server: the Rust engine + scheduler run
    /// in-process by default, driving a forward-only worker.
    Serve(Box<ServeArgs>),
    /// Run one headless engine process: dial a frontend's handshake
    /// endpoint, host the Rust scheduler + forward-only worker behind the
    /// engine wire protocol.
    Engine(Box<EngineArgs>),
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

/// Arguments for the `engine` command (one headless engine process).
#[derive(Educe, Clone, Args)]
#[educe(Debug)]
#[command(override_usage = "uniserve engine <MODEL> --handshake-address <ADDR> [OPTIONS]")]
pub(crate) struct EngineArgs {
    /// Model identifier or local model directory loaded by the forward-only
    /// worker.
    #[arg(value_name = "MODEL")]
    pub model: String,

    /// Frontend handshake endpoint to dial (e.g. `tcp://127.0.0.1:5557` or
    /// `ipc:///tmp/uniserve-handshake`).
    #[arg(long)]
    pub handshake_address: String,
    /// Engine index within the deployment; becomes the 2-byte little-endian
    /// socket identity.
    #[arg(long, default_value_t = 0)]
    pub engine_index: u32,
    /// Maximum seconds to wait for the frontend's INIT after HELLO.
    #[arg(long, default_value_t = 300)]
    pub init_timeout: u64,

    /// Run the GPU-free CPU simulation engine instead of spawning the real
    /// forward-only worker. No Python and no GPU are required.
    #[arg(long)]
    pub sim: bool,
    /// Compute device for the forward-only worker.
    #[arg(long, default_value = "cuda")]
    pub device: String,
    /// Attention backend preference forwarded to the Python worker.
    #[arg(long, default_value = "auto")]
    pub attention_backend: String,
    /// KV block size in tokens (the page size).
    #[arg(long = "page-size", default_value_t = 64, value_parser = clap::builder::RangedU64ValueParser::<u32>::new().range(1..))]
    pub block_size: u32,
    /// How many op-batches the scheduler keeps in flight against the worker.
    #[arg(long, default_value_t = 2, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub pipeline_depth: usize,
    /// Maximum number of ops assembled into one forward batch.
    #[arg(long, default_value_t = DEFAULT_MAX_BATCH, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
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
    #[arg(long, default_value_t = DEFAULT_MIXED_PREFILL_TOKENS)]
    pub mixed_prefill_tokens: usize,
    /// Waiting queue policy used by the scheduler.
    #[arg(long = "schedule-policy", value_enum, default_value_t = SchedulerPolicyArg::Fcfs)]
    pub scheduler_policy: SchedulerPolicyArg,
    /// Maximum model context length reported to the frontend.
    #[arg(long = "max-model-len")]
    pub max_model_len: Option<u32>,
    /// Optional explicit KV token capacity override for the worker.
    #[arg(long = "max-total-tokens")]
    pub kv_token_capacity: Option<u64>,
    /// Python interpreter used to launch the worker.
    #[arg(long, default_value_t = default_worker_python())]
    pub worker_python: String,
    /// Number of worker rank processes behind this engine (1 = single ring;
    /// >1 spawns the MultiprocExecutor with one ring per rank).
    #[arg(long = "tp-size", default_value_t = 1, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub worker_ranks: usize,
    /// Staged-worker topology, e.g. `encoder:2,prefill:1:tp=4,decode:1:tp=4`.
    /// Unset = single Full pool.
    #[arg(long)]
    pub workers: Option<String>,
    /// Per-edge data-plane transfer backend, e.g.
    /// `encoder->prefill=cuda_ipc,prefill->decode=mooncake`.
    #[arg(long)]
    pub transfer: Option<String>,
    /// Response-ring slot capacity in bytes for the worker IPC transport.
    #[arg(long, default_value_t = EngineSettings::DEFAULT_RESP_SLOT_CAP, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub resp_slot_cap: usize,
    /// Explicit Python worker launch/runtime arguments.
    #[command(flatten)]
    pub worker_launch: WorkerLaunchArgs,
}

impl EngineArgs {
    fn resolved_model(&self) -> String {
        self.model.clone()
    }

    /// Build the engine-proc configuration. Control tokens default to the
    /// sim-compatible values and are overridden by the frontend's INIT
    /// generation control tokens during the handshake.
    pub(crate) fn to_proc_config(&self) -> uniserve_engine_process::EngineProcConfig {
        let mut core = uniserve_engine_runtime::EngineCoreConfig::sim(self.resolved_model());
        core.backend = if self.sim {
            uniserve_engine_runtime::EngineBackend::Sim
        } else {
            uniserve_engine_runtime::EngineBackend::Worker
        };
        core.device = self.device.clone();
        core.attention_backend = self.attention_backend.clone();
        core.block_size = self.block_size;
        core.pipeline_depth = self.pipeline_depth;
        core.max_batch = self.max_batch;
        core.max_num_batched_tokens = self.max_num_batched_tokens;
        core.max_num_seqs = self.max_num_seqs;
        core.long_prefill_threshold = self.long_prefill_threshold;
        core.mixed_prefill_tokens = self.mixed_prefill_tokens;
        core.scheduler_policy = self.scheduler_policy.into();
        // The engine subprocess does not load the frontend model backend, so it
        // cannot derive the model's real context length here; an explicit
        // override wins and otherwise the built-in default applies. The
        // engine re-reports its effective `max_model_len` to the frontend after
        // KV auto-fitting during the handshake.
        core.max_model_len = self
            .max_model_len
            .unwrap_or(EngineSettings::DEFAULT_MAX_MODEL_LEN);
        core.kv_token_capacity = self.kv_token_capacity;
        core.worker_python = self.worker_python.clone();
        core.worker_ranks = self.worker_ranks;
        core.workers = self.workers.clone();
        core.transfer = self.transfer.clone();
        core.worker_launch = self.worker_launch.to_config();
        core.resp_slot_cap = self.resp_slot_cap;
        uniserve_engine_process::EngineProcConfig {
            handshake_address: self.handshake_address.clone(),
            engine_index: self.engine_index,
            init_timeout: std::time::Duration::from_secs(self.init_timeout),
            core,
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

    /// Build the server config with an explicit engine connection (the socket
    /// modes: managed subprocesses or external engines).
    pub(crate) fn to_uniserve_config_with_connection(
        &self,
        connection: uniserve_server::EngineConnection,
    ) -> Config {
        let mut config = self.to_uniserve_config();
        config.engine.connection = connection;
        config
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
    pub attention_backend: String,
    /// Python interpreter used to launch the forward-only worker.
    #[arg(long, default_value_t = default_worker_python(), hide = true)]
    pub worker_python: String,
    /// Number of tensor-parallel worker rank processes behind each engine.
    #[arg(long = "tp-size", default_value_t = 1, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub worker_ranks: usize,
    /// Staged-worker topology, e.g. `encoder:2,prefill:1:tp=4,decode:1:tp=4`.
    /// Unset = a single Full pool (the non-disaggregated default); a multi-stage
    /// spec composes pools behind a StageRouter.
    #[arg(long, hide = true)]
    pub workers: Option<String>,
    /// Per-edge data-plane transfer backend, e.g.
    /// `encoder->prefill=cuda_ipc,prefill->decode=mooncake,decode->sampler=shm`.
    #[arg(long, hide = true)]
    pub transfer: Option<String>,
    /// KV block size in tokens (the page size).
    #[arg(long = "page-size", default_value_t = 64, value_parser = clap::builder::RangedU64ValueParser::<u32>::new().range(1..))]
    pub block_size: u32,
    /// Explicit Python worker launch/runtime arguments.
    #[command(flatten)]
    pub worker_launch: WorkerLaunchArgs,
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

    /// Run the engine out-of-process: expect this many engine cores behind the
    /// wire protocol (vLLM's process topology). `0` (the default) keeps the
    /// in-process engine — the deliberate single-node zero-hop path.
    #[arg(long, default_value_t = 0, hide = true)]
    pub engine_count: usize,
    /// Of `--engine-count`, how many engines this process spawns and
    /// supervises locally (managed mode). Defaults to all of them; `0` runs
    /// frontend-only — externally started `uniserve engine` processes dial in.
    #[arg(long, hide = true)]
    pub local_engine_count: Option<usize>,
    /// Engine handshake endpoint (`tcp://host:port`). Auto-allocated on
    /// 127.0.0.1 when unset (managed mode); set it explicitly for
    /// frontend-only mode so external engines know where to dial.
    #[arg(long, hide = true)]
    pub handshake_address: Option<String>,
    /// Host engines use to connect back to this frontend's data-plane sockets.
    #[arg(long, default_value = "127.0.0.1", hide = true)]
    pub advertised_host: String,
    /// Seconds to wait for engines to become ready (must cover model load).
    #[arg(long, default_value_t = 1800, hide = true)]
    pub engine_ready_timeout: u64,

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
    /// Disable periodic logging of engine statistics.
    #[arg(long)]
    pub disable_log_stats: bool,
    /// Enable or disable periodic logging of engine statistics.
    #[arg(long = "log-stats", action = ArgAction::Set)]
    pub log_stats: Option<bool>,

    /// The single model name used in the API. Defaults to the resolved model ID.
    #[arg(long)]
    pub served_model_name: Option<String>,
}

impl SharedRuntimeArgs {
    pub(crate) fn resolved_model(&self) -> String {
        self.model.clone()
    }

    fn disable_log_stats(&self) -> bool {
        self.log_stats
            .map(|enabled| !enabled)
            .unwrap_or(self.disable_log_stats)
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
        EngineSettings {
            connection: uniserve_server::EngineConnection::InProcess,
            backend: if self.sim {
                EngineBackendKind::Sim
            } else {
                EngineBackendKind::Worker
            },
            device: self.device.clone(),
            attention_backend: self.attention_backend.clone(),
            block_size: self.block_size,
            pipeline_depth: self.pipeline_depth,
            max_batch: self.max_batch,
            max_num_batched_tokens: self.max_num_batched_tokens,
            max_num_seqs: self.max_num_seqs,
            long_prefill_threshold: self.long_prefill_threshold,
            mixed_prefill_tokens: self.mixed_prefill_tokens,
            scheduler_policy: self.scheduler_policy.into(),
            // `None` lets `build_state` derive the model's real context length;
            // an explicit `--max-model-len` overrides it.
            max_model_len: self.max_model_len,
            kv_token_capacity: self.kv_token_capacity,
            resp_slot_cap: self.resp_slot_cap,
            worker_python: self.worker_python.clone(),
            worker_ranks: self.worker_ranks,
            workers: self.workers.clone(),
            transfer: self.transfer.clone(),
            worker_launch: self.worker_launch.to_config(),
        }
    }

    /// CLI arguments forwarded verbatim to each managed `uniserve engine`
    /// subprocess (the engine-tier settings of this serve invocation).
    pub(crate) fn engine_cli_args(&self) -> Vec<String> {
        let mut args = vec![
            "--device".to_string(),
            self.device.clone(),
            "--worker-python".to_string(),
            self.worker_python.clone(),
            "--tp-size".to_string(),
            self.worker_ranks.to_string(),
            "--attention-backend".to_string(),
            self.attention_backend.clone(),
            "--page-size".to_string(),
            self.block_size.to_string(),
            "--pipeline-depth".to_string(),
            self.pipeline_depth.to_string(),
            "--max-batch".to_string(),
            self.max_batch.to_string(),
            "--max-num-batched-tokens".to_string(),
            self.max_num_batched_tokens.to_string(),
            "--max-running-requests".to_string(),
            self.max_num_seqs.to_string(),
            "--chunked-prefill-size".to_string(),
            self.long_prefill_threshold.to_string(),
            "--schedule-policy".to_string(),
            format!("{:?}", self.scheduler_policy).to_ascii_lowercase(),
            "--resp-slot-cap".to_string(),
            self.resp_slot_cap.to_string(),
        ];
        if let Some(len) = self.max_model_len {
            args.push("--max-model-len".to_string());
            args.push(len.to_string());
        }
        if let Some(capacity) = self.kv_token_capacity {
            args.push("--max-total-tokens".to_string());
            args.push(capacity.to_string());
        }
        if let Some(workers) = &self.workers {
            args.push("--workers".to_string());
            args.push(workers.clone());
        }
        if let Some(transfer) = &self.transfer {
            args.push("--transfer".to_string());
            args.push(transfer.clone());
        }
        self.worker_launch.append_engine_cli_args(&mut args);
        if self.sim {
            args.push("--sim".to_string());
        }
        args
    }

    /// Build the OpenAI-server config for the in-process UniServe engine.
    fn into_config(self, listener_mode: HttpListenerMode) -> Config {
        let engine = self.engine_settings();
        let model = self.resolved_model();
        let disable_log_stats = self.disable_log_stats();
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
            disable_log_stats,
            api_key,
            request_timeout,
            max_concurrent_requests: self.max_concurrent_requests,
            shutdown_timeout: Duration::from_secs(self.shutdown_timeout),
        }
    }
}

/// Worker-runtime arguments forwarded to the Python worker process.
#[derive(Educe, Clone, Args)]
#[educe(Debug)]
pub(crate) struct WorkerLaunchArgs {
    #[arg(long, hide = true)]
    pub worker_stub: bool,
    #[arg(long = "dtype", default_value = "bfloat16")]
    pub model_dtype: String,
    #[arg(long)]
    pub kv_cache_dtype: Option<String>,
    #[arg(long = "mem-fraction-static", default_value = "0.70")]
    pub kv_memory_fraction: String,
    #[arg(long = "trust-remote-code")]
    pub transformers_trust_remote_code: bool,
    #[arg(long, default_value = "uniserve", hide = true)]
    pub transformers_attn_implementation: String,
    #[arg(long = "disable-model-arch", hide = true)]
    pub disable_model_arch: Vec<String>,
    /// Parallelism mesh forwarded to the Python worker, e.g.
    /// `tower=text:cuda:0;gen:cuda:1,tower-kv-capacity=65536`.
    #[arg(long, hide = true)]
    pub worker_mesh: Option<String>,
    #[arg(long, hide = true)]
    pub tp_backend: Option<String>,
    #[arg(long, default_value = "", hide = true)]
    pub mooncake_device: String,
    #[arg(long, default_value = "rdma", hide = true)]
    pub mooncake_protocol: String,
    #[arg(long = "enable-torch-compile")]
    pub torch_compile: bool,
    #[arg(long, default_value = "inductor", hide = true)]
    pub torch_compile_backend: String,
    #[arg(long, hide = true)]
    pub torch_compile_mode: Option<String>,
    #[arg(long, hide = true)]
    pub torch_compile_fullgraph: bool,
    #[arg(long, hide = true)]
    pub torch_compile_dynamic: Option<String>,
    #[arg(long, action = ArgAction::Set, default_value_t = true, hide = true)]
    pub cuda_graph: bool,
    #[arg(long, action = ArgAction::Set, default_value_t = true, hide = true)]
    pub cuda_graph_warmup: bool,
    #[arg(long, hide = true)]
    pub cuda_graph_warmup_batches: Option<String>,
    #[arg(long, action = ArgAction::Set, default_value_t = false, hide = true)]
    pub prefill_cuda_graph: bool,
    #[arg(long, action = ArgAction::Set, default_value_t = false, hide = true)]
    pub prefill_cuda_graph_warmup: bool,
    #[arg(long, hide = true)]
    pub prefill_cuda_graph_warmup_tokens: Option<String>,
    #[arg(long, default_value_t = 8192, hide = true)]
    pub mixed_text_max_tokens: u32,
    #[arg(long, action = ArgAction::Set, default_value_t = true, hide = true)]
    pub varlen_prefill: bool,
    #[arg(long, default_value_t = 512 * 1024 * 1024, hide = true)]
    pub flashinfer_workspace_size: u64,
    #[arg(long, hide = true)]
    pub flashinfer_use_tensor_core: Option<String>,
    #[arg(long, default_value = "fa2", hide = true)]
    pub flashinfer_decode_backend: String,
    #[arg(long, default_value = "auto", hide = true)]
    pub flashinfer_prefill_backend: String,
    #[arg(long, hide = true)]
    pub flashinfer_decode_split_tile_size: Option<u32>,
    #[arg(long, hide = true)]
    pub flashinfer_prefill_split_tile_size: Option<u32>,
    #[arg(long, hide = true)]
    pub flashinfer_disable_split_kv: bool,
    #[arg(long, action = ArgAction::Set, default_value_t = true, hide = true)]
    pub flashinfer_fast_decode_plan: bool,
    #[arg(long, hide = true)]
    pub snapshot_dir: Option<String>,
}

impl WorkerLaunchArgs {
    fn to_config(&self) -> WorkerLaunchConfig {
        WorkerLaunchConfig {
            stub: self.worker_stub,
            model_dtype: self.model_dtype.clone(),
            kv_cache_dtype: self.kv_cache_dtype.clone(),
            kv_memory_fraction: self.kv_memory_fraction.clone(),
            transformers_trust_remote_code: self.transformers_trust_remote_code,
            transformers_attn_implementation: self.transformers_attn_implementation.clone(),
            disable_model_arch: self.disable_model_arch.clone(),
            mesh: self.worker_mesh.clone(),
            tp_backend: self.tp_backend.clone(),
            mooncake_device: self.mooncake_device.clone(),
            mooncake_protocol: self.mooncake_protocol.clone(),
            torch_compile: self.torch_compile,
            torch_compile_backend: self.torch_compile_backend.clone(),
            torch_compile_mode: self.torch_compile_mode.clone(),
            torch_compile_fullgraph: self.torch_compile_fullgraph,
            torch_compile_dynamic: self.torch_compile_dynamic.clone(),
            cuda_graph: self.cuda_graph,
            cuda_graph_warmup: self.cuda_graph_warmup,
            cuda_graph_warmup_batches: self.cuda_graph_warmup_batches.clone(),
            prefill_cuda_graph: self.prefill_cuda_graph,
            prefill_cuda_graph_warmup: self.prefill_cuda_graph_warmup,
            prefill_cuda_graph_warmup_tokens: self.prefill_cuda_graph_warmup_tokens.clone(),
            mixed_text_max_tokens: self.mixed_text_max_tokens,
            varlen_prefill: self.varlen_prefill,
            flashinfer_workspace_size: self.flashinfer_workspace_size,
            flashinfer_use_tensor_core: self.flashinfer_use_tensor_core.clone(),
            flashinfer_decode_backend: self.flashinfer_decode_backend.clone(),
            flashinfer_prefill_backend: self.flashinfer_prefill_backend.clone(),
            flashinfer_decode_split_tile_size: self.flashinfer_decode_split_tile_size,
            flashinfer_prefill_split_tile_size: self.flashinfer_prefill_split_tile_size,
            flashinfer_disable_split_kv: self.flashinfer_disable_split_kv,
            flashinfer_fast_decode_plan: self.flashinfer_fast_decode_plan,
            snapshot_dir: self.snapshot_dir.clone(),
            restore_snapshots: false,
        }
    }

    fn append_engine_cli_args(&self, args: &mut Vec<String>) {
        let cfg = self.to_config();
        let default = WorkerLaunchConfig::default();
        if cfg.stub {
            args.push("--worker-stub".to_string());
        }
        push_if_changed(args, "--dtype", &cfg.model_dtype, &default.model_dtype);
        push_option(args, "--kv-cache-dtype", cfg.kv_cache_dtype.as_ref());
        push_if_changed(
            args,
            "--mem-fraction-static",
            &cfg.kv_memory_fraction,
            &default.kv_memory_fraction,
        );
        if cfg.transformers_trust_remote_code {
            args.push("--trust-remote-code".to_string());
        }
        push_if_changed(
            args,
            "--transformers-attn-implementation",
            &cfg.transformers_attn_implementation,
            &default.transformers_attn_implementation,
        );
        for arch in &cfg.disable_model_arch {
            args.push("--disable-model-arch".to_string());
            args.push(arch.clone());
        }
        push_option(args, "--worker-mesh", cfg.mesh.as_ref());
        push_option(args, "--tp-backend", cfg.tp_backend.as_ref());
        push_if_changed(
            args,
            "--mooncake-device",
            &cfg.mooncake_device,
            &default.mooncake_device,
        );
        push_if_changed(
            args,
            "--mooncake-protocol",
            &cfg.mooncake_protocol,
            &default.mooncake_protocol,
        );
        if cfg.torch_compile {
            args.push("--enable-torch-compile".to_string());
        }
        push_if_changed(
            args,
            "--torch-compile-backend",
            &cfg.torch_compile_backend,
            &default.torch_compile_backend,
        );
        push_option(
            args,
            "--torch-compile-mode",
            cfg.torch_compile_mode.as_ref(),
        );
        if cfg.torch_compile_fullgraph {
            args.push("--torch-compile-fullgraph".to_string());
        }
        push_option(
            args,
            "--torch-compile-dynamic",
            cfg.torch_compile_dynamic.as_ref(),
        );
        push_bool_value(args, "--cuda-graph", cfg.cuda_graph, default.cuda_graph);
        push_bool_value(
            args,
            "--cuda-graph-warmup",
            cfg.cuda_graph_warmup,
            default.cuda_graph_warmup,
        );
        push_option(
            args,
            "--cuda-graph-warmup-batches",
            cfg.cuda_graph_warmup_batches.as_ref(),
        );
        push_bool_value(
            args,
            "--prefill-cuda-graph",
            cfg.prefill_cuda_graph,
            default.prefill_cuda_graph,
        );
        push_bool_value(
            args,
            "--prefill-cuda-graph-warmup",
            cfg.prefill_cuda_graph_warmup,
            default.prefill_cuda_graph_warmup,
        );
        push_option(
            args,
            "--prefill-cuda-graph-warmup-tokens",
            cfg.prefill_cuda_graph_warmup_tokens.as_ref(),
        );
        push_u32_if_changed(
            args,
            "--mixed-text-max-tokens",
            cfg.mixed_text_max_tokens,
            default.mixed_text_max_tokens,
        );
        push_bool_value(
            args,
            "--varlen-prefill",
            cfg.varlen_prefill,
            default.varlen_prefill,
        );
        if cfg.flashinfer_workspace_size != default.flashinfer_workspace_size {
            args.push("--flashinfer-workspace-size".to_string());
            args.push(cfg.flashinfer_workspace_size.to_string());
        }
        push_option(
            args,
            "--flashinfer-use-tensor-core",
            cfg.flashinfer_use_tensor_core.as_ref(),
        );
        push_if_changed(
            args,
            "--flashinfer-decode-backend",
            &cfg.flashinfer_decode_backend,
            &default.flashinfer_decode_backend,
        );
        push_if_changed(
            args,
            "--flashinfer-prefill-backend",
            &cfg.flashinfer_prefill_backend,
            &default.flashinfer_prefill_backend,
        );
        push_u32_option(
            args,
            "--flashinfer-decode-split-tile-size",
            cfg.flashinfer_decode_split_tile_size,
        );
        push_u32_option(
            args,
            "--flashinfer-prefill-split-tile-size",
            cfg.flashinfer_prefill_split_tile_size,
        );
        if cfg.flashinfer_disable_split_kv {
            args.push("--flashinfer-disable-split-kv".to_string());
        }
        push_bool_value(
            args,
            "--flashinfer-fast-decode-plan",
            cfg.flashinfer_fast_decode_plan,
            default.flashinfer_fast_decode_plan,
        );
        push_option(args, "--snapshot-dir", cfg.snapshot_dir.as_ref());
    }
}

fn push_option(args: &mut Vec<String>, name: &str, value: Option<&String>) {
    if let Some(value) = value {
        args.push(name.to_string());
        args.push(value.clone());
    }
}

fn push_if_changed(args: &mut Vec<String>, name: &str, value: &str, default: &str) {
    if value != default {
        args.push(name.to_string());
        args.push(value.to_string());
    }
}

fn push_bool_value(args: &mut Vec<String>, name: &str, value: bool, default: bool) {
    if value != default {
        args.push(name.to_string());
        args.push(value.to_string());
    }
}

fn push_u32_if_changed(args: &mut Vec<String>, name: &str, value: u32, default: u32) {
    if value != default {
        args.push(name.to_string());
        args.push(value.to_string());
    }
}

fn push_u32_option(args: &mut Vec<String>, name: &str, value: Option<u32>) {
    if let Some(value) = value {
        args.push(name.to_string());
        args.push(value.to_string());
    }
}

fn parse_json<T: DeserializeOwned>(value: &str) -> Result<T, String> {
    serde_json::from_str(value).map_err(|e| format!("invalid JSON object: {}", e.as_report()))
}

fn non_empty_secret(value: Option<&str>) -> Option<String> {
    let trimmed = value?.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Default worker interpreter: a `python3`/`python` next to the running binary
/// (the env `bin/` for a `pip install`), then `$VIRTUAL_ENV`, then `python3`.
fn default_worker_python() -> String {
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        for name in ["python3", "python"] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return candidate.to_string_lossy().into_owned();
            }
        }
    }
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        let candidate = std::path::Path::new(&venv).join("bin").join("python");
        if candidate.is_file() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    "python3".to_string()
}

#[cfg(test)]
mod tests {
    // `Parser` (for `try_parse_from`) is re-exported via `super::*`.
    use super::*;

    #[test]
    fn serve_rejects_zero_block_size() {
        let res = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "model",
            "--page-size",
            "0",
        ]);
        assert!(res.is_err(), "block-size 0 must be rejected by clap range");
    }

    #[test]
    fn serve_accepts_valid_args_and_keeps_drain_default() {
        let cli = <Cli as clap::Parser>::try_parse_from(["uniserve", "serve", "model"])
            .expect("default serve invocation must parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };
        // Graceful drain is enabled by default (non-zero), not disabled.
        assert_eq!(args.runtime.shutdown_timeout, 30);
        assert_eq!(args.runtime.block_size, 64);
    }

    #[test]
    fn serve_configures_resp_slot_cap_and_defaults_max_model_len_to_derived() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "model",
            "--resp-slot-cap",
            "1048576",
        ])
        .expect("serve invocation with --resp-slot-cap must parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };
        let settings = args.runtime.engine_settings();
        // The in-process serve path configures resp_slot_cap...
        assert_eq!(settings.resp_slot_cap, 1 << 20);
        //...and leaves max_model_len unset so build_state derives the model's
        // real context length instead of forcing 8192.
        assert_eq!(settings.max_model_len, None);
    }

    #[test]
    fn serve_forwards_resp_slot_cap_to_managed_engine_args() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "model",
            "--resp-slot-cap",
            "1048576",
        ])
        .expect("serve invocation must parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };
        let engine_args = args.runtime.engine_cli_args();
        let idx = engine_args
            .iter()
            .position(|a| a == "--resp-slot-cap")
            .expect("managed engine args must forward --resp-slot-cap");
        assert_eq!(engine_args[idx + 1], "1048576");
    }

    #[test]
    fn serve_parses_and_forwards_worker_bool_values() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "model",
            "--cuda-graph",
            "false",
            "--prefill-cuda-graph",
            "true",
            "--prefill-cuda-graph-warmup",
            "true",
            "--varlen-prefill",
            "false",
            "--flashinfer-fast-decode-plan",
            "false",
        ])
        .expect("serve invocation with worker bool values must parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };
        let settings = args.runtime.engine_settings();
        assert!(!settings.worker_launch.cuda_graph);
        assert!(settings.worker_launch.prefill_cuda_graph);
        assert!(settings.worker_launch.prefill_cuda_graph_warmup);
        assert!(!settings.worker_launch.varlen_prefill);
        assert!(!settings.worker_launch.flashinfer_fast_decode_plan);

        let engine_args = args.runtime.engine_cli_args();
        assert!(
            engine_args
                .windows(2)
                .any(|pair| pair[0] == "--cuda-graph" && pair[1] == "false")
        );
        assert!(
            engine_args
                .windows(2)
                .any(|pair| pair[0] == "--prefill-cuda-graph" && pair[1] == "true")
        );
        assert!(
            engine_args
                .windows(2)
                .any(|pair| pair[0] == "--prefill-cuda-graph-warmup" && pair[1] == "true")
        );
        assert!(
            engine_args
                .windows(2)
                .any(|pair| pair[0] == "--varlen-prefill" && pair[1] == "false")
        );
        assert!(
            engine_args
                .windows(2)
                .any(|pair| pair[0] == "--flashinfer-fast-decode-plan" && pair[1] == "false")
        );
    }

    // ---- Cross-mode consistency: serve-settable flags forward to engine ----
    //
    // Every flag that `serve` forwards through `engine_cli_args()` is also a
    // real flag of the `engine` subcommand. These tests parameterize over the
    // forwarded flag set: for each `serve` flag/value, the forwarded args must
    // carry the same value AND must round-trip through the `engine` parser.

    /// Parse a `serve` invocation and return its `SharedRuntimeArgs`.
    fn parse_serve(extra: &[&str]) -> SharedRuntimeArgs {
        let mut argv = vec!["uniserve", "serve", "model"];
        argv.extend_from_slice(extra);
        let cli = <Cli as clap::Parser>::try_parse_from(argv)
            .expect("serve invocation under test must parse");
        match cli.command {
            Command::Serve(args) => args.runtime,
            Command::Engine(_) => panic!("expected serve command"),
        }
    }

    /// Find the value following `flag` in a forwarded args vector.
    fn forwarded_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    }

    /// `serve` flags that are always forwarded, with a non-default value and
    /// the exact string the forwarded engine arg should carry.
    fn always_forwarded_cases() -> Vec<(&'static str, &'static str, &'static str, &'static str)> {
        // (serve_flag, serve_value, engine_flag, expected_forwarded_value)
        vec![
            ("--device", "cpu", "--device", "cpu"),
            (
                "--worker-python",
                "/usr/bin/py",
                "--worker-python",
                "/usr/bin/py",
            ),
            ("--tp-size", "4", "--tp-size", "4"),
            (
                "--attention-backend",
                "flashinfer",
                "--attention-backend",
                "flashinfer",
            ),
            ("--page-size", "512", "--page-size", "512"),
            ("--pipeline-depth", "3", "--pipeline-depth", "3"),
            ("--max-batch", "7", "--max-batch", "7"),
            (
                "--max-num-batched-tokens",
                "4096",
                "--max-num-batched-tokens",
                "4096",
            ),
            (
                "--max-running-requests",
                "33",
                "--max-running-requests",
                "33",
            ),
            (
                "--chunked-prefill-size",
                "1234",
                "--chunked-prefill-size",
                "1234",
            ),
            (
                "--schedule-policy",
                "priority",
                "--schedule-policy",
                "priority",
            ),
            ("--resp-slot-cap", "1048576", "--resp-slot-cap", "1048576"),
        ]
    }

    #[test]
    fn serve_forwards_each_always_forwarded_flag_with_identical_value() {
        for (serve_flag, serve_value, engine_flag, expected) in always_forwarded_cases() {
            let runtime = parse_serve(&[serve_flag, serve_value]);
            let engine_args = runtime.engine_cli_args();
            assert_eq!(
                forwarded_value(&engine_args, engine_flag),
                Some(expected),
                "{serve_flag} {serve_value} must forward as {engine_flag} {expected}",
            );
        }
    }

    #[test]
    fn every_forwarded_flag_is_accepted_by_the_engine_subcommand() {
        // The full forwarded arg set (all always-forwarded flags set to
        // non-default values at once) must parse cleanly as `engine` flags,
        // proving the two modes agree on flag names and value formats.
        let extra: Vec<&str> = always_forwarded_cases()
            .iter()
            .flat_map(|(flag, value, _, _)| [*flag, *value])
            .collect();
        let runtime = parse_serve(&extra);
        let engine_args = runtime.engine_cli_args();

        let mut argv = vec![
            "uniserve".to_string(),
            "engine".to_string(),
            "model".to_string(),
            "--handshake-address".to_string(),
            "tcp://127.0.0.1:5557".to_string(),
        ];
        argv.extend(engine_args.iter().cloned());

        let parsed = <Cli as clap::Parser>::try_parse_from(&argv);
        assert!(
            parsed.is_ok(),
            "forwarded args {engine_args:?} must parse as engine flags: {:?}",
            parsed.err().map(|e| e.to_string()),
        );
    }

    #[test]
    fn forwarded_scheduler_policy_value_round_trips_into_engine_mode() {
        // `--schedule-policy` is forwarded as a lowercased Debug string; the
        // engine subcommand must accept that exact spelling.
        let runtime = parse_serve(&["--schedule-policy", "priority"]);
        let engine_args = runtime.engine_cli_args();
        assert_eq!(
            forwarded_value(&engine_args, "--schedule-policy"),
            Some("priority"),
        );

        let engine = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "engine",
            "model",
            "--handshake-address",
            "tcp://127.0.0.1:5557",
            "--schedule-policy",
            "priority",
        ])
        .expect("engine must accept forwarded scheduler-policy spelling");
        match engine.command {
            Command::Engine(args) => {
                assert!(matches!(
                    args.scheduler_policy,
                    SchedulerPolicyArg::Priority
                ));
            }
            Command::Serve(_) => panic!("expected engine command"),
        }
    }

    #[test]
    fn optional_max_model_len_is_forwarded_only_when_set() {
        // Unset on serve -> absent from forwarded args.
        let default_runtime = parse_serve(&[]);
        assert_eq!(
            forwarded_value(&default_runtime.engine_cli_args(), "--max-model-len"),
            None,
            "--max-model-len must not be forwarded when unset",
        );

        // Set on serve -> forwarded with the identical value.
        let set_runtime = parse_serve(&["--max-model-len", "8192"]);
        assert_eq!(
            forwarded_value(&set_runtime.engine_cli_args(), "--max-model-len"),
            Some("8192"),
        );
    }

    #[test]
    fn optional_kv_token_capacity_is_forwarded_only_when_set() {
        let default_runtime = parse_serve(&[]);
        assert_eq!(
            forwarded_value(&default_runtime.engine_cli_args(), "--max-total-tokens"),
            None,
        );

        let set_runtime = parse_serve(&["--max-total-tokens", "100000"]);
        assert_eq!(
            forwarded_value(&set_runtime.engine_cli_args(), "--max-total-tokens"),
            Some("100000"),
        );
    }

    #[test]
    fn optional_workers_and_transfer_topology_forward_verbatim() {
        let runtime = parse_serve(&[
            "--workers",
            "encoder:2,prefill:1:tp=4,decode:1:tp=4",
            "--transfer",
            "prefill->decode=mooncake",
        ]);
        let engine_args = runtime.engine_cli_args();
        assert_eq!(
            forwarded_value(&engine_args, "--workers"),
            Some("encoder:2,prefill:1:tp=4,decode:1:tp=4"),
        );
        assert_eq!(
            forwarded_value(&engine_args, "--transfer"),
            Some("prefill->decode=mooncake"),
        );
    }

    #[test]
    fn sim_flag_forwards_as_bare_switch_at_the_tail() {
        let runtime = parse_serve(&["--sim"]);
        let engine_args = runtime.engine_cli_args();
        assert!(
            engine_args.iter().any(|a| a == "--sim"),
            "--sim must be forwarded as a bare switch",
        );
        // `--sim` takes no value: it must not be followed by a stray value that
        // looks like an unrelated token (it is the final pushed arg).
        assert_eq!(engine_args.last().map(String::as_str), Some("--sim"));
    }

    #[test]
    fn default_serve_does_not_forward_the_sim_switch() {
        let runtime = parse_serve(&[]);
        assert!(
            !runtime.engine_cli_args().iter().any(|a| a == "--sim"),
            "default (non-sim) serve must not forward --sim",
        );
    }

    #[test]
    fn worker_launch_flag_kept_at_default_is_not_forwarded() {
        // `--dtype` defaults to "bfloat16"; an unchanged value is omitted.
        let runtime = parse_serve(&[]);
        assert_eq!(
            forwarded_value(&runtime.engine_cli_args(), "--dtype"),
            None,
            "default --dtype must be omitted from forwarded args",
        );
    }

    #[test]
    fn worker_launch_flag_changed_from_default_forwards_identical_value() {
        let runtime = parse_serve(&["--dtype", "float16"]);
        assert_eq!(
            forwarded_value(&runtime.engine_cli_args(), "--dtype"),
            Some("float16"),
        );
    }

    #[test]
    fn worker_mesh_forwards_to_managed_engine_args() {
        let runtime = parse_serve(&[
            "--worker-mesh",
            "tower=text:cuda:0;gen:cuda:1,tower-kv-capacity=65536",
        ]);
        assert_eq!(
            forwarded_value(&runtime.engine_cli_args(), "--worker-mesh"),
            Some("tower=text:cuda:0;gen:cuda:1,tower-kv-capacity=65536"),
        );
        assert_eq!(
            runtime.engine_settings().worker_launch.mesh.as_deref(),
            Some("tower=text:cuda:0;gen:cuda:1,tower-kv-capacity=65536"),
        );
    }

    #[test]
    fn serve_runtime_flags_parse_to_target_fields() {
        let runtime = parse_serve(&[
            "--tp-size",
            "2",
            "--page-size",
            "128",
            "--max-total-tokens",
            "4096",
            "--chunked-prefill-size",
            "2048",
            "--schedule-policy",
            "priority",
            "--dtype",
            "float16",
            "--mem-fraction-static",
            "0.5",
            "--trust-remote-code",
            "--enable-torch-compile",
        ]);

        assert_eq!(runtime.worker_ranks, 2);
        assert_eq!(runtime.block_size, 128);
        assert_eq!(runtime.kv_token_capacity, Some(4096));
        assert_eq!(runtime.long_prefill_threshold, 2048);
        assert!(matches!(
            runtime.scheduler_policy,
            SchedulerPolicyArg::Priority
        ));
        assert_eq!(runtime.worker_launch.model_dtype, "float16");
        assert_eq!(runtime.worker_launch.kv_memory_fraction, "0.5");
        assert!(runtime.worker_launch.transformers_trust_remote_code);
        assert!(runtime.worker_launch.torch_compile);
    }

    #[test]
    fn serve_uses_required_positional_model() {
        let runtime = parse_serve(&[]);
        assert_eq!(runtime.resolved_model(), "model");
    }

    #[test]
    fn serve_configures_public_server_controls() {
        let runtime = parse_serve(&[
            "--api-key",
            "public-key",
            "--request-timeout",
            "9",
            "--max-concurrent-requests",
            "11",
        ]);
        let config = runtime.into_config(HttpListenerMode::BindTcp {
            host: "127.0.0.1".to_string(),
            port: 8000,
        });

        assert_eq!(config.api_key.as_deref(), Some("public-key"));
        assert_eq!(config.request_timeout, Some(Duration::from_secs(9)));
        assert_eq!(config.max_concurrent_requests, Some(11));
    }

    #[test]
    fn forwarded_args_are_well_formed_flag_value_pairs() {
        // Every forwarded token that introduces a value must be a recognized
        // engine flag (no orphaned positionals leaking into the passthrough).
        let runtime = parse_serve(&[
            "--device",
            "cpu",
            "--page-size",
            "512",
            "--workers",
            "encoder:1",
        ]);
        let engine_args = runtime.engine_cli_args();
        // Every entry starting with "--" is a flag; the construction never emits
        // an empty token.
        assert!(
            engine_args.iter().all(|a| !a.is_empty()),
            "forwarded args must contain no empty tokens",
        );
        // The forwarded args, prefixed with the required engine positionals,
        // must parse without error (end-to-end well-formedness).
        let mut argv = vec![
            "uniserve".to_string(),
            "engine".to_string(),
            "model".to_string(),
            "--handshake-address".to_string(),
            "tcp://127.0.0.1:5557".to_string(),
        ];
        argv.extend(engine_args.iter().cloned());
        assert!(<Cli as clap::Parser>::try_parse_from(&argv).is_ok());
    }
}
