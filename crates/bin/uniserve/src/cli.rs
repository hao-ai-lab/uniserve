//! CLI argument definitions for the `uniserve` (UniServe) binary.
//!
//! UniServe owns the engine and scheduler in Rust; Python (or `--sim`) only runs
//! the model forward pass. There is a single `serve` command — there is no
//! separate Python engine process to bootstrap or supervise.

use std::collections::HashMap;
use std::time::Duration;

use clap::{Args, Parser, Subcommand, ValueEnum};
use educe::Educe;
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror_ext::AsReport as _;
use uniserve_engine_runtime::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS,
};
use uniserve_server_app::{
    ChatTemplateContentFormatOption, Config, EngineBackendKind, EngineSettings, HttpListenerMode,
    ParserSelection, RendererSelection, SchedulingPolicy,
};

/// Top-level parser for the `uniserve` binary.
#[derive(Debug, Parser)]
#[command(
    name = "uniserve",
    about = "UniServe OpenAI-compatible server: Rust engine + scheduler, Python forwards only."
)]
pub(crate) struct Cli {
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
 /// in-process by default, driving a forward-only worker (or `--sim`).
    Serve(Box<ServeArgs>),
 /// Run one headless engine process: dial a frontend's handshake
 /// endpoint, host the Rust scheduler + forward-only worker behind the
 /// engine wire protocol.
    Engine(EngineArgs),
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
    #[arg(long, default_value_t = 256, value_parser = clap::builder::RangedU64ValueParser::<u32>::new().range(1..))]
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
    #[arg(long, default_value_t = DEFAULT_MAX_NUM_SEQS, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub max_num_seqs: usize,
 /// Per-request ceiling for one prefill chunk (SGLang's chunked prefill size).
    #[arg(long, default_value_t = DEFAULT_LONG_PREFILL_THRESHOLD, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub long_prefill_threshold: usize,
 /// Waiting queue policy used by the scheduler.
    #[arg(long, value_enum, default_value_t = SchedulerPolicyArg::Fcfs)]
    pub scheduler_policy: SchedulerPolicyArg,
 /// Maximum model context length reported to the frontend.
    #[arg(long)]
    pub max_model_len: Option<u32>,
 /// Optional explicit KV token capacity override for the worker.
    #[arg(long)]
    pub kv_token_capacity: Option<u64>,
    /// Python interpreter used to launch the worker.
    #[arg(long, default_value_t = default_worker_python())]
    pub worker_python: String,
 /// Number of worker rank processes behind this engine (1 = single ring;
 /// >1 spawns the MultiprocExecutor with one ring per rank).
    #[arg(long, default_value_t = 1, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
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
}

impl EngineArgs {
 /// Build the engine-proc configuration. Control tokens default to the
 /// sim-compatible values and are overridden by the frontend's INIT
 /// `native_controls` extension during the handshake.
    pub(crate) fn to_proc_config(&self) -> uniserve_engine_process::EngineProcConfig {
        let mut core = uniserve_engine_runtime::EngineCoreConfig::sim(self.model.clone());
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
 /// Build the UniServe-native server config (in-process Rust engine +
 /// forward-only worker or `--sim`), binding the HTTP listener directly.
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
        connection: uniserve_server_app::EngineConnection,
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
    pub model: String,

 /// Select the tool call parser depending on the model that you're using.
 /// Use `auto` to infer from the model or `none` to disable parsing.
    #[arg(long, default_value_t)]
    pub tool_call_parser: ParserSelection,
 /// Select the reasoning parser depending on the model that you're using.
 /// Use `auto` to infer from the model or `none` to disable parsing.
    #[arg(long, default_value_t)]
    pub uniserve_reasoning_parser: ParserSelection,
 /// Select the chat renderer implementation.
    #[arg(long = "tokenizer-mode", default_value_t)]
    pub renderer: RendererSelection,
 /// Disable multimodal inputs and treat the model as language-only.
    #[arg(long)]
    pub language_model_only: bool,
 /// Override the maximum model context length. When unset, the model's real
 /// context length (`max_position_embeddings`) is used.
    #[arg(long)]
    pub max_model_len: Option<u32>,
 /// Optional explicit KV token capacity override for the worker.
    #[arg(long)]
    pub kv_token_capacity: Option<u64>,
 /// Response-ring slot capacity in bytes for the worker IPC transport.
    #[arg(long, default_value_t = EngineSettings::DEFAULT_RESP_SLOT_CAP, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub resp_slot_cap: usize,

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
    /// Python interpreter used to launch the forward-only worker.
    #[arg(long, default_value_t = default_worker_python())]
    pub worker_python: String,
 /// Number of tensor-parallel worker rank processes behind each engine.
    #[arg(long, default_value_t = 1, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub worker_ranks: usize,
 /// Staged-worker topology, e.g. `encoder:2,prefill:1:tp=4,decode:1:tp=4`.
 /// Unset = a single Full pool (the non-disaggregated default); a multi-stage
 /// spec composes pools behind a StageRouter.
    #[arg(long)]
    pub workers: Option<String>,
 /// Per-edge data-plane transfer backend, e.g.
 /// `encoder->prefill=cuda_ipc,prefill->decode=mooncake,decode->sampler=shm`.
    #[arg(long)]
    pub transfer: Option<String>,
 /// KV block size in tokens (the page size).
    #[arg(long, default_value_t = 256, value_parser = clap::builder::RangedU64ValueParser::<u32>::new().range(1..))]
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
    #[arg(long, default_value_t = DEFAULT_MAX_NUM_SEQS, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub max_num_seqs: usize,
 /// Per-request ceiling for one prefill chunk (SGLang's chunked prefill size).
    #[arg(long, default_value_t = DEFAULT_LONG_PREFILL_THRESHOLD, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub long_prefill_threshold: usize,
 /// Waiting queue policy used by the scheduler.
    #[arg(long, value_enum, default_value_t = SchedulerPolicyArg::Fcfs)]
    pub scheduler_policy: SchedulerPolicyArg,
 /// TCP port for the gRPC Generate service. When not set, no gRPC server is
 /// started.
    #[arg(long)]
    pub grpc_port: Option<u16>,
 /// Maximum seconds to wait for active requests to drain during shutdown.
 /// `0` disables graceful drain (terminate immediately).
    #[arg(long, default_value_t = 30)]
    pub shutdown_timeout: u64,

 /// Run the engine out-of-process: expect this many engine cores behind the
 /// wire protocol (vLLM's process topology). `0` (the default) keeps the
 /// in-process engine — the deliberate single-node zero-hop path.
    #[arg(long, default_value_t = 0)]
    pub engine_count: usize,
 /// Of `--engine-count`, how many engines this process spawns and
 /// supervises locally (managed mode). Defaults to all of them; `0` runs
 /// frontend-only — externally started `uniserve engine` processes dial in.
    #[arg(long)]
    pub local_engine_count: Option<usize>,
 /// Engine handshake endpoint (`tcp://host:port`). Auto-allocated on
 /// 127.0.0.1 when unset (managed mode); set it explicitly for
 /// frontend-only mode so external engines know where to dial.
    #[arg(long)]
    pub handshake_address: Option<String>,
 /// Host engines use to connect back to this frontend's data-plane sockets.
    #[arg(long, default_value = "127.0.0.1")]
    pub advertised_host: String,
 /// Seconds to wait for engines to become ready (must cover model load).
    #[arg(long, default_value_t = 1800)]
    pub engine_ready_timeout: u64,

 /// The file path to the chat template, or the template in single-line form
 /// for the specified model.
    #[arg(long)]
    pub chat_template: Option<String>,

 /// Default keyword arguments to pass to the chat template renderer, merged
 /// with request-level `chat_template_kwargs` (request values take precedence).
    #[arg(long, value_parser = parse_json::<HashMap<String, Value>>, value_name = "JSON")]
    pub default_chat_template_kwargs: Option<HashMap<String, Value>>,

 /// The format to render message content within a chat template (`auto`,
 /// `string`, or `openai`).
    #[arg(long, default_value_t)]
    pub chat_template_content_format: ChatTemplateContentFormatOption,

 /// Log a summary line for each completed request.
    #[arg(long)]
    pub enable_log_requests: bool,

 /// If specified, API server will add an X-Request-Id header to responses.
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    pub enable_request_id_headers: bool,

 /// Disable periodic logging of engine statistics.
    #[arg(long)]
    pub disable_log_stats: bool,

 /// The model name(s) used in the API. The first is the primary ID returned
 /// in responses; all are accepted in requests. Defaults to `--model`.
    #[arg(long, num_args = 0..)]
    pub served_model_name: Vec<String>,
}

impl SharedRuntimeArgs {
 /// Build the UniServe Rust-engine settings from these CLI arguments.
    pub(crate) fn engine_settings(&self) -> EngineSettings {
        EngineSettings {
            connection: uniserve_server_app::EngineConnection::InProcess,
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
            "--worker-ranks".to_string(),
            self.worker_ranks.to_string(),
            "--attention-backend".to_string(),
            self.attention_backend.clone(),
            "--block-size".to_string(),
            self.block_size.to_string(),
            "--pipeline-depth".to_string(),
            self.pipeline_depth.to_string(),
            "--max-batch".to_string(),
            self.max_batch.to_string(),
            "--max-num-batched-tokens".to_string(),
            self.max_num_batched_tokens.to_string(),
            "--max-num-seqs".to_string(),
            self.max_num_seqs.to_string(),
            "--long-prefill-threshold".to_string(),
            self.long_prefill_threshold.to_string(),
            "--scheduler-policy".to_string(),
            format!("{:?}", self.scheduler_policy).to_ascii_lowercase(),
            "--resp-slot-cap".to_string(),
            self.resp_slot_cap.to_string(),
        ];
        if let Some(len) = self.max_model_len {
            args.push("--max-model-len".to_string());
            args.push(len.to_string());
        }
        if let Some(capacity) = self.kv_token_capacity {
            args.push("--kv-token-capacity".to_string());
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
        if self.sim {
            args.push("--sim".to_string());
        }
        args
    }

 /// Build the OpenAI-server config for the in-process UniServe engine.
    fn into_config(self, listener_mode: HttpListenerMode) -> Config {
        let engine = self.engine_settings();
        Config {
            engine,
            model: self.model,
            served_model_name: self.served_model_name,
            listener_mode,
            tool_call_parser: self.tool_call_parser,
            uniserve_reasoning_parser: self.uniserve_reasoning_parser,
            renderer: self.renderer,
            language_model_only: self.language_model_only,
            chat_template: self.chat_template,
            default_chat_template_kwargs: self.default_chat_template_kwargs,
            chat_template_content_format: self.chat_template_content_format,
            enable_log_requests: self.enable_log_requests,
            enable_request_id_headers: self.enable_request_id_headers,
            disable_log_stats: self.disable_log_stats,
            grpc_port: self.grpc_port,
            shutdown_timeout: Duration::from_secs(self.shutdown_timeout),
        }
    }
}

fn parse_json<T: DeserializeOwned>(value: &str) -> Result<T, String> {
    serde_json::from_str(value).map_err(|e| format!("invalid JSON object: {}", e.as_report()))
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
        let res = <Cli as clap::Parser>::try_parse_from(["uniserve", "serve", "model", "--block-size", "0"]);
        assert!(res.is_err(), "block-size 0 must be rejected by clap range");
    }

    #[test]
    fn serve_rejects_zero_max_num_seqs() {
        let res = <Cli as clap::Parser>::try_parse_from(["uniserve", "serve", "model", "--max-num-seqs", "0"]);
        assert!(res.is_err(), "max-num-seqs 0 must be rejected by clap range");
    }

    #[test]
    fn engine_rejects_zero_worker_ranks() {
        let res = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "engine",
            "model",
            "--handshake-address",
            "tcp://127.0.0.1:5557",
            "--worker-ranks",
            "0",
        ]);
        assert!(res.is_err(), "worker-ranks 0 must be rejected by clap range");
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
        assert_eq!(args.runtime.block_size, 256);
    }

    #[test]
    fn serve_rejects_zero_resp_slot_cap() {
        let res = <Cli as clap::Parser>::try_parse_from([
            "uniserve",
            "serve",
            "model",
            "--resp-slot-cap",
            "0",
        ]);
        assert!(res.is_err(), "resp-slot-cap 0 must be rejected by clap range");
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
}
