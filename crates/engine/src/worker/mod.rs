//! WorkerGroup process configuration, IPC execution, rank aggregation, and recovery.
//!
//! The engine drives GPU and host work through Python worker processes. A
//! `WorkerGroup` (in `instance`) is one worker's cooperating ranks: it projects
//! each submitted `Batch` onto the ranks whose components own its calls, joins
//! their agreeing results into `WorkerResult`s, and replaces the whole group
//! when a rank is lost. `WorkerExecutor` (in `executor`) submits explicitly
//! targeted work to the groups of a deployment. `process` owns one rank's
//! process and IPC channel, `registration` the addresses ranks rendezvous and
//! report their endpoints at, `launcher` the per-host launchers that start
//! ranks placed on other hosts, `checkpoint` the checkpoint identity every
//! rank is checked against, and `death_watch` the Linux watcher that fires a
//! rank channel's death wake when a rank process this engine spawned exits.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod checkpoint;
mod death_watch;
mod executor;
mod instance;
mod launcher;
mod process;
mod registration;

pub use executor::WorkerExecutor;
pub use instance::WorkerGroup;
pub use process::{FlashInferBackend, FlashInferBackendParseError, LaneConfig};
use process::{PendingRank, RankProcess};

/// Backpressure and terminal failures returned by a WorkerGroup run submission.
#[derive(Debug, thiserror::Error)]
pub enum BatchSubmitError {
    /// Returns the unaccepted physical submission when the instance has no capacity.
    ///
    /// The batch is handed back unchanged so the caller can resubmit it
    /// later; nothing was recorded or sent for it.
    #[error("worker run queue is full")]
    WouldBlock(Box<uniserve_worker_ipc::Batch>),
    /// Reports a terminal transport or execution failure.
    #[error(transparent)]
    Failed(anyhow::Error),
}

/// Everything required to create or recreate one worker rank process.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct WorkerProcessArgs {
    /// Logical instance identity assigned by the engine.
    pub worker_id: String,
    /// Python interpreter used to launch the worker entry point.
    pub python: std::path::PathBuf,
    /// Model identifier or local model path.
    pub model: String,
    /// Identity of the checkpoint at `model`, derived by the head when the
    /// model is a local directory of a non-stub launch and left unset
    /// otherwise. Every rank verifies the checkpoint it loads against it and
    /// reports what it loaded; a launch fills it once before any rank starts.
    pub checkpoint_identity: Option<String>,
    /// Ordered physical members of this WorkerGroup instance.
    pub ranks: Vec<crate::WorkerRank>,
    /// This process's own host identity. The engine owns exactly the ranks
    /// whose placement node matches it, so a placement may name its host
    /// explicitly instead of relying on a reserved local name.
    pub host: String,
    /// Component membership and parallel geometry.
    pub components: std::collections::BTreeMap<String, crate::ComponentConfig>,
    /// Every worker's component membership, keyed by worker identity, this
    /// group included. A rank is told which ranks read its products, and the
    /// consuming component may belong to another worker.
    pub peers: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, crate::ComponentConfig>,
    >,
    /// How long the head waits for every other host's launcher to present.
    ///
    /// The cluster starts a launcher on each host alongside the head, so this
    /// covers one that is starting rather than one waiting to be scheduled.
    /// Deployments differ in how quickly that happens, so it is a launch value
    /// rather than a constant.
    pub launcher_timeout: std::time::Duration,
    /// Launch the worker without model weights, using its deterministic test
    /// model. Only the engine's own tests and examples set this; the serving
    /// command line cannot request it.
    pub stub: bool,
    /// Maximum number of physical runs concurrently in flight per rank.
    pub queue_depth: usize,
    /// Request IPC slot capacity in bytes.
    pub req_slot_cap: usize,
    /// Response IPC slot capacity in bytes.
    pub resp_slot_cap: usize,
    /// Optional KV cache capacity expressed in tokens.
    pub kv_token_capacity: Option<u64>,
    /// Tokens per KV page of the cache group with the widest token rows;
    /// the worker derives every other group's page size from it. `None`
    /// lets the worker choose it from its cache layers and attention
    /// kernels.
    pub block_size: Option<u32>,
    /// Maximum calls accepted in one worker run.
    pub max_batch_calls: u32,
    /// Maximum tokens accepted in one worker run.
    pub max_batch_tokens: u32,
    /// Attention implementation selected for model execution.
    pub attention_backend: uniserve_worker_ipc::AttentionBackend,
    /// Optional public worker capability selectors; empty uses the model default.
    /// Startup resolves each selector to its concrete computation set.
    pub capability_groups: Vec<String>,
    /// Product transport exposed by the worker pool.
    pub transfer: crate::executor::TransferConfig,
    /// Checkpoint loader format.
    pub load_format: String,
    /// Optional cache directory for downloaded model artifacts.
    pub download_dir: Option<std::path::PathBuf>,
    /// Optional number of concurrent checkpoint reader threads.
    pub load_threads: Option<u32>,
    /// Optional manifest of checkpoint file checksums.
    pub checksum_manifest: Option<std::path::PathBuf>,
    /// Numeric data type used by model parameters and activations.
    pub model_dtype: uniserve_core::ModelDtype,
    /// WorkerGroup-specific quantization policy.
    pub quantization_config: serde_json::Value,
    /// Numeric data type used by the KV cache, when explicitly selected.
    pub kv_cache_dtype: Option<uniserve_core::KvCacheDtype>,
    /// Each rank's share of its device's total storage, which bounds the
    /// process's static allocations; `WorkerConfig::validate_storage_fraction`
    /// states its range.
    pub kv_storage_fraction: f64,
    /// Optional device mesh specification for staged model components.
    pub mesh: Option<String>,
    /// Optional process-world communication backend.
    pub distributed_backend: Option<String>,
    /// Configuration-static execution lane descriptors.
    pub lanes: Vec<LaneConfig>,
    /// Module graph coverage policy: off, auto, or full.
    pub graph_policy: String,
    /// Optional decode batch sizes selected for CUDA graph capture.
    pub decode_graph_batch_sizes: Option<String>,
    /// Whether prefill execution may use captured CUDA graphs.
    pub prefill_cuda_graph: bool,
    /// Optional prefill token counts selected for CUDA graph capture.
    pub prefill_graph_token_sizes: Option<String>,
    /// Optional diffusion batch sizes selected for CUDA graph capture.
    pub flow_graph_batch_sizes: Option<String>,
    /// Optional diffusion tensor shapes selected for CUDA graph capture.
    pub flow_graph_shapes: Option<String>,
    /// Optional text capacities, in prompt tokens, of a video denoiser's
    /// layouts, as a comma-separated increasing list.
    pub video_text_capacities: Option<String>,
    /// Argmax canvases each generating block-diffusion canvas keeps for its
    /// stopping rule: the stability threshold of the canvas sampling the
    /// deployment serves, which no admitted request exceeds. Zero keeps
    /// none, which serves confidence-only sampling and readouts.
    pub canvas_history_depth: u32,
    /// FlashInfer workspace capacity in bytes.
    pub flashinfer_workspace_size: u64,
    /// Optional FlashInfer tensor-core selection forwarded to the worker.
    pub flashinfer_use_tensor_core: Option<String>,
    /// FlashInfer backend used for decode attention.
    pub flashinfer_decode_backend: FlashInferBackend,
    /// FlashInfer backend used for prefill attention.
    pub flashinfer_prefill_backend: FlashInferBackend,
    /// Optional split-KV tile size used for FlashInfer decode attention.
    pub flashinfer_decode_split_tile_size: Option<u32>,
    /// Optional split-KV tile size used for FlashInfer prefill attention.
    pub flashinfer_prefill_split_tile_size: Option<u32>,
    /// Whether FlashInfer split-KV execution is disabled.
    pub flashinfer_disable_split_kv: bool,
    /// Maximum model context length in tokens.
    pub max_model_len: u32,
    /// Maximum accepted video duration in seconds.
    pub max_video_seconds: f64,
    /// Shortest video duration in seconds the serving API admits, which
    /// bounds the frame counts a video worker provisions from below; `None`
    /// provisions every frame count the model generates.
    pub min_video_seconds: Option<f64>,
}

/// Parks until one descriptor becomes readable or `timeout` expires.
///
/// Returning `Ok` does not say which happened or which descriptor fired;
/// callers re-poll their channels afterwards. An empty `fds` is a bounded
/// sleep: nothing in the engine unparks the waiting thread. Wake events are
/// the ones each `pollfd` requests, plus the error and hangup conditions poll
/// always reports. Fails on a poll error other than an interrupt, and on
/// every call with descriptors on targets other than Linux.
pub(crate) fn park_descriptors(
    fds: &[libc::pollfd],
    timeout: std::time::Duration,
) -> anyhow::Result<()> {
    if fds.is_empty() {
        std::thread::park_timeout(timeout);
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        let mut pollfds = fds.to_vec();
        // The millisecond timeout rounds a sub-millisecond remainder up, so a
        // nonzero wait never becomes a zero-timeout poll that spins, and is
        // capped at the largest value poll accepts.
        let timeout_ms = timeout
            .as_millis()
            .saturating_add(u128::from(
                !timeout.subsec_nanos().is_multiple_of(1_000_000),
            ))
            .min(i32::MAX as u128) as i32;
        loop {
            // SAFETY: `pollfds` is a live contiguous allocation for the stated
            // element count and poll does not retain the pointer.
            let result = unsafe {
                libc::poll(
                    pollfds.as_mut_ptr(),
                    pollfds.len() as libc::nfds_t,
                    timeout_ms,
                )
            };
            if result >= 0 {
                return Ok(());
            }
            // An interrupted poll retries with the full timeout rather than
            // the remainder, so a signal can extend the wait.
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error.into());
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = timeout;
        anyhow::bail!("worker progress descriptors require Linux poll support")
    }
}
