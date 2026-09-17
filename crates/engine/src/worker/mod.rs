//! WorkerGroup process configuration, IPC execution, rank aggregation, and recovery.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod death_watch;
mod executor;
mod instance;
mod process;

pub use executor::WorkerExecutor;
pub use instance::WorkerGroup;
use process::RankProcess;
pub use process::{FlashInferBackend, FlashInferBackendParseError, LaneConfig};

/// Backpressure and terminal failures returned by a WorkerGroup run submission.
#[derive(Debug, thiserror::Error)]
pub enum RunSubmitError {
    /// Returns the unaccepted physical submission when the instance has no capacity.
    #[error("worker run queue is full")]
    WouldBlock(uniserve_worker_ipc::ScheduleBatch),
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
    /// Ordered physical members of this WorkerGroup instance.
    pub ranks: Vec<crate::WorkerRank>,
    /// Computation entry membership and parallel geometry.
    pub entries: std::collections::BTreeMap<String, crate::ComponentConfig>,
    /// Launch the worker without model weights, using its deterministic test
    /// model. Only the engine's own IPC and process tests set this; the serving
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
    /// Tokens represented by one physical KV cache block.
    pub block_size: u32,
    /// Maximum operations accepted in one worker run.
    pub max_batch_operations: u32,
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
    /// Fraction of available device memory reserved for the KV cache.
    pub kv_memory_fraction: f64,
    /// Optional device mesh specification for staged model components.
    pub mesh: Option<String>,
    /// Optional process-world communication backend.
    pub distributed_backend: Option<String>,
    /// configuration-static execution lane descriptors.
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
    /// Optional video request shapes whose denoising ladders warmup captures.
    pub video_graph_shapes: Option<String>,
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
}

/// Parks until one descriptor becomes readable or `timeout` expires.
pub(crate) fn park_descriptors(fds: &[i32], timeout: std::time::Duration) -> anyhow::Result<()> {
    if fds.is_empty() {
        std::thread::park_timeout(timeout);
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        let mut pollfds = fds
            .iter()
            .copied()
            .map(|fd| libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            })
            .collect::<Vec<_>>();
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
