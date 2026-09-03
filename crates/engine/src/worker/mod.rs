//! Worker process construction, IPC execution, rank aggregation, and respawn.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod death_watch;
mod multiproc;
mod staged_executor;
mod uniproc;

pub use multiproc::MultiprocExecutor;
pub use staged_executor::StagedExecutor;
pub use uniproc::{FlashInferBackend, FlashInferBackendParseError, LaneConfig, UniprocExecutor};

/// Everything required to create or recreate one worker rank process.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct WorkerProcessArgs {
    pub python: std::path::PathBuf,
    pub model: String,
    pub device: String,
    pub world_size: usize,
    pub pipeline_depth: usize,
    pub req_slot_cap: usize,
    pub resp_slot_cap: usize,
    pub kv_token_capacity: Option<u64>,
    pub block_size: u32,
    pub max_batch_operations: u32,
    pub max_batch_tokens: u32,
    pub attention_backend: uniserve_worker_ipc::AttentionBackend,
    pub supported_ops: Vec<uniserve_worker_ipc::OpKind>,
    pub transfer_backend: crate::executor::TransferBackend,
    pub stub: bool,
    pub load_format: String,
    pub download_dir: Option<std::path::PathBuf>,
    pub load_threads: Option<u32>,
    pub checksum_manifest: Option<std::path::PathBuf>,
    pub model_dtype: uniserve_core::ModelDtype,
    pub quantization_config: serde_json::Value,
    pub kv_cache_dtype: Option<uniserve_core::KvCacheDtype>,
    pub kv_memory_fraction: f64,
    pub mesh: Option<String>,
    pub tp_backend: Option<String>,
    pub lanes: Vec<LaneConfig>,
    pub cuda_graph: bool,
    pub decode_graph_batch_sizes: Option<String>,
    pub prefill_cuda_graph: bool,
    pub prefill_graph_token_sizes: Option<String>,
    pub flow_graph_batch_sizes: Option<String>,
    pub flow_graph_shapes: Option<String>,
    pub flashinfer_workspace_size: u64,
    pub flashinfer_use_tensor_core: Option<String>,
    pub flashinfer_decode_backend: FlashInferBackend,
    pub flashinfer_prefill_backend: FlashInferBackend,
    pub flashinfer_decode_split_tile_size: Option<u32>,
    pub flashinfer_prefill_split_tile_size: Option<u32>,
    pub flashinfer_disable_split_kv: bool,
    pub flashinfer_fast_decode_plan: bool,
    pub max_model_len: u32,
    pub max_video_seconds: f64,
}

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
