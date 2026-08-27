//! Worker IPC implementations over iceoryx2 request-response services.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod death_watch;
mod multiproc;
mod staged_executor;
mod uniproc;

pub use multiproc::MultiprocExecutor;
pub use staged_executor::StagedExecutor;
pub use uniproc::{
    FlashInferBackend, FlashInferBackendParseError, LaneConfig, UniprocExecutor, WorkerLaunchConfig,
};

/// Complete context required to spawn one worker pool.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerSpawnSpec {
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
    pub worker_kind: Option<crate::executor::WorkerKind>,
    pub transfer_backend: crate::executor::TransferBackend,
    pub launch: WorkerLaunchConfig,
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
