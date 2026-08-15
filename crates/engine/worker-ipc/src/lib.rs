//! Worker IPC implementations over iceoryx2 request-response services.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod death_watch;
mod multiproc;
mod stage_router;
mod uniproc;

pub use multiproc::MultiprocExecutor;
pub use stage_router::StageRouter;
pub use uniproc::{LaneConfig, UniprocExecutor, WorkerLaunchConfig};

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
