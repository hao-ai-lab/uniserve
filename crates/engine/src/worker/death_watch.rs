//! Edge-triggered worker-exit notification for event-driven executors.
//!
//! A watcher signals [`WakeSender`] when a child exits, allowing the scheduler to
//! share one wait boundary for results, commands, and worker death.
//!
//! Linux uses `pidfd` readiness without reaping the child. Unsupported platforms
//! rely on the executor's bounded liveness check.

#[cfg(target_os = "linux")]
mod imp {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread::JoinHandle;

    use uniserve_worker_ipc::WakeSender;

    /// How long each `poll()` blocks before re-checking the stop flag, bounding
    /// how long teardown waits to join this thread when the worker is still
    /// alive. Worker death itself wakes `poll()` immediately regardless.
    const POLL_TICK_MS: libc::c_int = 200;

    pub(crate) struct DeathWatcher {
        stop: Arc<AtomicBool>,
        handle: Option<JoinHandle<()>>,
    }

    impl DeathWatcher {
        /// Starts a pidfd watcher that wakes the engine when the child exits.
        pub(crate) fn spawn(
            pid: u32,
            wake: WakeSender,
            startup_abort: Arc<AtomicBool>,
        ) -> Option<Self> {
            // SAFETY: `pidfd_open` receives the PID of the child just spawned. A
            // negative return leaves liveness monitoring to the caller's probe.
            let pidfd = unsafe {
                libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0 as libc::c_uint)
            };
            if pidfd < 0 {
                return None;
            }
            let pidfd = pidfd as libc::c_int;
            let stop = Arc::new(AtomicBool::new(false));
            let stop_thread = Arc::clone(&stop);
            let handle = std::thread::Builder::new()
                .name("uniserve-worker-death".into())
                .spawn(move || run(pidfd, &stop_thread, &wake, &startup_abort))
                .ok()?;
            Some(Self {
                stop,
                handle: Some(handle),
            })
        }
    }

    /// Polls the process descriptor until exit or an explicit watcher stop.
    fn run(pidfd: libc::c_int, stop: &AtomicBool, wake: &WakeSender, startup_abort: &AtomicBool) {
        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let mut pfd = libc::pollfd {
                fd: pidfd,
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: single valid pollfd, count 1.
            let n = unsafe { libc::poll(&mut pfd as *mut libc::pollfd, 1, POLL_TICK_MS) };
            if n < 0 {
                // EINTR or similar: loop to re-check the stop flag and retry.
                continue;
            }
            if n > 0 && (pfd.revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR)) != 0 {
                // Peers can be alive but waiting for this member during group
                // initialization. Cancel that incomplete startup as one instance.
                startup_abort.store(true, Ordering::Release);
                // The child has exited. Fire the death wake once and stop; the
                // executor reaps the zombie via its own `try_wait`.
                wake.wake();
                break;
            }
            // n == 0: poll timeout; loop to re-check the stop flag.
        }
        // SAFETY: the watcher exclusively owns this valid descriptor and closes it once.
        unsafe {
            libc::close(pidfd);
        }
    }

    impl Drop for DeathWatcher {
        /// Releases resources owned by this value.
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use uniserve_worker_ipc::WakeSender;

    pub(crate) struct DeathWatcher;

    impl DeathWatcher {
        /// Spawns a watcher that reports unexpected worker termination.
        pub(crate) fn spawn(
            _pid: u32,
            _wake: WakeSender,
            _startup_abort: std::sync::Arc<std::sync::atomic::AtomicBool>,
        ) -> Option<Self> {
            // No pidfd equivalent off Linux; death is caught by the bounded
            // liveness probe instead.
            None
        }
    }
}

pub(crate) use imp::DeathWatcher;
