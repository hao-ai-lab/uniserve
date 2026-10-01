//! Edge-triggered worker-exit notification for event-driven executors.
//!
//! A watcher fires the rank channel's death `Wake` (`RankChannel::death_wake`)
//! when a child exits, allowing the scheduler to share one wait boundary for
//! results, commands, and worker death.
//!
//! Only a rank process this engine spawned can have a watcher;
//! `PendingRank::adopt` attempts to start one. Linux uses `pidfd` readiness
//! without reaping the child; `RankProcess` reaps it later through
//! `Child::try_wait` or `Child::wait`. Other platforms have no watcher, and a
//! local rank's exit is then noticed by `RankProcess::check_worker`, which
//! `RankProcess::poll_batch` runs at least once per `WORKER_CHECK_INTERVAL`
//! while it waits.

#[cfg(target_os = "linux")]
mod imp {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread::JoinHandle;

    use uniserve_worker_ipc::Wake;

    /// How long each `poll()` blocks before re-checking the stop flag, bounding
    /// how long teardown waits to join this thread when the worker is still
    /// alive. Worker death itself wakes `poll()` immediately regardless.
    const POLL_TICK_MS: libc::c_int = 200;

    /// A background thread that owns a `pidfd` for one child process.
    ///
    /// Dropping the watcher stops and joins the thread. `RankProcess` drops it
    /// before it kills or shuts down the child, so an intentional exit is not
    /// reported as a death.
    pub(crate) struct DeathWatcher {
        stop: Arc<AtomicBool>,
        handle: Option<JoinHandle<()>>,
    }

    impl DeathWatcher {
        /// Starts a pidfd watcher that wakes the engine when the child exits.
        ///
        /// On exit the watcher also sets `startup_abort`, the startup
        /// cancellation flag shared by the ranks launched together with this
        /// one. Returns `None` when the descriptor cannot be opened or the
        /// thread cannot be started.
        pub(crate) fn spawn(pid: u32, wake: Wake, startup_abort: Arc<AtomicBool>) -> Option<Self> {
            // SAFETY: `pidfd_open` receives the PID of a child this engine
            // spawned and has not reaped. A negative return leaves exit
            // detection to `RankProcess::check_worker`.
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
    fn run(pidfd: libc::c_int, stop: &AtomicBool, wake: &Wake, startup_abort: &AtomicBool) {
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
                // The release store pairs with the acquire loads in
                // `RankProcess::check_worker` and `RankProcess::request_close`. Once the
                // group is ready its ranks stop consulting the flag, so a later
                // exit only wakes.
                startup_abort.store(true, Ordering::Release);
                // The child has exited. Fire the death wake once and stop;
                // `RankProcess` reaps the zombie.
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
        /// Stops the thread and joins it, waiting at most about one
        /// `POLL_TICK_MS` while the child is still alive.
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
    use uniserve_worker_ipc::Wake;

    pub(crate) struct DeathWatcher;

    impl DeathWatcher {
        /// Returns `None`: this platform has no watcher.
        pub(crate) fn spawn(
            _pid: u32,
            _wake: Wake,
            _startup_abort: std::sync::Arc<std::sync::atomic::AtomicBool>,
        ) -> Option<Self> {
            // No pidfd equivalent off Linux; `RankProcess::check_worker`
            // notices the exit when it next runs.
            None
        }
    }
}

pub(crate) use imp::DeathWatcher;
