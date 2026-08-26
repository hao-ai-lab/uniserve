//! Edge-triggered worker-death detection for the event-driven boundary.
//!
//! The scheduler park waits on a single event listener over {result, command,
//! death}. Result and command wakes are fired by the transport and the command
//! ingress; this module supplies the third source: a watcher that fires a
//! [`WakeSender`] when the worker child exits.
//!
//! On Linux this uses a `pidfd` (Linux 5.3+) parked on with `poll()`, which
//! goes readable on child exit without reaping it (the executor still reaps via
//! `try_wait`). On other platforms — or if `pidfd_open` is unavailable — the
//! watcher is absent and death is detected by the bounded liveness probe instead.

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
        pub(crate) fn spawn(pid: u32, wake: WakeSender) -> Option<Self> {
            // SAFETY: pidfd_open is a thin syscall wrapper; pid is the child we
            // just spawned. A negative return means the kernel lacks the
            // syscall (pre-5.3) — we fall back to the liveness probe.
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
                .spawn(move || run(pidfd, &stop_thread, &wake))
                .ok()?;
            Some(Self {
                stop,
                handle: Some(handle),
            })
        }
    }

    fn run(pidfd: libc::c_int, stop: &AtomicBool, wake: &WakeSender) {
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
                // The child has exited. Fire the death wake once and stop; the
                // executor reaps the zombie via its own `try_wait`.
                wake.wake();
                break;
            }
            // n == 0: poll timeout; loop to re-check the stop flag.
        }
        // SAFETY: we own this fd and no longer use it.
        unsafe {
            libc::close(pidfd);
        }
    }

    impl Drop for DeathWatcher {
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
        pub(crate) fn spawn(_pid: u32, _wake: WakeSender) -> Option<Self> {
            // No pidfd equivalent off Linux; death is caught by the bounded
            // liveness probe instead.
            None
        }
    }
}

pub(crate) use imp::DeathWatcher;
