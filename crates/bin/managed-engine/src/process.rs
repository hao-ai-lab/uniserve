use std::io;
use std::net::TcpListener;
use std::process::{Command as StdCommand, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::info;

const MIN_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Allocate one ephemeral TCP port for the managed engine handshake on the
/// given host.
pub fn allocate_handshake_port(host: &str) -> Result<u16> {
    let listener = TcpListener::bind((host, 0)).context("failed to allocate handshake port")?;
    let port = listener
        .local_addr()
        .context("failed to inspect allocated handshake listener address")?
        .port();
    Ok(port)
}

/// Spawn configuration for one managed headless `uniserve engine` process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedEngineConfig {
    /// Path to the `uniserve` binary hosting the `engine` subcommand
    /// (normally `std::env::current_exe`).
    pub binary: String,
    /// Model identifier passed to `uniserve engine <model>`.
    pub model: String,
    /// Host portion of the engine handshake endpoint.
    pub handshake_host: String,
    /// Port portion of the engine handshake endpoint.
    pub handshake_port: u16,
    /// Engine index of this replica within the deployment.
    pub engine_index: u32,
    /// Extra CLI arguments forwarded verbatim to `uniserve engine`
    /// (`--sim`, `--device`, `--block-size`, …).
    pub engine_args: Vec<String>,
}

impl ManagedEngineConfig {
    /// Render the handshake address the frontend binds and the engine dials.
    pub fn handshake_address(&self) -> String {
        format!("tcp://{}:{}", self.handshake_host, self.handshake_port)
    }

    /// Build the concrete command line for the managed headless engine.
    pub fn to_command(&self) -> StdCommand {
        let mut command = StdCommand::new(&self.binary);
        command
            .arg("engine")
            .arg(&self.model)
            .arg("--handshake-address")
            .arg(self.handshake_address())
            .arg("--engine-index")
            .arg(self.engine_index.to_string())
            .args(&self.engine_args);
        command
    }
}

/// RAII-style handle for one managed headless engine subprocess.
#[derive(Clone)]
pub struct ManagedEngineHandle {
    child: Arc<Mutex<Child>>,
    /// PID captured at spawn time. Cached so signalling (and shutdown) never
    /// has to lock the `Child`, which lets `wait_for_exit` hold the lock across
    /// an event-driven `Child::wait` await without deadlocking a concurrent
    /// shutdown.
    pid: Option<u32>,
    shutdown_started: Arc<AtomicBool>,
}

impl ManagedEngineHandle {
    /// Spawn one managed headless engine and return a handle for monitoring it.
    pub async fn spawn(config: ManagedEngineConfig) -> Result<Self> {
        let command = config.to_command();
        info!(
            handshake_address = %config.handshake_address(),
            engine_index = config.engine_index,
            ?command,
            "starting managed headless engine"
        );

        let mut command = Command::from(command);
        command
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());

        process_group::configure(&mut command);

        let child = command.spawn().context("failed to spawn managed engine")?;
        let pid = child.id();

        Ok(Self {
            child: Arc::new(Mutex::new(child)),
            pid,
            shutdown_started: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Poll whether the managed engine has exited yet.
    pub async fn try_wait(&self) -> Result<Option<ExitStatus>> {
        let mut child = self.child.lock().await;
        child
            .try_wait()
            .context("failed to poll the status of managed engine")
    }

    /// Wait until the managed engine exits.
    /// Event-driven: awaits `tokio::process::Child::wait` (which registers for
    /// the child's SIGCHLD) rather than busy-polling `try_wait`. The lock is
    /// held across the await, but signalling reads the cached `pid` instead of
    /// the `Child`, so a concurrent `shutdown` cannot deadlock against it.
    pub async fn wait_for_exit(&self) -> Result<ExitStatus> {
        let mut child = self.child.lock().await;
        child
            .wait()
            .await
            .context("failed to wait for managed engine to exit")
    }

    /// Terminate the managed engine process group and wait for it to stop.
    pub async fn shutdown(&self, timeout: Duration) -> Result<()> {
        if self.shutdown_started.swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        // Use the PID captured at spawn time so we never lock the `Child` here;
        // `wait_for_exit` may be holding that lock across its event-driven
        // `Child::wait` await.
        let Some(pid) = self.pid else {
            return Ok(());
        };

        // Enforce a minimum shutdown timeout to give the engine process (and
        // its Python worker) enough time to clean up.
        let shutdown_timeout = std::cmp::max(timeout, MIN_SHUTDOWN_TIMEOUT);

        // First, try to gracefully terminate.
        info!(
            pid,
            ?shutdown_timeout,
            "shutting down managed engine with SIGTERM"
        );
        process_group::terminate(pid)?;

        // Wait for the process to exit on its own.
        match tokio::time::timeout(shutdown_timeout, self.wait_for_exit()).await {
            Ok(Ok(_)) => return Ok(()),
            Ok(Err(error)) => return Err(error),
            Err(_) => {}
        }

        // If it doesn't exit within the timeout, force kill it.
        info!(
            pid,
            "managed engine did not exit within timeout, sending SIGKILL"
        );
        process_group::kill(pid)?;

        let _ = self.wait_for_exit().await?;
        Ok(())
    }
}

/// Process group helper functions for managing the engine subprocess and its
/// children (the forward-only Python worker) in a platform-aware way.
mod process_group {
    use super::*;

    /// Place the engine child into its own process group so `serve` can tear
    /// down the whole subtree (engine + its Python worker) rather than just
    /// the immediate process.
    pub(super) fn configure(command: &mut Command) {
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }

    /// Send SIGTERM to the managed engine process group.
    pub(super) fn terminate(pid: u32) -> Result<()> {
        signal(pid, libc::SIGTERM)
    }

    /// Send SIGKILL to the managed engine process group.
    pub(super) fn kill(pid: u32) -> Result<()> {
        signal(pid, libc::SIGKILL)
    }

    /// Deliver one signal to the managed engine process group.
    fn signal(pid: u32, signal: i32) -> Result<()> {
        let rc = unsafe { libc::kill(-(pid as i32), signal) };
        if rc == 0 {
            return Ok(());
        }

        let error = io::Error::last_os_error();
        if matches!(error.raw_os_error(), Some(code) if code == libc::ESRCH) {
            return Ok(());
        }
        Err(error).context("failed to signal managed engine process group")
    }
}

#[cfg(test)]
mod tests {
    use super::allocate_handshake_port;

    #[test]
    fn handshake_port_allocation_returns_a_valid_port() {
        let port = allocate_handshake_port("127.0.0.1").unwrap();
        assert_ne!(port, 0);
    }
}
