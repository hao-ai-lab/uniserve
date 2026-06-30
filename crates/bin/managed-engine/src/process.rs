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
    use expect_test::expect;

    use super::{ManagedEngineConfig, allocate_handshake_port};

    #[test]
    fn command_snapshot() {
        let config = ManagedEngineConfig {
            binary: "/usr/local/bin/uniserve".to_string(),
            model: "models/ThinkMorph-7B".to_string(),
            handshake_host: "127.0.0.1".to_string(),
            handshake_port: 62100,
            engine_index: 1,
            engine_args: vec![
                "--device".to_string(),
                "cuda".to_string(),
                "--block-size".to_string(),
                "256".to_string(),
                "--sim".to_string(),
            ],
        };
        let command = config.to_command();
        let args = command.get_args().collect::<Vec<_>>();

        expect![[r#"
            [
                "engine",
                "models/ThinkMorph-7B",
                "--handshake-address",
                "tcp://127.0.0.1:62100",
                "--engine-index",
                "1",
                "--device",
                "cuda",
                "--block-size",
                "256",
                "--sim",
            ]
        "#]]
        .assert_debug_eq(&args);
    }

    #[test]
    fn allocate_handshake_port_returns_non_zero_port() {
        let port = allocate_handshake_port("127.0.0.1").unwrap();
        assert_ne!(port, 0);
    }

    /// Build a config with the given engine_args and otherwise fixed fields,
    /// so each test only varies the dimension it asserts on.
    fn config_with(engine_index: u32, engine_args: Vec<String>) -> ManagedEngineConfig {
        ManagedEngineConfig {
            binary: "/opt/uniserve/bin/uniserve".to_string(),
            model: "org/Model-32B".to_string(),
            handshake_host: "127.0.0.1".to_string(),
            handshake_port: 5557,
            engine_index,
            engine_args,
        }
    }

    /// Collect `to_command()` args as owned `String`s for easy slice asserts.
    fn rendered_args(config: &ManagedEngineConfig) -> Vec<String> {
        config
            .to_command()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn handshake_address_formats_tcp_host_and_port() {
        let config = ManagedEngineConfig {
            binary: "uniserve".to_string(),
            model: "m".to_string(),
            handshake_host: "10.0.0.7".to_string(),
            handshake_port: 49152,
            engine_index: 0,
            engine_args: vec![],
        };
        assert_eq!(config.handshake_address(), "tcp://10.0.0.7:49152");
    }

    #[test]
    fn to_command_targets_the_configured_binary() {
        let config = config_with(0, vec![]);
        let command = config.to_command();
        assert_eq!(
            command.get_program().to_string_lossy(),
            "/opt/uniserve/bin/uniserve"
        );
    }

    #[test]
    fn to_command_uses_the_engine_subcommand_as_first_arg() {
        let config = config_with(0, vec![]);
        let args = rendered_args(&config);
        assert_eq!(args.first().map(String::as_str), Some("engine"));
    }

    #[test]
    fn to_command_passes_model_as_positional_after_subcommand() {
        let config = config_with(0, vec![]);
        let args = rendered_args(&config);
        // The model is the positional immediately following `engine`, before any
        // `--` flag.
        assert_eq!(args.first().map(String::as_str), Some("engine"));
        assert_eq!(args.get(1).map(String::as_str), Some("org/Model-32B"));
    }

    #[test]
    fn to_command_renders_handshake_address_flag_with_tcp_value() {
        let config = config_with(0, vec![]);
        let args = rendered_args(&config);
        let idx = args
            .iter()
            .position(|a| a == "--handshake-address")
            .expect("command must carry --handshake-address");
        assert_eq!(args[idx + 1], "tcp://127.0.0.1:5557");
    }

    #[test]
    fn to_command_renders_engine_index_as_decimal_string() {
        let config = config_with(3, vec![]);
        let args = rendered_args(&config);
        let idx = args
            .iter()
            .position(|a| a == "--engine-index")
            .expect("command must carry --engine-index");
        assert_eq!(args[idx + 1], "3");
    }

    #[test]
    fn to_command_renders_zero_engine_index_explicitly() {
        let config = config_with(0, vec![]);
        let args = rendered_args(&config);
        let idx = args
            .iter()
            .position(|a| a == "--engine-index")
            .expect("command must carry --engine-index");
        assert_eq!(args[idx + 1], "0");
    }

    #[test]
    fn to_command_appends_engine_args_verbatim_and_in_order() {
        let engine_args = vec![
            "--device".to_string(),
            "cuda".to_string(),
            "--block-size".to_string(),
            "256".to_string(),
            "--sim".to_string(),
        ];
        let config = config_with(0, engine_args.clone());
        let args = rendered_args(&config);

        // engine_args occupy the tail of the command line, byte-for-byte and in
        // the same order they were supplied.
        let tail = &args[args.len() - engine_args.len()..];
        assert_eq!(tail, engine_args.as_slice());
    }

    #[test]
    fn to_command_engine_args_follow_the_engine_index_block() {
        let engine_args = vec!["--attention-backend".to_string(), "auto".to_string()];
        let config = config_with(2, engine_args);
        let args = rendered_args(&config);

        let index_pos = args
            .iter()
            .position(|a| a == "--engine-index")
            .expect("command must carry --engine-index");
        let first_engine_arg = args
            .iter()
            .position(|a| a == "--attention-backend")
            .expect("forwarded engine arg must appear");
        // Forwarded args come strictly after the fixed engine-index block.
        assert!(first_engine_arg > index_pos + 1);
    }

    #[test]
    fn to_command_with_empty_engine_args_has_only_the_fixed_prefix() {
        let config = config_with(0, vec![]);
        let args = rendered_args(&config);
        assert_eq!(
            args,
            vec![
                "engine".to_string(),
                "org/Model-32B".to_string(),
                "--handshake-address".to_string(),
                "tcp://127.0.0.1:5557".to_string(),
                "--engine-index".to_string(),
                "0".to_string(),
            ]
        );
    }

    #[test]
    fn to_command_preserves_duplicate_engine_args_without_dedup() {
        // The supervisor forwards verbatim; it must not collapse repeats.
        let engine_args = vec![
            "--disable-model-arch".to_string(),
            "ArchA".to_string(),
            "--disable-model-arch".to_string(),
            "ArchB".to_string(),
        ];
        let config = config_with(0, engine_args.clone());
        let args = rendered_args(&config);
        let occurrences = args.iter().filter(|a| *a == "--disable-model-arch").count();
        assert_eq!(occurrences, 2);
        let tail = &args[args.len() - engine_args.len()..];
        assert_eq!(tail, engine_args.as_slice());
    }
}
