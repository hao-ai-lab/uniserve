#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod cli;
mod logging;

use std::env;
use std::process::ExitStatus;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uniserve_managed_engine::{ManagedEngineConfig, ManagedEngineHandle, allocate_handshake_port};

use crate::cli::{Cli, Command, ServeArgs};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const TOKIO_WORKER_THREADS_ENV: &str = "TOKIO_WORKER_THREADS";
const DEFAULT_MAX_TOKIO_WORKER_THREADS: usize = 32;

/// Cap the default number of Tokio worker threads if the user did not
/// explicitly set `TOKIO_WORKER_THREADS` to avoid spawning too many threads on
/// machines with a large number of CPUs, which may lead to excessive context
/// switching and degraded performance.
fn tokio_worker_threads() -> Option<usize> {
    if env::var_os(TOKIO_WORKER_THREADS_ENV).is_some() {
        return None;
    }

    std::thread::available_parallelism()
        .map(|parallelism| {
            let available = parallelism.get();
            let worker_threads = available.min(DEFAULT_MAX_TOKIO_WORKER_THREADS);
            if worker_threads < available {
                info!(
                    available_parallelism = available,
                    capped_worker_threads = worker_threads,
                    "capping tokio worker threads, set {TOKIO_WORKER_THREADS_ENV} to override"
                );
            }
            worker_threads
        })
        .ok()
}

/// Reason that caused a managed `serve` session to stop.
#[derive(Debug)]
enum ShutdownReason {
    Signal,
    Server(anyhow::Error),
    EngineExited(ExitStatus),
}

/// Cancellation token tripped by Ctrl-C or SIGTERM.
fn shutdown_signal() -> CancellationToken {
    let token = CancellationToken::new();
    let shutdown = token.clone();

    tokio::spawn(async move {
        let ctrl_c = async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                warn!(%error, "failed to install Ctrl-C signal handler");
                std::future::pending::<()>().await;
            }
        };

        let sigterm = async {
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(mut signal) => {
                    signal.recv().await;
                }
                Err(error) => {
                    warn!(%error, "failed to install SIGTERM signal handler");
                    std::future::pending::<()>().await;
                }
            }
        };

        tokio::select! {
            _ = ctrl_c => info!("received shutdown signal (Ctrl-C), shutting down..."),
            _ = sigterm => info!("received shutdown signal (SIGTERM), shutting down..."),
        }

        shutdown.cancel();
    });

    token
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    logging::init_tracing(cli.log_level.as_deref(), cli.log_level_http.as_deref());

    let mut runtime = tokio::runtime::Builder::new_multi_thread();
    runtime.enable_all();
    if let Some(worker_threads) = tokio_worker_threads() {
        runtime.worker_threads(worker_threads);
    }

    runtime
        .build()
        .context("failed to build Tokio runtime")?
        .block_on(async_main(cli))
}

async fn async_main(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Serve(args) => {
            if args.runtime.engine_count == 0 {
                // The deliberate single-node default: the engine runs
                // on a thread inside the server with no serialized hop.
                uniserve_server::serve(args.to_uniserve_config(), shutdown_signal()).await
            } else {
                serve_with_engines(*args).await
            }
        }
        // One headless engine process behind the engine wire protocol.
        Command::Engine(args) => {
            uniserve_engine_process::run_engine_proc(args.to_proc_config(), shutdown_signal()).await
        }
    }
}

/// Parse `tcp://host:port` into its host and port parts.
fn parse_tcp_address(address: &str) -> Result<(String, u16)> {
    let rest = address
        .strip_prefix("tcp://")
        .with_context(|| format!("handshake address must be tcp://host:port, got `{address}`"))?;
    let (host, port) = rest
        .rsplit_once(':')
        .with_context(|| format!("handshake address must be tcp://host:port, got `{address}`"))?;
    Ok((
        host.to_string(),
        port.parse().context("invalid handshake port")?,
    ))
}

/// Out-of-process serving (vLLM's process topology): spawn and supervise local
/// `uniserve engine` subprocesses (managed mode), or wait for external engines
/// to dial in (frontend-only mode when `--local-engine-count 0`).
async fn serve_with_engines(args: ServeArgs) -> Result<()> {
    let engine_count = args.runtime.engine_count;
    let local_count = args.runtime.local_engine_count.unwrap_or(engine_count);
    if local_count > engine_count {
        bail!("--local-engine-count ({local_count}) exceeds --engine-count ({engine_count})");
    }

    let (handshake_host, handshake_port) = match &args.runtime.handshake_address {
        Some(address) => parse_tcp_address(address)?,
        None => {
            if local_count < engine_count {
                bail!(
                    "--handshake-address must be set explicitly when external engines dial in \
                     (--local-engine-count < --engine-count)"
                );
            }
            let host = "127.0.0.1".to_string();
            let port = allocate_handshake_port(&host)?;
            (host, port)
        }
    };
    let handshake_address = format!("tcp://{handshake_host}:{handshake_port}");
    info!(
        %handshake_address,
        engine_count,
        local_count,
        "serving with out-of-process engine cores"
    );

    // Spawn the locally managed engines; they dial the handshake (with retry)
    // while the server builds its state and binds the handshake socket.
    let binary = env::current_exe()
        .context("failed to resolve the uniserve binary path")?
        .display()
        .to_string();
    let mut engines = Vec::with_capacity(local_count);
    for engine_index in 0..local_count {
        let config = ManagedEngineConfig {
            binary: binary.clone(),
            model: args.runtime.resolved_model(),
            handshake_host: handshake_host.clone(),
            handshake_port,
            engine_index: engine_index as u32,
            engine_args: args.runtime.engine_cli_args(),
        };
        engines.push(
            ManagedEngineHandle::spawn(config)
                .await
                .with_context(|| format!("failed to start managed engine {engine_index}"))?,
        );
    }

    let connection = uniserve_server::EngineConnection::Handshake {
        handshake_address,
        advertised_host: args.runtime.advertised_host.clone(),
        engine_count,
        ready_timeout: Duration::from_secs(args.runtime.engine_ready_timeout),
    };
    let config = args.to_uniserve_config_with_connection(connection);
    let shutdown_timeout = config.shutdown_timeout;

    let shutdown = shutdown_signal();
    let mut serve_task = {
        let shutdown = shutdown.clone();
        tokio::spawn(async move {
            let result = uniserve_server::serve(config, shutdown).await;
            if result.is_ok() {
                info!("OpenAI server shut down gracefully");
            }
            result
        })
    };

    // Watch for any managed engine exiting unexpectedly. Each engine is
    // awaited event-driven (no busy-poll): one waiter task per engine reports
    // the first exit over a channel, and we surface whichever fires first.
    let engine_exit = {
        let engines = engines.clone();
        async move {
            if engines.is_empty() {
                // Frontend-only mode: nothing local to watch.
                return std::future::pending::<anyhow::Result<ExitStatus>>().await;
            }
            let (tx, mut rx) = tokio::sync::mpsc::channel(engines.len());
            for engine in engines {
                let tx = tx.clone();
                tokio::spawn(async move {
                    // Drop is fine if the receiver already took a result.
                    let _ = tx.send(engine.wait_for_exit().await).await;
                });
            }
            drop(tx);
            // The first exit (or the first error) wins. The detached waiter
            // tasks for the still-running engines keep awaiting their own
            // exit; each one ends on its own once that engine stops (e.g. when
            // the shutdown path below signals it).
            match rx.recv().await {
                Some(result) => result,
                // Unreachable while at least one engine waiter holds a sender,
                // but stay pending rather than spuriously reporting an exit.
                None => std::future::pending::<anyhow::Result<ExitStatus>>().await,
            }
        }
    };

    let shutdown_reason = tokio::select! {
           biased;

    // Received shutdown signal via Ctrl-C or SIGTERM.
           _ = shutdown.cancelled() => ShutdownReason::Signal,

    // A managed engine exited unexpectedly.
           engine_exit = engine_exit => {
               match engine_exit {
                   Ok(status) => {
                       warn!(%status, "managed engine exited, shutting down...");
                       ShutdownReason::EngineExited(status)
                   }
                   Err(error) => ShutdownReason::Server(error.context("failed to monitor managed engine")),
               }
           }

    // Serve task exited unexpectedly.
           serve_result = &mut serve_task => {
               let serve_result = serve_result.context("serve task join failed")?;
               match serve_result {
                   Ok(()) => ShutdownReason::Server(anyhow!(
                       "OpenAI server shut down unexpectedly without error"
                   )),
                   Err(error) => ShutdownReason::Server(error),
               }
           }
       };
    // Regardless of the shutdown reason, broadcast shutdown so all serving
    // tasks are notified.
    shutdown.cancel();

    // Shutdown begins. Terminate the managed engines first (SIGTERM →
    // bounded wait → SIGKILL on the whole engine process group).
    for engine in &engines {
        engine.shutdown(shutdown_timeout).await?;
    }
    if !engines.is_empty() {
        info!("managed engines shut down gracefully");
    }
    // Wait for the API server to shut down gracefully by draining in-flight
    // requests.
    if !matches!(shutdown_reason, ShutdownReason::Server(_)) {
        serve_task.await.context("serve task join failed")??;
    }

    match shutdown_reason {
        ShutdownReason::Signal => Ok(()),
        ShutdownReason::Server(error) => Err(error.context("OpenAI server shut down unexpectedly")),
        ShutdownReason::EngineExited(status) => Err(anyhow!(
            "managed engine exited unexpectedly with status {status}"
        )),
    }
}
