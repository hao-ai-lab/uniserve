#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
//! UniServe command-line entry point and process lifecycle orchestration.

mod cli;
mod logging;

use std::env;

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::cli::{Cli, Command};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const TOKIO_WORKER_THREADS_ENV: &str = "TOKIO_WORKER_THREADS";
const DEFAULT_MAX_TOKIO_WORKER_THREADS: usize = 32;
const TOKIO_THREAD_STACK_BYTES: usize = 8 * 1024 * 1024;

/// Caps the default number of Tokio worker threads when `TOKIO_WORKER_THREADS` is unset.
///
/// The cap bounds scheduling overhead on machines with many logical CPUs.
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

/// Returns a cancellation token triggered by Ctrl-C or SIGTERM.
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

/// Starts the UniServe command-line process.
fn main() -> Result<()> {
    let cli = Cli::parse();
    logging::init_tracing(cli.log_level.as_deref(), cli.log_level_http.as_deref());

    let mut runtime = tokio::runtime::Builder::new_multi_thread();
    runtime.enable_all();
    // Chat-template rendering runs on Tokio's blocking pool. Complex model
    // templates require more than the platform's small pthread default.
    runtime.thread_stack_size(TOKIO_THREAD_STACK_BYTES);
    if let Some(worker_threads) = tokio_worker_threads() {
        runtime.worker_threads(worker_threads);
    }

    runtime
        .build()
        .context("failed to build Tokio runtime")?
        .block_on(async_main(cli))
}

/// Runs the UniServe command-line process.
async fn async_main(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Serve(args) => {
            uniserve_server::serve(args.to_uniserve_config(), shutdown_signal()).await
        }
    }
}
