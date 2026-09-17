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
        Command::Serve(mut args) => {
            args.runtime
                .served_model_name
                .get_or_insert_with(|| args.runtime.model.clone());
            // The checkpoint decides whether this deployment serves video, and
            // that choice sizes the queue, batch and IPC slot budgets.
            let is_media =
                uniserve_server::profile::assets::is_media_checkpoint(&args.runtime.model).await;
            let settings = args.runtime.engine_settings(is_media);
            info!(model = %args.runtime.model,
                workers = ?settings.workers, resident_requests = settings.max_num_seqs,
                python = %args.runtime.worker_python.display(), "resolved deployment");
            uniserve_server::serve(args.to_uniserve_config(is_media), shutdown_signal()).await
        }
    }
}
