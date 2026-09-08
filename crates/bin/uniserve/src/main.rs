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
            let mut checkpoint_source = None;
            if !args.runtime.sim && !args.runtime.worker_process.worker_stub {
                let inspected = inspect_checkpoint(
                    &args.runtime.worker_python,
                    &args.runtime.model,
                    args.runtime.revision.as_deref(),
                    args.runtime.worker_ranks,
                    None,
                )
                .await?;
                let description = inspected["description"]
                    .as_str()
                    .context("checkpoint inspector omitted model description")?
                    .parse()?;
                if args
                    .runtime
                    .model_description
                    .is_some_and(|value| value != description)
                {
                    anyhow::bail!("--model-description contradicts the checkpoint architecture");
                }
                if args.runtime.revision.is_some() && inspected["contract"].is_null() {
                    anyhow::bail!("--revision requires the supported H3 checkpoint contract");
                }
                if inspected["repository"].is_string() && !inspected["contract"].is_null() {
                    checkpoint_source = Some(args.runtime.model.clone());
                }
                args.runtime.model_description = Some(description);
                args.runtime.model_contract = inspected
                    .get("contract")
                    .filter(|value| !value.is_null())
                    .cloned();
                args.runtime
                    .served_model_name
                    .get_or_insert_with(|| args.runtime.model.clone());
                args.runtime.model = inspected["model_path"]
                    .as_str()
                    .context("checkpoint inspector omitted model path")?
                    .to_owned();
            } else if args.runtime.model_description.is_none() {
                anyhow::bail!("simulation and stub workers require --model-description");
            }
            let settings = args.runtime.engine_settings();
            info!(model = %args.runtime.model, contract = ?args.runtime.model_contract,
                workers = ?settings.workers, resident_requests = settings.max_num_seqs,
                python = %args.runtime.worker_python.display(), "resolved deployment");
            if let Some(source) = checkpoint_source {
                let downloaded = inspect_checkpoint(
                    &args.runtime.worker_python,
                    &source,
                    args.runtime.revision.as_deref(),
                    args.runtime.worker_ranks,
                    Some("--download"),
                )
                .await?;
                args.runtime.model = downloaded["model_path"]
                    .as_str()
                    .context("checkpoint download omitted its local path")?
                    .to_owned();
            }
            uniserve_server::serve(args.to_uniserve_config(), shutdown_signal()).await
        }
        Command::Doctor(args) => {
            let result = inspect_checkpoint(
                &args.worker_python,
                &args.model,
                args.revision.as_deref(),
                args.worker_ranks,
                Some("--doctor"),
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
    }
}

/// Run the installed worker's catalog/provider validation before allocating model weights.
async fn inspect_checkpoint(
    python: &std::path::Path,
    model: &str,
    revision: Option<&str>,
    ranks: usize,
    action: Option<&str>,
) -> Result<serde_json::Value> {
    let mut command = tokio::process::Command::new(python);
    command.args([
        "-m",
        "uniserve_worker.bootstrap.inspect_model",
        "--model",
        model,
        "--worker-ranks",
        &ranks.to_string(),
    ]);
    if let Some(action) = action {
        command.arg(action);
    }
    if let Some(revision) = revision {
        command.args(["--revision", revision]);
    }
    let output = command
        .stderr(std::process::Stdio::inherit())
        .output()
        .await
        .with_context(|| {
            format!(
                "could not run installed worker interpreter {}",
                python.display()
            )
        })?;
    anyhow::ensure!(
        output.status.success(),
        "checkpoint preflight failed ({})",
        output.status
    );
    serde_json::from_slice(&output.stdout).context("invalid checkpoint inspector response")
}
