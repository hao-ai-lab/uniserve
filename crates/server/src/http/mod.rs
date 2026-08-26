//! HTTP serving surface for UniServe's configured public APIs.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod listener;
mod middleware;
mod routes;
mod utils;

use std::sync::{Arc, OnceLock};

use anyhow::{Context as _, Result};
use axum::serve::ListenerExt as _;
use tokio::time::{Instant, sleep_until};
use tokio_util::either::Either;
use tokio_util::sync::CancellationToken;
use tracing::{info, trace, warn};

use crate::http::listener::Listener;
use crate::{Config, build_state};

pub use crate::openai::ApiError;
pub use routes::build_router;

/// Run the configured HTTP server until the shutdown token is cancelled.
pub async fn serve(config: Config, shutdown: CancellationToken) -> Result<()> {
    config
        .validate()
        .context("invalid HTTP frontend configuration")?;

    let state = tokio::select! {
        result = build_state(&config) => result?,
        _ = shutdown.cancelled() => return Ok(()),
    };
    let listener = Listener::bind(&config.listener_mode)
        .await
        .context("failed to bind listener for HTTP server")?;
    let bind_address = listener.local_addr()?;
    let model = state.served_model_name().to_owned();
    let app = build_router(Arc::clone(&state));

    info!(%bind_address, %model, "starting HTTP server");

    let listener = listener.tap_io(|io| {
        if let Either::Left(tcp_stream) = io
            && let Err(error) = tcp_stream.set_nodelay(true)
        {
            trace!(%error, "failed to enable TCP_NODELAY on accepted HTTP connection");
        }
    });

    let force_shutdown = CancellationToken::new();
    let shutdown_deadline = Arc::new(OnceLock::new());
    tokio::spawn({
        let shutdown = shutdown.clone();
        let force_shutdown = force_shutdown.clone();
        let shutdown_deadline = Arc::clone(&shutdown_deadline);
        let shutdown_timeout = config.shutdown_timeout;
        async move {
            shutdown.cancelled().await;
            let deadline = Instant::now() + shutdown_timeout;
            let _ = shutdown_deadline.set(deadline);
            if shutdown_timeout.is_zero() {
                force_shutdown.cancel();
            } else {
                sleep_until(deadline).await;
                force_shutdown.cancel();
            }
        }
    });

    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown.cancelled_owned());
    tokio::select! {
        result = server => result.context("HTTP server failed")?,
        _ = force_shutdown.cancelled() => {
            warn!("HTTP graceful shutdown deadline elapsed; aborting server");
        }
    }

    let shutdown_deadline = shutdown_deadline
        .get()
        .copied()
        .unwrap_or_else(|| Instant::now() + config.shutdown_timeout);
    state.shutdown(shutdown_deadline).await
}
