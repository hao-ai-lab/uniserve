//! HTTP router construction, listener binding, middleware, and shutdown.
//!
//! This is the server crate's HTTP frontend: [`serve`] builds the shared
//! `AppState` (serving runtime, engine client, video jobs), binds the
//! configured TCP, Unix-domain, or inherited listener, and runs the axum
//! router from `routes::build_router` until shutdown. Route handlers translate
//! OpenAI-compatible requests into serving-runtime calls; the runtime and the
//! engine own tokenization, scheduling, and generation.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod listener;
mod middleware;
mod routes;
#[cfg(test)]
pub(crate) mod test_support;
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

/// Runs the configured HTTP server until the shutdown token is cancelled.
///
/// Shutdown has one budget of `config.shutdown_timeout`, measured from
/// cancellation: axum first drains open connections gracefully, and the server
/// is aborted if the deadline passes first. `AppState::shutdown` then cancels
/// the detached video jobs and waits, against the same deadline, for every
/// other reference to the state to drop before shutting down the serving
/// runtime; it skips that runtime shutdown if the deadline passes first. A
/// zero timeout aborts the server immediately.
///
/// # Errors
///
/// Returns an error when the configuration is invalid, building the state
/// fails, the listener cannot be bound or report its address, the server
/// fails, or runtime shutdown fails. Cancellation before the state is built
/// returns `Ok(())` without binding a listener.
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

    // Only TCP connections take `TCP_NODELAY`; Unix-domain streams (the
    // `Either::Right` side) are left unchanged. A failure is traced and the
    // connection is served without the option.
    let listener = listener.tap_io(|io| {
        if let Either::Left(tcp_stream) = io
            && let Err(error) = tcp_stream.set_nodelay(true)
        {
            trace!(%error, "failed to enable TCP_NODELAY on accepted HTTP connection");
        }
    });

    // The watcher publishes the shutdown deadline once cancellation arrives and
    // fires `force_shutdown` when it expires, bounding the graceful drain.
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

    // The graceful shutdown and the watcher wake on the same token, so the
    // server can finish draining before the watcher has published the
    // deadline; the fallback then measures the budget from the current time.
    let shutdown_deadline = shutdown_deadline
        .get()
        .copied()
        .unwrap_or_else(|| Instant::now() + config.shutdown_timeout);
    state.shutdown(shutdown_deadline).await
}
