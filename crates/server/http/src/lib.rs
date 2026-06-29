//! HTTP serving surface for UniServe's public APIs.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod listener;
mod middleware;
mod routes;
mod utils;

pub mod error;

use std::sync::{Arc, OnceLock};

use anyhow::{Context as _, Result};
use axum::{Router, serve::ListenerExt as _};
use tokio::net::TcpListener;
use tokio::time::{Instant, sleep_until};
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::either::Either;
use tokio_util::sync::CancellationToken;
use tonic::transport::Server as TonicServer;
use tracing::{info, trace, warn};
use uniserve_server_app::{Config, HttpListenerMode};

pub use error::ApiError;
pub use routes::build_router;

use crate::listener::Listener;

/// Per-request handler timeout for the gRPC server.

/// Bounds how long any single RPC handler may run before tonic responds with a
/// timeout, preventing a slow or stuck handler from pinning a connection open
/// indefinitely.
const GRPC_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
/// Upper bound on concurrently buffered requests per gRPC connection.
const GRPC_CONCURRENCY_LIMIT_PER_CONNECTION: usize = 256;
/// HTTP/2 `SETTINGS_MAX_CONCURRENT_STREAMS` advertised to gRPC clients.
const GRPC_MAX_CONCURRENT_STREAMS: u32 = 256;
/// Interval between HTTP/2 keepalive PING frames on idle gRPC connections.
const GRPC_HTTP2_KEEPALIVE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);
/// Time to wait for a keepalive PING acknowledgement before closing the connection.
const GRPC_HTTP2_KEEPALIVE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);
/// Maximum HTTP/2 frame size accepted on gRPC connections (16 MiB).
const GRPC_MAX_FRAME_SIZE: u32 = 16 * 1024 * 1024;
/// Cap on pending reset streams before a GOAWAY is sent (HTTP/2 "Rapid Reset" mitigation).
const GRPC_MAX_PENDING_ACCEPT_RESET_STREAMS: usize = 32;

/// Run the HTTP server until the supplied shutdown token is cancelled.
pub async fn serve(config: Config, shutdown: CancellationToken) -> Result<()> {
    serve_with_router_extension(config, shutdown, |router| router).await
}

/// Run the HTTP server with an opt-in router extension.

/// The extension receives the finalized router and can merge additional routes
/// before the server starts accepting requests.
pub async fn serve_with_router_extension<F>(
    config: Config,
    shutdown: CancellationToken,
    extend_router: F,
) -> Result<()>
where
    F: FnOnce(Router) -> Router,
{
    config
        .validate()
        .context("invalid HTTP frontend configuration")?;

    let state = tokio::select! {
        result = uniserve_server_app::build_state(&config) => result?,
        _ = shutdown.cancelled() => return Ok(()),
    };
    let listener = Listener::bind(&config.listener_mode)
        .await
        .context("failed to bind listener for HTTP server")?;
    let bind_address = listener.local_addr()?;
    let model = state.primary_model_name().to_owned();
    let app = extend_router(build_router(Arc::clone(&state)));

    let grpc_setup = if let Some(grpc_port) = config.grpc_port {
        let grpc_host = match &config.listener_mode {
            HttpListenerMode::BindTcp { host, .. } => host.as_str(),
            HttpListenerMode::BindUnix { .. } | HttpListenerMode::InheritedFd { .. } => "0.0.0.0",
        };
        let grpc_listener = TcpListener::bind((grpc_host, grpc_port))
            .await
            .with_context(|| format!("failed to bind gRPC listener on {grpc_host}:{grpc_port}"))?;
        let addr = grpc_listener.local_addr()?;
        let svc = uniserve_server_grpc::GenerateServer::new(
            uniserve_server_grpc::GenerateServiceImpl::new(Arc::clone(&state)),
        );
        info!(%addr, "starting gRPC server");
        Some((grpc_listener, svc))
    } else {
        None
    };

    info!(%bind_address, %model, "starting HTTP server");

    let listener = listener.tap_io(|io| {
        if let Either::Left(tcp_stream) = io
            && let Err(err) = tcp_stream.set_nodelay(true)
        {
            trace!(error = %err, "failed to enable TCP_NODELAY on accepted HTTP connection");
        }
    });

    let server_shutdown = shutdown.child_token();
    let force_shutdown = CancellationToken::new();
    let shutdown_deadline = Arc::new(OnceLock::new());

    tokio::spawn({
        let shutdown = server_shutdown.clone();
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

    let http_fut = {
        let shutdown = server_shutdown.child_token();
        let server_shutdown = server_shutdown.clone();
        let force_shutdown = force_shutdown.clone();
        async move {
            let server =
                axum::serve(listener, app).with_graceful_shutdown(shutdown.cancelled_owned());

            let result = tokio::select! {
                result = server => result.context("HTTP server failed"),
                _ = force_shutdown.cancelled() => {
                    warn!("HTTP graceful shutdown deadline elapsed; aborting server");
                    Ok(())
                }
            };

            server_shutdown.cancel();
            result
        }
    };

    let grpc_fut = {
        let shutdown = server_shutdown.child_token();
        let server_shutdown = server_shutdown.clone();
        let force_shutdown = force_shutdown.clone();
        async move {
            let Some((grpc_listener, svc)) = grpc_setup else {
                shutdown.cancelled().await;
                return Ok(());
            };
            let server = TonicServer::builder()
                .timeout(GRPC_REQUEST_TIMEOUT)
                .concurrency_limit_per_connection(GRPC_CONCURRENCY_LIMIT_PER_CONNECTION)
                .max_concurrent_streams(GRPC_MAX_CONCURRENT_STREAMS)
                .http2_keepalive_interval(Some(GRPC_HTTP2_KEEPALIVE_INTERVAL))
                .http2_keepalive_timeout(Some(GRPC_HTTP2_KEEPALIVE_TIMEOUT))
                .http2_max_pending_accept_reset_streams(Some(
                    GRPC_MAX_PENDING_ACCEPT_RESET_STREAMS,
                ))
                .max_frame_size(GRPC_MAX_FRAME_SIZE)
 // Note: tonic's TCP-level knobs (tcp_keepalive, tcp_nodelay) are
 // documented as ignored under `serve_with_incoming*`, so they are
 // omitted here; only the HTTP/2-level and concurrency/timeout limits
 // above take effect on this path.
                .add_service(svc)
                .serve_with_incoming_shutdown(
                    TcpListenerStream::new(grpc_listener),
                    shutdown.cancelled_owned(),
                );

            let result = tokio::select! {
                result = server => result.context("gRPC server failed"),
                _ = force_shutdown.cancelled() => {
                    warn!("gRPC graceful shutdown deadline elapsed; aborting server");
                    Ok(())
                }
            };

            server_shutdown.cancel();
            result
        }
    };

    let (http_res, grpc_res) = tokio::join!(http_fut, grpc_fut);
    http_res.and(grpc_res)?;

    let shutdown_deadline = shutdown_deadline
        .get()
        .copied()
        .unwrap_or_else(|| Instant::now() + config.shutdown_timeout);
    state.shutdown(shutdown_deadline).await
}
