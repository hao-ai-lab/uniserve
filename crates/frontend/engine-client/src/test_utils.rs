//! In-process test harness for the scriptable [`crate::mock`] engine backend.
//!
//! A test connects an [`crate::EngineCoreClient`] in mock mode and hands the
//! returned [`crate::MockEngine`] to [`spawn_mock_engine_task`], whose closure
//! scripts the engine (receive requests, send outputs) entirely in-process.

use std::future::Future;
use std::pin::Pin;

use tokio::sync::oneshot;

use crate::MockEngine;

/// A boxed future returned by a mock-engine script closure.
pub type MockScriptFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

/// Spawn a task that runs `script` against the mock engine, then stays alive
/// (holding the engine handle, and thus the request streams) until the returned
/// shutdown sender is dropped or fired by the test.
pub fn spawn_mock_engine_task<F>(
    mock: MockEngine,
    script: F,
) -> (oneshot::Sender<()>, tokio::task::JoinHandle<()>)
where
    F: FnOnce(MockEngine) -> MockScriptFuture + Send + 'static,
{
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let join_handle = tokio::spawn(async move {
        script(mock).await;
        let _ = shutdown_rx.await;
    });
    (shutdown_tx, join_handle)
}
