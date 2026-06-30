//! In-process scriptable engine backend for tests.
//!
//! A test connects a [`crate::EngineCoreClient`] in mock mode and drives the
//! returned [`MockEngine`] handle: it receives the client's
//! [`EngineCoreRequest`]s and routes scripted [`EngineCoreOutputs`] back to the
//! matching per-request output streams — the same `EngineCoreRequest` →
//! `EngineCoreOutputs` contract the real engine speaks, but entirely in-process
//! (no GPU, Python, or sockets).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;

use crate::client::{AbortRequest, EngineCoreOutputStream, EngineCoreStreamOutput};
use crate::error::{Error, Result};
use crate::protocol::{EngineCoreOutputs, EngineCoreRequest};

type OutputSender = mpsc::UnboundedSender<Result<EngineCoreStreamOutput>>;
type Routing = Arc<Mutex<HashMap<String, OutputSender>>>;

/// A message the client sends to the mock engine.
#[derive(Debug)]
pub enum MockClientMessage {
    /// A new generate request.
    Add(Box<EngineCoreRequest>),
    /// One or more request ids to abort (from an explicit abort or a dropped
    /// output stream).
    Abort(Vec<String>),
}

/// The client-side mock backend held inside [`crate::EngineCoreClient`].
pub struct MockEngineClient {
    routing: Routing,
    inbound_tx: mpsc::UnboundedSender<MockClientMessage>,
    abort_tx: mpsc::UnboundedSender<AbortRequest>,
    model_name: String,
}

impl MockEngineClient {
    pub(crate) fn model_name(&self) -> &str {
        &self.model_name
    }

    pub(crate) fn call(&self, req: EngineCoreRequest) -> Result<EngineCoreOutputStream> {
        let request_id = req.request_id.clone();
        let (tx, rx) = mpsc::unbounded_channel();
        self.routing
            .lock()
            .map_err(|_| Error::ClientClosed {
                message: "mock engine routing mutex is poisoned".to_string(),
            })?
            .insert(request_id.clone(), tx);
        let _ = self.inbound_tx.send(MockClientMessage::Add(Box::new(req)));
        Ok(EngineCoreOutputStream::new(
            request_id,
            self.abort_tx.clone(),
            rx,
        ))
    }

    pub(crate) fn abort(&self, ids: &[String]) {
        let _ = self.inbound_tx.send(MockClientMessage::Abort(ids.to_vec()));
    }
}

/// The test-side handle used to script the engine.
pub struct MockEngine {
    routing: Routing,
    inbound_rx: mpsc::UnboundedReceiver<MockClientMessage>,
}

impl MockEngine {
    /// Receive the next message from the client (`Add` or `Abort`).
    pub async fn recv(&mut self) -> Option<MockClientMessage> {
        self.inbound_rx.recv().await
    }

    /// Receive the next `Add` request, skipping any aborts.

    /// # Panics

    /// Panics if the client disconnects before sending a request.
    pub async fn recv_request(&mut self) -> EngineCoreRequest {
        loop {
            match self.inbound_rx.recv().await {
                Some(MockClientMessage::Add(req)) => return *req,
                Some(MockClientMessage::Abort(_)) => continue,
                None => panic!("mock engine: client disconnected before sending a request"),
            }
        }
    }

    /// Route scripted outputs to the matching per-request output streams.

    /// An output whose `finish_reason` is set detaches its stream after
    /// delivery. Request ids in `finished_requests` are also detached: if no
    /// terminal output was delivered for them, dropping the sender closes the
    /// stream, which the consumer observes as an unexpected close — matching the
    /// engine's "finished without a final output" semantics.
    pub fn send_outputs(&self, outputs: EngineCoreOutputs) {
        let engine_index = outputs.engine_index;
        let timestamp = outputs.timestamp;
        let Ok(mut routing) = self.routing.lock() else {
            return;
        };
        for output in outputs.outputs {
            let request_id = output.request_id.clone();
            let finished = output.finished();
            if let Some(sender) = routing.get(&request_id) {
                let _ = sender.send(Ok(EngineCoreStreamOutput {
                    engine_index,
                    timestamp,
                    output,
                }));
                if finished {
                    routing.remove(&request_id);
                }
            }
        }
        for request_id in outputs.finished_requests.into_iter().flatten() {
            routing.remove(&request_id);
        }
    }
}

/// Build a mock-backed client plus its scripting handle.
pub fn connect_mock(model_name: impl Into<String>) -> (MockEngineClient, MockEngine) {
    let routing: Routing = Arc::new(Mutex::new(HashMap::new()));
    let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
    let (abort_tx, mut abort_rx) = mpsc::unbounded_channel::<AbortRequest>();

    // A dropped output stream surfaces to the mock as an abort, so tests can
    // observe client-driven cancellation just as they did over ZMQ.
    {
        let inbound_tx = inbound_tx.clone();
        tokio::spawn(async move {
            while let Some(request) = abort_rx.recv().await {
                let _ = inbound_tx.send(MockClientMessage::Abort(vec![request.request_id]));
            }
        });
    }

    let client = MockEngineClient {
        routing: Arc::clone(&routing),
        inbound_tx,
        abort_tx,
        model_name: model_name.into(),
    };
    (
        client,
        MockEngine {
            routing,
            inbound_rx,
        },
    )
}
