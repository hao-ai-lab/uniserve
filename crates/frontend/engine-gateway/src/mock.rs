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

use crate::client::{EngineCoreOutputStream, EngineCoreStreamOutput, StreamCancelRequest};
use crate::error::{Error, Result};
use crate::protocol::{EngineCoreOutputs, EngineCoreRequest};

type OutputSender = mpsc::Sender<Result<EngineCoreStreamOutput>>;
type Routing = Arc<Mutex<HashMap<String, OutputSender>>>;

/// A message the client sends to the mock engine.
#[derive(Debug)]
pub enum MockClientMessage {
    /// A new generate request.
    Add(Box<EngineCoreRequest>),
    /// One or more request ids explicitly aborted by the runtime or an administrator.
    Abort(Vec<String>),
    /// One or more request ids cancelled by their owner.
    Cancel(Vec<String>),
}

/// The client-side mock backend held inside [`crate::EngineCoreClient`].
pub struct MockEngineClient {
    routing: Routing,
    inbound_tx: mpsc::UnboundedSender<MockClientMessage>,
    cancel_tx: mpsc::UnboundedSender<StreamCancelRequest>,
    model_name: String,
}

impl MockEngineClient {
    pub(crate) fn model_name(&self) -> &str {
        &self.model_name
    }

    pub(crate) fn call(&self, req: EngineCoreRequest) -> Result<EngineCoreOutputStream> {
        let request_id = req.request_id.clone();
        let (tx, rx) = mpsc::channel(EngineCoreOutputStream::BUFFER_CAPACITY);
        self.routing
            .lock()
            .map_err(|_| Error::ClientClosed {
                message: "mock engine routing mutex is poisoned".to_string(),
            })?
            .insert(request_id.clone(), tx);
        let _ = self.inbound_tx.send(MockClientMessage::Add(Box::new(req)));
        Ok(EngineCoreOutputStream::new(
            request_id,
            self.cancel_tx.clone(),
            rx,
        ))
    }

    pub(crate) fn abort(&self, ids: &[String]) {
        let _ = self.inbound_tx.send(MockClientMessage::Abort(ids.to_vec()));
    }

    pub(crate) fn cancel(&self, ids: &[String]) {
        let _ = self
            .inbound_tx
            .send(MockClientMessage::Cancel(ids.to_vec()));
    }
}

/// The test-side handle used to script the engine.
pub struct MockEngine {
    routing: Routing,
    inbound_rx: mpsc::UnboundedReceiver<MockClientMessage>,
}

impl MockEngine {
    /// Receive the next message from the client.
    pub async fn recv(&mut self) -> Option<MockClientMessage> {
        self.inbound_rx.recv().await
    }

    /// Receive the next `Add` request, skipping lifecycle controls.
    ///
    /// # Panics
    ///
    /// Panics if the client disconnects before sending a request.
    pub async fn recv_request(&mut self) -> EngineCoreRequest {
        loop {
            match self.inbound_rx.recv().await {
                Some(MockClientMessage::Add(req)) => return *req,
                Some(MockClientMessage::Abort(_) | MockClientMessage::Cancel(_)) => continue,
                None => panic!("mock engine: client disconnected before sending a request"),
            }
        }
    }

    /// Route scripted outputs to the matching per-request output streams.
    ///
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
                let _ = sender.try_send(Ok(EngineCoreStreamOutput {
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
    let (cancel_tx, mut cancel_rx) = mpsc::unbounded_channel::<StreamCancelRequest>();

    // A dropped output stream surfaces to the mock as cancellation, matching ZMQ.
    {
        let inbound_tx = inbound_tx.clone();
        tokio::spawn(async move {
            while let Some(request) = cancel_rx.recv().await {
                let _ = inbound_tx.send(MockClientMessage::Cancel(vec![request.request_id]));
            }
        });
    }

    let client = MockEngineClient {
        routing: Arc::clone(&routing),
        inbound_tx,
        cancel_tx,
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
