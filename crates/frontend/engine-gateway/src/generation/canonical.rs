use std::collections::BTreeMap;

use futures::StreamExt as _;
use tokio::sync::mpsc;

use crate::client::EngineCoreOutputStream;
use crate::protocol::EngineCoreRequest;

pub use uniserve_core::{
    GenerationConstraint, ImageParams, SamplingParams as EngineSamplingParams,
};
use uniserve_core::{GenerationRequest, now_unix_secs};
pub use uniserve_engine_api::{
    FinishReason as GenerationFinishReason, GenEvent,
    PositionLogprobs as GenerationPositionLogprobs, TokenLogprob as GenerationTokenLogprob,
};

/// Transport metadata wrapped around one pure canonical generation request.
#[derive(Debug, Clone)]
pub struct GenerationSubmission {
    pub external_request_id: String,
    pub request: GenerationRequest,
    pub arrival_time: Option<f64>,
    pub data_parallel_rank: Option<u32>,
    pub trace_headers: Option<BTreeMap<String, String>>,
}

impl GenerationSubmission {
    pub fn new(external_request_id: impl Into<String>, request: GenerationRequest) -> Self {
        Self {
            external_request_id: external_request_id.into(),
            request,
            arrival_time: None,
            data_parallel_rank: None,
            trace_headers: None,
        }
    }
}

/// Reverse adaptation of one wire output into the canonical [`GenEvent`] stream.
use uniserve_engine_wire::translate::wire_output_to_gen_events;

/// Maximum number of canonical generation events buffered per request.
pub const GENERATION_EVENT_BUFFER_CAPACITY: usize = 64;

fn generation_event_channel() -> (mpsc::Sender<GenEvent>, mpsc::Receiver<GenEvent>) {
    mpsc::channel(GENERATION_EVENT_BUFFER_CAPACITY)
}

pub(crate) fn generation_request_to_wire(submission: GenerationSubmission) -> EngineCoreRequest {
    let mut request = EngineCoreRequest::new(submission.external_request_id, submission.request);
    request.arrival_time = submission.arrival_time.unwrap_or_else(now_unix_secs);
    request.data_parallel_rank = submission.data_parallel_rank;
    request.trace_headers = submission.trace_headers;
    request
}

pub(crate) fn generation_event_stream_from_wire(
    mut stream: EngineCoreOutputStream,
) -> GenerationEventStream {
    let (tx, rx) = generation_event_channel();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = tx.closed() => return,
                item = stream.next() => match item {
                    Some(Ok(out)) => {
                        for ev in wire_output_to_gen_events(&out.output) {
                            if tx.send(ev).await.is_err() {
                                return;
                            }
                        }
                    }
                    Some(Err(error)) => {
                        let _ = tx.send(GenEvent::Error { message: error.to_string() }).await;
                        return;
                    }
                    None => return,
                },
            }
        }
    });
    GenerationEventStream::new(rx)
}

/// A typed text-and-image event stream for one canonical generation request.
pub struct GenerationEventStream {
    rx: mpsc::Receiver<GenEvent>,
    cancel: Option<Box<dyn FnOnce() + Send + 'static>>,
    finished: bool,
}

impl GenerationEventStream {
    pub fn new(rx: mpsc::Receiver<GenEvent>) -> Self {
        Self {
            rx,
            cancel: None,
            finished: false,
        }
    }

    pub fn with_cancel(
        rx: mpsc::Receiver<GenEvent>,
        cancel: impl FnOnce() + Send + 'static,
    ) -> Self {
        Self {
            rx,
            cancel: Some(Box::new(cancel)),
            finished: false,
        }
    }

    /// Await the next event, or `None` once the stream is exhausted.
    pub async fn next(&mut self) -> Option<GenEvent> {
        let ev = self.rx.recv().await?;
        if matches!(
            ev,
            GenEvent::Finished { .. } | GenEvent::Rejected { .. } | GenEvent::Error { .. }
        ) {
            self.finished = true;
            self.cancel = None;
        }
        Some(ev)
    }
}

impl Drop for GenerationEventStream {
    fn drop(&mut self) {
        if !self.finished
            && let Some(cancel) = self.cancel.take()
        {
            cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::mpsc::error::TrySendError;

    use super::*;

    #[test]
    fn canonical_generation_queue_applies_backpressure_at_its_bound() {
        let (tx, _rx) = generation_event_channel();
        for index in 0..GENERATION_EVENT_BUFFER_CAPACITY {
            tx.try_send(GenEvent::Error {
                message: index.to_string(),
            })
            .expect("queue has declared capacity");
        }

        assert!(matches!(
            tx.try_send(GenEvent::Error {
                message: "overflow".to_string(),
            }),
            Err(TrySendError::Full(_))
        ));
    }
}
