use std::collections::BTreeMap;

use futures::StreamExt as _;
use tokio::sync::mpsc;

use crate::StreamCancelCause;
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
    decoder_ack_required: bool,
) -> GenerationEventStream {
    enum AdapterControl {
        Cancel(StreamCancelCause, usize),
        Acknowledge(usize),
    }

    stream.delegate_acknowledgement();
    let (tx, rx) = generation_event_channel();
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                Some(control) = control_rx.recv() => {
                    match control {
                        AdapterControl::Cancel(cause, output_token_count) => {
                            stream.cancel_at(cause, output_token_count);
                            return;
                        }
                        AdapterControl::Acknowledge(output_token_count) => {
                            stream.acknowledge_at(output_token_count);
                        }
                    }
                }
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
    let acknowledge_tx = control_tx.clone();
    GenerationEventStream::with_control_policy(
        rx,
        move |cause, output_token_count| {
            let _ = control_tx.send(AdapterControl::Cancel(cause, output_token_count));
        },
        move |output_token_count| {
            let _ = acknowledge_tx.send(AdapterControl::Acknowledge(output_token_count));
        },
        !decoder_ack_required,
    )
}

/// A typed text-and-image event stream for one canonical generation request.
pub struct GenerationEventStream {
    rx: mpsc::Receiver<GenEvent>,
    cancel: Option<Box<dyn FnOnce(StreamCancelCause, usize) + Send + 'static>>,
    acknowledge: Option<Box<dyn Fn(usize) + Send + 'static>>,
    output_token_count: usize,
    acknowledged_token_count: usize,
    acknowledge_on_receive: bool,
    finished: bool,
}

impl GenerationEventStream {
    pub fn new(rx: mpsc::Receiver<GenEvent>) -> Self {
        Self {
            rx,
            cancel: None,
            acknowledge: None,
            output_token_count: 0,
            acknowledged_token_count: 0,
            acknowledge_on_receive: false,
            finished: false,
        }
    }

    pub fn with_control(
        rx: mpsc::Receiver<GenEvent>,
        cancel: impl FnOnce(StreamCancelCause, usize) + Send + 'static,
        acknowledge: impl Fn(usize) + Send + 'static,
    ) -> Self {
        Self::with_control_policy(rx, cancel, acknowledge, false)
    }

    /// Attach exact-prefix controls and optionally acknowledge each consumed token.
    pub fn with_control_policy(
        rx: mpsc::Receiver<GenEvent>,
        cancel: impl FnOnce(StreamCancelCause, usize) + Send + 'static,
        acknowledge: impl Fn(usize) + Send + 'static,
        acknowledge_on_receive: bool,
    ) -> Self {
        Self {
            rx,
            cancel: Some(Box::new(cancel)),
            acknowledge: Some(Box::new(acknowledge)),
            output_token_count: 0,
            acknowledged_token_count: 0,
            acknowledge_on_receive,
            finished: false,
        }
    }

    /// Accept every text token consumed so far as a semantically valid prefix.
    pub fn acknowledge_text_prefix(&mut self) {
        if self.output_token_count <= self.acknowledged_token_count {
            return;
        }
        if let Some(acknowledge) = self.acknowledge.as_ref() {
            acknowledge(self.output_token_count);
        }
        self.acknowledged_token_count = self.output_token_count;
    }

    /// Cancel at the exact text-token prefix already consumed by this stream.
    pub fn cancel_at_consumed_prefix(&mut self, cause: StreamCancelCause) {
        if let Some(cancel) = self.cancel.take() {
            cancel(cause, self.output_token_count);
        }
        self.acknowledge = None;
    }

    /// Await the next event, or `None` once the stream is exhausted.
    pub async fn next(&mut self) -> Option<GenEvent> {
        let ev = self.rx.recv().await?;
        if matches!(ev, GenEvent::TextToken { .. }) {
            self.output_token_count = self.output_token_count.saturating_add(1);
            if self.acknowledge_on_receive {
                self.acknowledge_text_prefix();
            }
        }
        if matches!(
            ev,
            GenEvent::Finished { .. } | GenEvent::Rejected { .. } | GenEvent::Error { .. }
        ) {
            self.finished = true;
            self.cancel = None;
            self.acknowledge = None;
        }
        Some(ev)
    }
}

impl Drop for GenerationEventStream {
    fn drop(&mut self) {
        if !self.finished
            && let Some(cancel) = self.cancel.take()
        {
            let cause = StreamCancelCause::current();
            let output_token_count = match cause {
                StreamCancelCause::DroppedStream => self.acknowledged_token_count,
                StreamCancelCause::StopStringMatched => self.output_token_count,
            };
            cancel(cause, output_token_count);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

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

    #[tokio::test]
    async fn canonical_stream_cancels_at_its_consumed_text_token_prefix() {
        let (tx, rx) = generation_event_channel();
        for id in 1..=3 {
            tx.send(GenEvent::TextToken { id, logprob: None })
                .await
                .unwrap();
        }
        let cancellation = Arc::new(Mutex::new(None));
        let recorded = Arc::clone(&cancellation);
        let acknowledgements = Arc::new(Mutex::new(Vec::new()));
        let acknowledged = Arc::clone(&acknowledgements);
        let mut stream = GenerationEventStream::with_control(
            rx,
            move |cause, token_count| {
                *recorded.lock().unwrap() = Some((cause, token_count));
            },
            move |token_count| acknowledged.lock().unwrap().push(token_count),
        );
        assert!(matches!(
            stream.next().await,
            Some(GenEvent::TextToken { id: 1, .. })
        ));
        stream.acknowledge_text_prefix();
        assert!(matches!(
            stream.next().await,
            Some(GenEvent::TextToken { id: 2, .. })
        ));
        StreamCancelCause::StopStringMatched.drop_as(stream);
        assert_eq!(
            *cancellation.lock().unwrap(),
            Some((StreamCancelCause::StopStringMatched, 2))
        );
        assert_eq!(*acknowledgements.lock().unwrap(), vec![1]);
    }
}
