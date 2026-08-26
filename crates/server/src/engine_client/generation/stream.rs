use std::collections::BTreeMap;

use tokio::sync::mpsc;

use crate::engine_client::StreamCancelCause;

use uniserve_core::{GenerationEvent, GenerationRequest};

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

/// A typed text-and-image event stream for one canonical generation request.
pub struct GenerationEventStream {
    rx: mpsc::Receiver<GenerationEvent>,
    cancel: Option<Box<dyn FnOnce(StreamCancelCause, usize) + Send + 'static>>,
    acknowledge: Option<Box<dyn Fn(usize) + Send + 'static>>,
    output_token_count: usize,
    acknowledged_token_count: usize,
    acknowledge_on_receive: bool,
    finished: bool,
}

impl GenerationEventStream {
    pub fn new(rx: mpsc::Receiver<GenerationEvent>) -> Self {
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
        rx: mpsc::Receiver<GenerationEvent>,
        cancel: impl FnOnce(StreamCancelCause, usize) + Send + 'static,
        acknowledge: impl Fn(usize) + Send + 'static,
    ) -> Self {
        Self::with_control_policy(rx, cancel, acknowledge, false)
    }

    /// Attach exact-prefix controls and optionally acknowledge each consumed token.
    pub fn with_control_policy(
        rx: mpsc::Receiver<GenerationEvent>,
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
    pub async fn next(&mut self) -> Option<GenerationEvent> {
        let ev = self.rx.recv().await?;
        if matches!(ev, GenerationEvent::TextToken { .. }) {
            self.output_token_count = self.output_token_count.saturating_add(1);
            if self.acknowledge_on_receive {
                self.acknowledge_text_prefix();
            }
        }
        if matches!(
            ev,
            GenerationEvent::Finished { .. }
                | GenerationEvent::Rejected { .. }
                | GenerationEvent::Error { .. }
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

    use super::*;

    #[tokio::test]
    async fn canonical_stream_cancels_at_its_consumed_text_token_prefix() {
        let (tx, rx) = mpsc::channel(uniserve_engine::EVENT_BUFFER_CAPACITY);
        for id in 1..=3 {
            tx.send(GenerationEvent::TextToken {
                id,
                logprob: None,
                public_commit: None,
            })
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
            Some(GenerationEvent::TextToken { id: 1, .. })
        ));
        stream.acknowledge_text_prefix();
        assert!(matches!(
            stream.next().await,
            Some(GenerationEvent::TextToken { id: 2, .. })
        ));
        StreamCancelCause::StopStringMatched.drop_as(stream);
        assert_eq!(
            *cancellation.lock().unwrap(),
            Some((StreamCancelCause::StopStringMatched, 2))
        );
        assert_eq!(*acknowledgements.lock().unwrap(), vec![1]);
    }
}
