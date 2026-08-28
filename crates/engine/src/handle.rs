//! Engine interface for streamed generation and terminal media requests.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use tokio::sync::mpsc;
use uniserve_core::{GenerationEvent, GenerationRequest, MediaEvent, MediaRequest, RequestId};

/// Maximum number of canonical generation events buffered between one
/// scheduler request and its immediate consumer.
pub const EVENT_BUFFER_CAPACITY: usize = 64;

#[derive(Debug, thiserror::Error)]
pub enum EventSendError {
    #[error("generation event channel is full")]
    Full(Box<GenerationEvent>),
    #[error("generation event channel is closed")]
    Closed(Box<GenerationEvent>),
}

#[derive(Debug, thiserror::Error)]
pub enum MediaEventSendError {
    #[error("media event channel is full")]
    Full(Box<MediaEvent>),
    #[error("media event channel is closed")]
    Closed(Box<MediaEvent>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamCancelCause {
    #[default]
    DroppedStream,
    StopStringMatched,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SubmitError {
    #[error("engine command channel is closed")]
    Closed,
    #[error("engine is unavailable after a worker failure")]
    Dead,
}

/// Bounded engine-to-caller event sender.
#[derive(Clone)]
pub struct EventTx {
    inner: mpsc::Sender<GenerationEvent>,
}

impl EventTx {
    pub fn send(&self, event: GenerationEvent) -> Result<(), EventSendError> {
        self.inner.try_send(event).map_err(|error| match error {
            mpsc::error::TrySendError::Full(event) => EventSendError::Full(Box::new(event)),
            mpsc::error::TrySendError::Closed(event) => EventSendError::Closed(Box::new(event)),
        })
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
}

/// Bounded engine-to-caller event receiver. Releasing one channel slot wakes
/// the scheduler so an output-capacity-stalled lineage becomes runnable without
/// polling.
pub struct EventRx {
    inner: mpsc::Receiver<GenerationEvent>,
    waker: uniserve_core::CommandWaker,
    cancellation: Option<EventCancellation>,
    text_tokens_received: usize,
    acknowledged_token_count: usize,
    on_finish: Option<Box<dyn FnOnce() + Send + 'static>>,
}

struct EventCancellation {
    tx: crossbeam_channel::Sender<Command>,
    request_id: RequestId,
    acknowledge_on_receive: bool,
}

impl EventRx {
    pub fn from_receiver(inner: mpsc::Receiver<GenerationEvent>) -> Self {
        Self {
            inner,
            waker: uniserve_core::CommandWaker::noop(),
            cancellation: None,
            text_tokens_received: 0,
            acknowledged_token_count: 0,
            on_finish: None,
        }
    }

    pub fn set_on_finish(&mut self, on_finish: impl FnOnce() + Send + 'static) {
        self.on_finish = Some(Box::new(on_finish));
    }

    pub async fn recv(&mut self) -> Option<GenerationEvent> {
        let event = self.inner.recv().await;
        match event.as_ref() {
            Some(event) => {
                self.observe(event);
                self.waker.wake();
            }
            None => self.finish(),
        }
        event
    }

    pub async fn next(&mut self) -> Option<GenerationEvent> {
        self.recv().await
    }

    pub fn try_recv(&mut self) -> Result<GenerationEvent, mpsc::error::TryRecvError> {
        let event = self.inner.try_recv();
        if let Ok(event) = event.as_ref() {
            self.observe(event);
            self.waker.wake();
        }
        event
    }

    fn observe(&mut self, event: &GenerationEvent) {
        match event {
            GenerationEvent::TextToken { .. } => {
                self.text_tokens_received = self.text_tokens_received.saturating_add(1);
                if self
                    .cancellation
                    .as_ref()
                    .filter(|cancellation| cancellation.acknowledge_on_receive)
                    .is_some()
                {
                    self.acknowledge_consumed_prefix();
                }
            }
            GenerationEvent::Finished { .. }
            | GenerationEvent::Rejected { .. }
            | GenerationEvent::Error { .. }
            | GenerationEvent::MediaCompleted { .. }
            | GenerationEvent::MediaFailed { .. }
            | GenerationEvent::MediaAborted => {
                self.finish();
            }
            _ => {}
        }
    }

    pub fn acknowledge_consumed_prefix(&mut self) {
        if self.text_tokens_received <= self.acknowledged_token_count {
            return;
        }
        if let Some(cancellation) = self.cancellation.as_ref() {
            let _ = cancellation.tx.send(Command::Acknowledge {
                request_id: cancellation.request_id,
                output_token_count: self.text_tokens_received,
            });
            self.acknowledged_token_count = self.text_tokens_received;
        }
    }

    pub fn cancel_at_consumed_prefix(&mut self, cause: StreamCancelCause) {
        let Some(cancellation) = self.cancellation.take() else {
            return;
        };
        let command = match cause {
            StreamCancelCause::DroppedStream => Command::Cancel {
                request_id: cancellation.request_id,
                output_token_count: Some(self.acknowledged_token_count),
            },
            StreamCancelCause::StopStringMatched => Command::StopAt {
                request_id: cancellation.request_id,
                output_token_count: self.text_tokens_received,
            },
        };
        let _ = cancellation.tx.send(command);
    }

    fn finish(&mut self) {
        self.cancellation = None;
        if let Some(on_finish) = self.on_finish.take() {
            on_finish();
        }
    }
}

impl Drop for EventRx {
    fn drop(&mut self) {
        if let Some(cancellation) = self.cancellation.take() {
            let _ = cancellation.tx.send(Command::Cancel {
                request_id: cancellation.request_id,
                output_token_count: Some(self.acknowledged_token_count),
            });
        }
        if let Some(on_finish) = self.on_finish.take() {
            on_finish();
        }
        self.waker.wake();
    }
}

pub(crate) fn event_channel() -> (EventTx, EventRx) {
    event_channel_with_waker(uniserve_core::CommandWaker::noop(), None)
}

fn event_channel_with_waker(
    waker: uniserve_core::CommandWaker,
    cancellation: Option<EventCancellation>,
) -> (EventTx, EventRx) {
    let (tx, rx) = mpsc::channel(EVENT_BUFFER_CAPACITY);
    (
        EventTx { inner: tx },
        EventRx {
            inner: rx,
            waker,
            cancellation,
            text_tokens_received: 0,
            acknowledged_token_count: 0,
            on_finish: None,
        },
    )
}

#[derive(Clone)]
pub struct MediaEventTx {
    inner: mpsc::Sender<MediaEvent>,
}

impl MediaEventTx {
    pub fn send(&self, event: MediaEvent) -> Result<(), MediaEventSendError> {
        self.inner.try_send(event).map_err(|error| match error {
            mpsc::error::TrySendError::Full(event) => MediaEventSendError::Full(Box::new(event)),
            mpsc::error::TrySendError::Closed(event) => {
                MediaEventSendError::Closed(Box::new(event))
            }
        })
    }

    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
}

pub struct MediaEventRx {
    inner: mpsc::Receiver<MediaEvent>,
    waker: uniserve_core::CommandWaker,
    cancellation: Option<EventCancellation>,
    on_finish: Option<Box<dyn FnOnce() + Send + 'static>>,
}

impl MediaEventRx {
    pub fn set_on_finish(&mut self, on_finish: impl FnOnce() + Send + 'static) {
        self.on_finish = Some(Box::new(on_finish));
    }

    pub async fn recv(&mut self) -> Option<MediaEvent> {
        let event = self.inner.recv().await;
        if event.is_some() {
            self.finish();
            self.waker.wake();
        }
        event
    }

    pub async fn next(&mut self) -> Option<MediaEvent> {
        self.recv().await
    }

    pub fn cancel(&mut self) {
        if let Some(cancellation) = self.cancellation.take() {
            let _ = cancellation.tx.send(Command::Cancel {
                request_id: cancellation.request_id,
                output_token_count: None,
            });
        }
    }

    fn finish(&mut self) {
        self.cancellation = None;
        if let Some(on_finish) = self.on_finish.take() {
            on_finish();
        }
    }
}

impl Drop for MediaEventRx {
    fn drop(&mut self) {
        if let Some(cancellation) = self.cancellation.take() {
            let _ = cancellation.tx.send(Command::Cancel {
                request_id: cancellation.request_id,
                output_token_count: None,
            });
        }
        if let Some(on_finish) = self.on_finish.take() {
            on_finish();
        }
        self.waker.wake();
    }
}

/// Command sent from a frontend handler to the scheduler thread.
pub enum Command {
    Submit {
        request: Box<GenerationRequest>,
        event_tx: EventTx,
    },
    SubmitMedia {
        request: MediaRequest,
        event_tx: MediaEventTx,
    },
    /// Client-side cancel → `FinishReason::Cancelled`.
    Cancel {
        request_id: RequestId,
        output_token_count: Option<usize>,
    },
    /// Frontend decoder matched a stop string at this exact token prefix.
    StopAt {
        request_id: RequestId,
        output_token_count: usize,
    },
    /// Frontend decoder accepted this exact public token prefix.
    Acknowledge {
        request_id: RequestId,
        output_token_count: usize,
    },
    /// Server-side abort → `FinishReason::Aborted`.
    Abort(RequestId),
    Shutdown,
}

/// Cloneable front door over the scheduler. `submit` enqueues to the scheduler;
/// dropping the last handle tears the engine down.
///
/// Every send fires the [`CommandWaker`] right after enqueuing, so when the
/// scheduler is parked on an event-driven executor it wakes immediately to
/// observe the command instead of waiting out the park's safety-net timeout.
/// With the no-op waker (the polling path / sim), this is free.
#[derive(Clone)]
pub struct EngineHandle {
    tx: crossbeam_channel::Sender<Command>,
    waker: uniserve_core::CommandWaker,
}

impl EngineHandle {
    /// Construct a handle with the no-op waker (the polling / sim path, which
    /// observes commands through its own timed wait).
    pub fn new(tx: crossbeam_channel::Sender<Command>) -> Self {
        Self::with_waker(tx, uniserve_core::CommandWaker::noop())
    }

    /// Construct a handle that fires `waker` after every enqueue, used when the
    /// engine drives an event-driven executor that parks between steps.
    pub fn with_waker(
        tx: crossbeam_channel::Sender<Command>,
        waker: uniserve_core::CommandWaker,
    ) -> Self {
        Self { tx, waker }
    }

    /// Enqueue a command and wake any parked scheduler. Centralizes the
    /// send-then-wake order so no caller can forget the wake.
    fn send(&self, cmd: Command) -> Result<(), crossbeam_channel::SendError<Command>> {
        let r = self.tx.send(cmd);
        // Wake only on a successful enqueue: if the channel is closed there is
        // no scheduler to wake, and the error is propagated to the caller.
        if r.is_ok() {
            self.waker.wake();
        }
        r
    }

    pub fn submit(&self, request: GenerationRequest) -> Result<EventRx, SubmitError> {
        let request_id = request.request_id;
        let acknowledge_on_receive = request.stop_strings.is_empty();
        let (event_tx, event_rx) = event_channel_with_waker(
            self.waker.clone(),
            Some(EventCancellation {
                tx: self.tx.clone(),
                request_id,
                acknowledge_on_receive,
            }),
        );
        self.send(Command::Submit {
            request: Box::new(request),
            event_tx,
        })
        .map_err(|_| SubmitError::Closed)?;
        Ok(event_rx)
    }

    pub fn submit_media(&self, request: MediaRequest) -> Result<MediaEventRx, SubmitError> {
        let request_id = request.request_id;
        let (tx, rx) = mpsc::channel(1);
        let event_tx = MediaEventTx { inner: tx };
        let event_rx = MediaEventRx {
            inner: rx,
            waker: self.waker.clone(),
            cancellation: Some(EventCancellation {
                tx: self.tx.clone(),
                request_id,
                acknowledge_on_receive: false,
            }),
            on_finish: None,
        };
        self.send(Command::SubmitMedia { request, event_tx })
            .map_err(|_| SubmitError::Closed)?;
        Ok(event_rx)
    }
    pub fn cancel(&self, id: RequestId) {
        let _ = self.send(Command::Cancel {
            request_id: id,
            output_token_count: None,
        });
    }
    pub fn cancel_at(&self, id: RequestId, output_token_count: usize) {
        let _ = self.send(Command::Cancel {
            request_id: id,
            output_token_count: Some(output_token_count),
        });
    }
    pub fn stop_at(&self, id: RequestId, output_token_count: usize) {
        let _ = self.send(Command::StopAt {
            request_id: id,
            output_token_count,
        });
    }
    pub fn acknowledge_at(&self, id: RequestId, output_token_count: usize) {
        let _ = self.send(Command::Acknowledge {
            request_id: id,
            output_token_count,
        });
    }
    /// Server-side abort, distinct from a client cancel.
    pub fn abort(&self, id: RequestId) {
        let _ = self.send(Command::Abort(id));
    }
    pub fn shutdown(&self) {
        let _ = self.send(Command::Shutdown);
    }
}

#[cfg(test)]
mod tests {
    use uniserve_core::FinishReason;
    use uniserve_core::{
        ContextSegment, GenerationBehaviorDescriptor, GenerationConstraint,
        GenerationPolicyDescriptor, GenerationResourceBounds, ImageParams, RequestId,
        SamplingParams, UndVisibility,
    };

    use super::*;

    fn test_request(request_id: u64) -> GenerationRequest {
        let constraint = GenerationConstraint::UndOnly;
        let policy = GenerationPolicyDescriptor::default();
        GenerationRequest {
            request_id: RequestId(request_id),
            context: vec![ContextSegment::UndTokens {
                token_ids: vec![1, 2, 3],
                visibility: UndVisibility::Internal,
            }],
            negative_context: Vec::new(),
            constraint,
            behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_und_tokens: 32,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            policy,
            resources: GenerationResourceBounds {
                context_tokens: 3,
                max_kv_tokens: 35,
                ..GenerationResourceBounds::default()
            },
        }
    }

    /// The canonical request is pure value data with explicit context, policy,
    /// behavior, and resource declarations.
    #[test]
    fn generation_request_is_canonical_pure_data() {
        let request = test_request(7);

        assert_eq!(request.request_id, RequestId(7));
        assert_eq!(request.prompt_token_count(), 3);
        assert_eq!(request.max_und_tokens, 32);
        assert_eq!(request.constraint, GenerationConstraint::UndOnly);
        assert!(request.negative_context.is_empty());
        assert!(request.stop_strings.is_empty());
        assert!(request.stop_token_ids.is_empty());
        assert_eq!(request.priority, 0);
        assert_eq!(request.context_image_count(), 0);
        assert!(request.cache.read);
        assert!(request.cache.write);
        assert!(request.validate().is_ok());
    }

    /// `submit` enqueues a `Command::Submit` carrying the request, and the
    /// scheduler side receives exactly that request id.
    #[test]
    fn submit_enqueues_submit_command_with_request() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);

        let _events = handle.submit(test_request(11)).unwrap();

        match rx.recv().unwrap() {
            Command::Submit { request, .. } => {
                assert_eq!(request.request_id, RequestId(11));
            }
            _ => panic!("expected Submit command"),
        }
    }

    #[test]
    fn dropping_event_receiver_cancels_at_the_consumed_text_prefix() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let mut events = handle.submit(test_request(12)).unwrap();
        let event_tx = match rx.recv().unwrap() {
            Command::Submit { event_tx, .. } => event_tx,
            _ => panic!("expected Submit command"),
        };
        event_tx
            .send(GenerationEvent::TextToken {
                id: 7,
                logprob: None,
            })
            .unwrap();
        event_tx
            .send(GenerationEvent::TextToken {
                id: 8,
                logprob: None,
            })
            .unwrap();

        assert!(matches!(
            events.try_recv(),
            Ok(GenerationEvent::TextToken { id: 7, .. })
        ));
        drop(events);

        match rx.recv().unwrap() {
            Command::Acknowledge {
                request_id,
                output_token_count,
            } => {
                assert_eq!(request_id, RequestId(12));
                assert_eq!(output_token_count, 1);
            }
            _ => panic!("expected consumed-prefix Acknowledge command"),
        }
        match rx.recv().unwrap() {
            Command::Cancel {
                request_id,
                output_token_count,
            } => {
                assert_eq!(request_id, RequestId(12));
                assert_eq!(output_token_count, Some(1));
            }
            _ => panic!("expected exact-prefix Cancel command"),
        }
    }

    /// Cancellation, exact-prefix cancellation, semantic acknowledgement, and
    /// server abort preserve their distinct command payloads.
    #[test]
    fn request_control_commands_preserve_their_semantics() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);

        handle.cancel(RequestId(1));
        handle.cancel_at(RequestId(2), 7);
        handle.stop_at(RequestId(4), 8);
        handle.acknowledge_at(RequestId(3), 9);
        handle.abort(RequestId(2));

        match rx.recv().unwrap() {
            Command::Cancel {
                request_id,
                output_token_count,
            } => {
                assert_eq!(request_id, RequestId(1));
                assert_eq!(output_token_count, None);
            }
            _ => panic!("expected Cancel command"),
        }
        match rx.recv().unwrap() {
            Command::Cancel {
                request_id,
                output_token_count,
            } => {
                assert_eq!(request_id, RequestId(2));
                assert_eq!(output_token_count, Some(7));
            }
            _ => panic!("expected exact-prefix Cancel command"),
        }
        match rx.recv().unwrap() {
            Command::StopAt {
                request_id,
                output_token_count,
            } => {
                assert_eq!(request_id, RequestId(4));
                assert_eq!(output_token_count, 8);
            }
            _ => panic!("expected exact-prefix Stop command"),
        }
        match rx.recv().unwrap() {
            Command::Acknowledge {
                request_id,
                output_token_count,
            } => {
                assert_eq!(request_id, RequestId(3));
                assert_eq!(output_token_count, 9);
            }
            _ => panic!("expected Acknowledge command"),
        }
        match rx.recv().unwrap() {
            Command::Abort(id) => assert_eq!(id, RequestId(2)),
            _ => panic!("expected Abort command"),
        }
    }

    /// After the scheduler side of the channel is dropped, `submit` reports an
    /// error rather than panicking or silently dropping the request.
    #[test]
    fn submit_after_receiver_dropped_returns_error() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        drop(rx);

        let result = handle.submit(test_request(99));
        assert!(result.is_err());
    }

    /// The waker fires exactly once per successful enqueue, and not at all when
    /// the channel is closed (no scheduler to wake).
    #[test]
    fn waker_fires_on_successful_enqueue_only() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let wakes = Arc::new(AtomicUsize::new(0));
        let wakes_clone = Arc::clone(&wakes);
        let waker = uniserve_core::CommandWaker::new(move || {
            wakes_clone.fetch_add(1, Ordering::SeqCst);
        });

        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::with_waker(tx, waker);

        handle.cancel(RequestId(1));
        handle.abort(RequestId(2));
        assert_eq!(wakes.load(Ordering::SeqCst), 2);

        // Closing the channel means there is no scheduler to wake.
        drop(rx);
        handle.cancel(RequestId(3));
        assert_eq!(wakes.load(Ordering::SeqCst), 2);
    }

    /// Cloned handles share the same underlying channel: a command sent through
    /// the clone is observed by the original's receiver.
    #[test]
    fn cloned_handle_shares_channel() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let clone = handle.clone();

        clone.shutdown();

        assert!(matches!(rx.recv().unwrap(), Command::Shutdown));
    }

    /// Finish reasons are distinct values, so a client-side cancel never
    /// compares equal to a server-side abort.
    #[test]
    fn finish_reason_variants_are_distinct() {
        assert_ne!(FinishReason::Cancelled, FinishReason::Aborted);
        assert_ne!(FinishReason::Eos, FinishReason::Stop);
        assert_eq!(FinishReason::MaxTokens, FinishReason::MaxTokens);
    }

    /// A `GenerationEvent::Finished` carries the finish reason and terminal token
    /// counts as its payload (the type is not `PartialEq`, so match on it).
    #[test]
    fn gen_event_finished_carries_reason_and_counts() {
        let event = GenerationEvent::Finished {
            reason: FinishReason::Stop,
            stop_reason: Some(uniserve_core::StopReason::String("</s>".to_string())),
            prompt_tokens: 4,
            completion_tokens: 9,
            images: 0,
        };

        match event {
            GenerationEvent::Finished {
                reason,
                stop_reason,
                prompt_tokens,
                completion_tokens,
                images,
            } => {
                assert_eq!(reason, FinishReason::Stop);
                assert_eq!(
                    stop_reason,
                    Some(uniserve_core::StopReason::String("</s>".to_string()))
                );
                assert_eq!(prompt_tokens, 4);
                assert_eq!(completion_tokens, 9);
                assert_eq!(images, 0);
            }
            other => panic!("expected Finished, got {other:?}"),
        }
    }
}
