//! Cloneable submission and control handles for an engine owner thread.
//!
//! Each accepted request receives a bounded event channel. Commands are queued
//! independently and wake the engine after enqueueing.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use tokio::sync::mpsc;
use uniserve_core::{Event, Request, RequestId};

/// Pollable command ingress whose lifetime is independent of Worker membership.
pub(crate) struct WakeSignal {
    reader: std::os::unix::net::UnixStream,
    waker: uniserve_core::CommandWaker,
}

impl WakeSignal {
    pub(crate) fn new() -> std::io::Result<Self> {
        use std::io::Write as _;

        let (reader, writer) = std::os::unix::net::UnixStream::pair()?;
        reader.set_nonblocking(true)?;
        writer.set_nonblocking(true)?;
        let waker = uniserve_core::CommandWaker::new(move || {
            loop {
                match (&writer).write(&[1]) {
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    // A full socket already has a pending notification. A closed
                    // reader means its control owner has stopped accepting work.
                    _ => break,
                }
            }
        });
        Ok(Self { reader, waker })
    }

    pub(crate) fn waker(&self) -> uniserve_core::CommandWaker {
        self.waker.clone()
    }

    pub(crate) fn descriptor(&self) -> i32 {
        use std::os::fd::AsRawFd as _;
        self.reader.as_raw_fd()
    }

    /// Consumes latched notifications before the owner considers parking again.
    pub(crate) fn drain(&mut self) -> std::io::Result<bool> {
        use std::io::Read as _;

        let mut received = false;
        let mut bytes = [0_u8; 256];
        loop {
            match self.reader.read(&mut bytes) {
                Ok(0) => return Ok(received),
                Ok(_) => received = true,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    return Ok(received);
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
    }
}

/// Maximum number of generation events buffered for one request consumer.
pub const EVENT_BUFFER_CAPACITY: usize = 64;

/// Failure to publish an event into a request's bounded channel.
#[derive(Debug, thiserror::Error)]
pub enum EventSendError {
    #[error("generation event channel is full")]
    /// Returns the event rejected by a full bounded channel.
    Full(Box<Event>),
    #[error("generation event channel is closed")]
    /// Returns the event rejected after the receiver closed.
    Closed(Box<Event>),
}

/// Cause recorded when an event receiver closes before terminal completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamCancelCause {
    #[default]
    /// The consumer dropped or explicitly closed the event stream.
    DroppedStream,
    /// Incremental decoding matched a configured stop string.
    StopStringMatched,
}

/// Request-submission failures visible to engine clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SubmitError {
    #[error("engine command channel is closed")]
    /// The scheduler command channel is closed to submissions.
    Closed,
    #[error("engine is unavailable after a worker failure")]
    /// A terminal worker failure made the engine unavailable.
    Dead,
}

/// Bounded engine-to-caller event sender.
#[derive(Clone)]
pub struct EventTx {
    inner: mpsc::Sender<Event>,
}

impl EventTx {
    /// Attempts to publish an event without waiting for channel capacity.
    pub fn send(&self, event: Event) -> Result<(), EventSendError> {
        self.inner.try_send(event).map_err(|error| match error {
            mpsc::error::TrySendError::Full(event) => EventSendError::Full(Box::new(event)),
            mpsc::error::TrySendError::Closed(event) => EventSendError::Closed(Box::new(event)),
        })
    }

    /// Returns the channel's currently available event slots.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns whether the receiving side has closed.
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
}

/// Bounded engine-to-caller event receiver. Releasing one channel slot wakes
/// the scheduler so an output-capacity-stalled lineage becomes runnable without
/// polling.
pub struct EventRx {
    inner: mpsc::Receiver<Event>,
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
    /// Wraps a Tokio receiver without engine cancellation or wake integration.
    pub fn from_receiver(inner: mpsc::Receiver<Event>) -> Self {
        Self {
            inner,
            waker: uniserve_core::CommandWaker::noop(),
            cancellation: None,
            text_tokens_received: 0,
            acknowledged_token_count: 0,
            on_finish: None,
        }
    }

    /// Registers a callback to run once on terminal completion or channel close.
    pub fn set_on_finish(&mut self, on_finish: impl FnOnce() + Send + 'static) {
        self.on_finish = Some(Box::new(on_finish));
    }

    /// Receives the next event and advances output acknowledgement state.
    pub async fn recv(&mut self) -> Option<Event> {
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

    /// Receives the next event.
    pub async fn next(&mut self) -> Option<Event> {
        self.recv().await
    }

    /// Attempts to receive an event without waiting.
    pub fn try_recv(&mut self) -> Result<Event, mpsc::error::TryRecvError> {
        let event = self.inner.try_recv();
        if let Ok(event) = event.as_ref() {
            self.observe(event);
            self.waker.wake();
        }
        event
    }

    /// Updates acknowledgement and completion state for a received event.
    fn observe(&mut self, event: &Event) {
        match event {
            Event::TextToken { .. } => {
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
            Event::Finished { .. } | Event::Rejected { .. } | Event::Error { .. } => {
                self.finish();
            }
            _ => {}
        }
    }

    /// Acknowledges every text token observed through this receiver.
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

    /// Cancels the request at its last acknowledged public token prefix.
    pub fn cancel(&mut self) {
        self.cancel_at_consumed_prefix(StreamCancelCause::DroppedStream);
    }

    /// Closes generation at the receiver's safe public-token boundary.
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

    /// Disarms cancellation and invokes the completion callback exactly once.
    fn finish(&mut self) {
        self.cancellation = None;
        if let Some(on_finish) = self.on_finish.take() {
            on_finish();
        }
    }
}

impl Drop for EventRx {
    /// Releases resources owned by this value.
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

/// Creates a bounded request event channel with cancellation tracking.
pub(crate) fn event_channel() -> (EventTx, EventRx) {
    event_channel_with_waker(uniserve_core::CommandWaker::noop(), None)
}

/// Creates an event channel connected to a wake descriptor.
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

/// Command sent from a frontend handler to the scheduler thread.
pub enum Command {
    /// Admits a request and binds its event channel.
    Submit {
        /// Request admitted by the scheduler.
        request: Request,
        /// Destination for public request events.
        event_tx: EventTx,
    },
    /// Cancels a request at its acknowledged public-token boundary.
    Cancel {
        /// Request to cancel.
        request_id: RequestId,
        /// Safe public-token prefix, when known.
        output_token_count: Option<usize>,
    },
    /// Frontend decoder matched a stop string at this exact token prefix.
    StopAt {
        /// Request whose decoder matched the stop string.
        request_id: RequestId,
        /// Exact public-token prefix at which generation stops.
        output_token_count: usize,
    },
    /// Frontend decoder accepted this exact public token prefix.
    Acknowledge {
        /// Request whose public output was consumed.
        request_id: RequestId,
        /// Number of public tokens consumed by the frontend.
        output_token_count: usize,
    },
    /// Aborts a request immediately at the server boundary.
    Abort(RequestId),
    /// Requests orderly engine shutdown.
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
    /// Constructs a handle with the no-op waker (the polling / sim path, which
    /// observes commands through its own timed wait).
    pub fn new(tx: crossbeam_channel::Sender<Command>) -> Self {
        Self::with_waker(tx, uniserve_core::CommandWaker::noop())
    }

    /// Constructs a handle that fires `waker` after every enqueue, used when the
    /// engine drives an event-driven executor that parks between steps.
    pub fn with_waker(
        tx: crossbeam_channel::Sender<Command>,
        waker: uniserve_core::CommandWaker,
    ) -> Self {
        Self { tx, waker }
    }

    /// Enqueues a command, then wakes the scheduler after a successful send.
    fn send(&self, cmd: Command) -> Result<(), crossbeam_channel::SendError<Command>> {
        let r = self.tx.send(cmd);
        // Wake only on a successful enqueue: if the channel is closed there is
        // no scheduler to wake, and the error is propagated to the caller.
        if r.is_ok() {
            self.waker.wake();
        }
        r
    }

    /// Submits a request and returns its bounded event stream.
    pub fn submit(&self, request: impl Into<Request>) -> Result<EventRx, SubmitError> {
        let request = request.into();
        let request_id = request.request_id();
        let acknowledge_on_receive = match &request {
            Request::Ar(request) | Request::Umm(request) => request.stop_strings.is_empty(),
            Request::Diffusion(_) => false,
        };
        let (event_tx, event_rx) = event_channel_with_waker(
            self.waker.clone(),
            Some(EventCancellation {
                tx: self.tx.clone(),
                request_id,
                acknowledge_on_receive,
            }),
        );
        self.send(Command::Submit { request, event_tx })
            .map_err(|_| SubmitError::Closed)?;
        Ok(event_rx)
    }

    /// Cancels a request without constraining its public output prefix.
    pub fn cancel(&self, id: RequestId) {
        let _ = self.send(Command::Cancel {
            request_id: id,
            output_token_count: None,
        });
    }
    /// Cancels a request after the specified public token count.
    pub fn cancel_at(&self, id: RequestId, output_token_count: usize) {
        let _ = self.send(Command::Cancel {
            request_id: id,
            output_token_count: Some(output_token_count),
        });
    }
    /// Finishes a request successfully at the specified public token count.
    pub fn stop_at(&self, id: RequestId, output_token_count: usize) {
        let _ = self.send(Command::StopAt {
            request_id: id,
            output_token_count,
        });
    }
    /// Acknowledges consumption through the specified public token count.
    pub fn acknowledge_at(&self, id: RequestId, output_token_count: usize) {
        let _ = self.send(Command::Acknowledge {
            request_id: id,
            output_token_count,
        });
    }
    /// Aborts a request immediately without applying client cancellation semantics.
    pub fn abort(&self, id: RequestId) {
        let _ = self.send(Command::Abort(id));
    }

    /// Requests orderly engine shutdown.
    pub fn shutdown(&self) {
        let _ = self.send(Command::Shutdown);
    }
}

#[cfg(test)]
mod tests {
    use uniserve_core::FinishReason;
    use uniserve_core::{
        ContextSegment, GenerationBehaviorDescriptor, GenerationConstraint,
        GenerationPolicyDescriptor, GenerationRequest, GenerationResourceBounds, ImageParams,
        RequestId, SamplingParams, UndVisibility,
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
                assert_eq!(request.request_id(), RequestId(11));
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
            .send(Event::TextToken {
                id: 7,
                logprob: None,
            })
            .unwrap();
        event_tx
            .send(Event::TextToken {
                id: 8,
                logprob: None,
            })
            .unwrap();

        assert!(matches!(
            events.try_recv(),
            Ok(Event::TextToken { id: 7, .. })
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

    /// Finishes reasons are distinct values, so a client-side cancel never
    /// compares equal to a server-side abort.
    #[test]
    fn finish_reason_variants_are_distinct() {
        assert_ne!(FinishReason::Cancelled, FinishReason::Aborted);
        assert_ne!(FinishReason::Eos, FinishReason::Stop);
        assert_eq!(FinishReason::MaxTokens, FinishReason::MaxTokens);
    }

    /// A `Event::Finished` carries the finish reason and terminal token
    /// counts as its payload (the type is not `PartialEq`, so match on it).
    #[test]
    fn gen_event_finished_carries_reason_and_counts() {
        let event = Event::Finished {
            reason: FinishReason::Stop,
            stop_reason: Some(uniserve_core::StopReason::String("</s>".to_string())),
            prompt_tokens: 4,
            completion_tokens: 9,
            images: 0,
        };

        match event {
            Event::Finished {
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
