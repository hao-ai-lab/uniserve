//! Cloneable submission and control handles for an engine owner thread.
//!
//! Frontends control the scheduler thread with [`Command`]s that an
//! [`EngineHandle`] sends on a crossbeam channel; each successful send fires
//! the handle's `CommandWaker`, which wakes a parked scheduler unless it is
//! the no-op waker.
//! Each accepted request receives a bounded event channel ([`EventTx`] and
//! [`EventRx`], [`EVENT_BUFFER_CAPACITY`] events). A scheduler that stops
//! while a request's channel is full hands the events it could not send to
//! the receiver, which yields them after the channel's buffered events.
//!
//! [`EventRx`] also drives output acknowledgement. It counts received
//! `TextToken` events and reports consumed prefixes as
//! [`Command::Acknowledge`]. A token request without stop strings is
//! acknowledged on receipt; for a request with stop strings the frontend
//! decoder calls [`EventRx::acknowledge_consumed_prefix`] after checking that
//! a token completes no stop string, or [`EventRx::cancel_at_consumed_prefix`]
//! on a match. Dropping an [`EventRx`] before a terminal event cancels the
//! request at its acknowledged prefix.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};

use tokio::sync::mpsc;
use uniserve_core::{EngineCoreOutput, Request, RequestId};

/// Events a closing sender hands to its receiver, shared by the two ends of
/// one request's event channel.
type Handoff = Arc<Mutex<VecDeque<EngineCoreOutput>>>;

/// Pollable command ingress whose lifetime is independent of Worker membership.
///
/// A nonblocking Unix socket pair: the `CommandWaker` from [`Self::waker`]
/// owns the write end and writes a byte per wake (a full socket already holds
/// a pending notification), and the owner polls [`Self::descriptor`] for
/// readability and consumes the bytes with [`Self::drain`]. `WorkerExecutor`
/// owns one beside its worker groups and polls it with their progress
/// descriptors.
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

    /// Raw read-end descriptor for `poll`; valid while this value lives.
    pub(crate) fn descriptor(&self) -> i32 {
        use std::os::fd::AsRawFd as _;
        self.reader.as_raw_fd()
    }

    /// Consumes latched notifications before the owner considers parking again.
    ///
    /// Returns whether at least one notification was pending. Reads until the
    /// socket is empty or its write end has closed, so any number of wakes
    /// coalesce into one `true`. Read errors other than `WouldBlock` and
    /// `Interrupted` propagate.
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
///
/// The scheduler sizes its per-request output journal
/// (`OUTPUT_JOURNAL_CAPACITY`) from this value.
pub const EVENT_BUFFER_CAPACITY: usize = 64;

/// Failure to publish an event into a request's bounded channel.
#[derive(Debug, thiserror::Error)]
pub enum EventSendError {
    #[error("generation event channel is full")]
    /// Carries the event rejected by a full bounded channel.
    Full(Box<EngineCoreOutput>),
    #[error("generation event channel is closed")]
    /// Carries the event rejected after the receiver closed.
    Closed(Box<EngineCoreOutput>),
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
    inner: mpsc::Sender<EngineCoreOutput>,
    handoff: Handoff,
}

impl EventTx {
    /// Closes this sender and hands `events`, which did not fit into the
    /// channel, to the receiver.
    ///
    /// The receiver yields them in order once it has drained the channel and
    /// every sender has closed, so only a request's last publisher uses this.
    /// Nothing is kept when the receiver has already closed.
    pub(crate) fn close_with(self, events: VecDeque<EngineCoreOutput>) {
        if events.is_empty() || self.inner.is_closed() {
            return;
        }
        // The events are in place before this sender drops, so a receiver
        // that observes the closed channel also observes them.
        self.handoff
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend(events);
    }

    /// Attempts to publish an event without waiting for channel capacity.
    ///
    /// Never blocks. The scheduler keeps an event rejected as `Full` in the
    /// request's output journal (`EventJournal`) and resends it, in order, on
    /// a later flush that finds the channel has room.
    pub fn send(&self, event: EngineCoreOutput) -> Result<(), EventSendError> {
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
/// the scheduler so an output-capacity-stalled request becomes runnable without
/// polling.
///
/// After the channel closes and drains, the receiver yields the events its
/// sender handed over on closing (`EventTx::close_with`), and only then
/// reports the stream closed.
pub struct EventRx {
    inner: mpsc::Receiver<EngineCoreOutput>,
    handoff: Handoff,
    waker: uniserve_core::CommandWaker,
    cancellation: Option<EventCancellation>,
    text_tokens_received: usize,
    acknowledged_token_count: usize,
    on_finish: Option<Box<dyn FnOnce() + Send + 'static>>,
}

/// Control state that lets an [`EventRx`] act on its request.
///
/// The `handle` clone keeps the command channel open while the receiver
/// holds it; it is dropped on the terminal event, channel close, or first
/// cancellation.
struct EventCancellation {
    handle: EngineHandle,
    request_id: RequestId,
    /// Acknowledge each `TextToken` as soon as it is received. `submit` sets
    /// this only for token requests without stop strings; for token requests
    /// with stop strings the frontend decoder acknowledges explicitly.
    acknowledge_on_receive: bool,
}

impl EventRx {
    /// Wraps a Tokio receiver without engine cancellation or wake integration.
    pub fn from_receiver(inner: mpsc::Receiver<EngineCoreOutput>) -> Self {
        Self {
            inner,
            handoff: Handoff::default(),
            waker: uniserve_core::CommandWaker::noop(),
            cancellation: None,
            text_tokens_received: 0,
            acknowledged_token_count: 0,
            on_finish: None,
        }
    }

    /// Registers a callback to run once on terminal completion or channel close.
    ///
    /// Dropping the receiver also runs a callback that has not yet run.
    pub fn set_on_finish(&mut self, on_finish: impl FnOnce() + Send + 'static) {
        self.on_finish = Some(Box::new(on_finish));
    }

    /// Receives the next event and advances output acknowledgement state.
    ///
    /// Returns `None` once the channel is closed and empty and every handed
    /// over event has been yielded, which also runs the completion callback.
    pub async fn recv(&mut self) -> Option<EngineCoreOutput> {
        let event = match self.inner.recv().await {
            Some(event) => Some(event),
            None => self.take_handoff(),
        };
        match event.as_ref() {
            Some(event) => {
                self.observe(event);
                // The freed channel slot may unblock a request stalled on
                // output capacity.
                self.waker.wake();
            }
            None => self.finish(),
        }
        event
    }

    /// Receives the next event.
    pub async fn next(&mut self) -> Option<EngineCoreOutput> {
        self.recv().await
    }

    /// Attempts to receive an event without waiting.
    ///
    /// Reports `Disconnected` only after every handed over event has been
    /// yielded.
    pub fn try_recv(&mut self) -> Result<EngineCoreOutput, mpsc::error::TryRecvError> {
        let event = match self.inner.try_recv() {
            Err(mpsc::error::TryRecvError::Disconnected) => self
                .take_handoff()
                .ok_or(mpsc::error::TryRecvError::Disconnected),
            event => event,
        };
        if let Ok(event) = event.as_ref() {
            self.observe(event);
            self.waker.wake();
        }
        event
    }

    /// Removes the next event a closed sender handed over.
    fn take_handoff(&self) -> Option<EngineCoreOutput> {
        self.handoff
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front()
    }

    /// Updates acknowledgement and completion state for a received event.
    fn observe(&mut self, event: &EngineCoreOutput) {
        match event {
            EngineCoreOutput::TextToken { .. } => {
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
            EngineCoreOutput::Finished { .. }
            | EngineCoreOutput::Rejected { .. }
            | EngineCoreOutput::Error { .. } => {
                self.finish();
            }
            _ => {}
        }
    }

    /// Acknowledges every text token observed through this receiver.
    ///
    /// Sends nothing when no token arrived since the last acknowledgement or
    /// when the receiver has no armed cancellation (built by `from_receiver`,
    /// finished, or already cancelled).
    pub fn acknowledge_consumed_prefix(&mut self) {
        if self.text_tokens_received <= self.acknowledged_token_count {
            return;
        }
        if let Some(cancellation) = self.cancellation.as_ref() {
            let _ = cancellation.handle.send(Command::Acknowledge {
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
    ///
    /// `DroppedStream` cancels at the acknowledged token count;
    /// `StopStringMatched` stops successfully after every text token received
    /// so far. Only the first cancellation of an armed receiver sends a
    /// command; see `acknowledge_consumed_prefix` for when it is disarmed.
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
        let _ = cancellation.handle.send(command);
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
    /// Cancels an unfinished request at its acknowledged prefix, runs an unrun
    /// completion callback, and wakes the scheduler so it observes the closed
    /// channel.
    fn drop(&mut self) {
        if let Some(cancellation) = self.cancellation.take() {
            let _ = cancellation.handle.send(Command::Cancel {
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

fn event_channel_with_waker(
    waker: uniserve_core::CommandWaker,
    cancellation: Option<EventCancellation>,
) -> (EventTx, EventRx) {
    let (tx, rx) = mpsc::channel(EVENT_BUFFER_CAPACITY);
    let handoff = Handoff::default();
    (
        EventTx {
            inner: tx,
            handoff: Arc::clone(&handoff),
        },
        EventRx {
            inner: rx,
            handoff,
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
        request: Box<Request>,
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
/// The scheduler stops when every sender of its command channel has dropped,
/// which includes the clones held by live [`EventRx`] values that are still
/// armed for cancellation.
///
/// Every send fires the `CommandWaker` right after enqueuing, so when the
/// scheduler is parked on an event-driven executor it wakes immediately to
/// observe the command instead of waiting out the park's safety-net timeout.
/// With the no-op waker, a parked scheduler observes commands only when its
/// executor poll returns.
#[derive(Clone)]
pub struct EngineHandle {
    tx: crossbeam_channel::Sender<Command>,
    waker: uniserve_core::CommandWaker,
}

impl EngineHandle {
    /// Constructs a handle with the no-op waker, for a scheduler that observes
    /// commands only through its timed executor poll.
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
    ///
    /// The returned receiver cancels the request when dropped before its
    /// terminal event. Fails with `SubmitError::Closed` when the command
    /// channel is closed; this handle never reports `SubmitError::Dead`,
    /// which `EngineCore::submit` checks before calling it.
    pub fn submit(&self, request: impl Into<Request>) -> Result<EventRx, SubmitError> {
        let request = request.into();
        let request_id = request.request_id();
        // Stop-string matching happens in the frontend decoder, which must
        // withhold acknowledgement until it has checked each token.
        let acknowledge_on_receive = match &request {
            Request::Ar(request) | Request::Umm(request) => request.stop_strings.is_empty(),
            Request::Diffusion(_) => false,
        };
        let (event_tx, event_rx) = event_channel_with_waker(
            self.waker.clone(),
            Some(EventCancellation {
                handle: self.clone(),
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
    use uniserve_core::{
        GenerationConstraint, GenerationRequest, ImageGenerationConfig, ImageParams, RequestId,
        SamplingParams,
    };

    use super::*;

    fn test_request(request_id: u64) -> GenerationRequest {
        let constraint = GenerationConstraint::UndOnly;
        let policy = ImageGenerationConfig::default();
        GenerationRequest {
            request_id: RequestId(request_id),
            prompt_token_ids: vec![1, 2, 3],
            multimodal_inputs: Default::default(),
            negative_prompt_token_ids: Vec::new(),
            constraint,
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_und_tokens: 32,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            image_generation: policy,
        }
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

    /// A request without stop strings acknowledges tokens on receipt. When the
    /// receiver drops after reading one of two published tokens, the cancel
    /// covers only the token the consumer actually received.
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
            .send(EngineCoreOutput::TextToken {
                id: 7,
                logprob: None,
            })
            .unwrap();
        event_tx
            .send(EngineCoreOutput::TextToken {
                id: 8,
                logprob: None,
            })
            .unwrap();

        assert!(matches!(
            events.try_recv(),
            Ok(EngineCoreOutput::TextToken { id: 7, .. })
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

    /// Events a closing sender hands over follow every event already buffered
    /// in the full channel, in order, and the stream reports closed only after
    /// the last of them.
    #[tokio::test]
    async fn receiver_yields_handed_over_events_after_the_channel_drains() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let mut events = handle.submit(test_request(13)).unwrap();
        let event_tx = match rx.recv().unwrap() {
            Command::Submit { event_tx, .. } => event_tx,
            _ => panic!("expected Submit command"),
        };

        let token = |id| EngineCoreOutput::TextToken { id, logprob: None };
        for id in 0..EVENT_BUFFER_CAPACITY as u32 {
            event_tx.send(token(id)).unwrap();
        }
        let capacity = EVENT_BUFFER_CAPACITY as u32;
        assert!(matches!(
            event_tx.send(token(capacity)),
            Err(EventSendError::Full(_))
        ));
        let terminal = EngineCoreOutput::Finished {
            reason: uniserve_core::FinishReason::Aborted,
            stop_reason: None,
            prompt_tokens: 3,
            completion_tokens: EVENT_BUFFER_CAPACITY + 1,
            images: 0,
        };
        event_tx.close_with(VecDeque::from([token(capacity), terminal]));

        for expected in 0..=capacity {
            match events.recv().await {
                Some(EngineCoreOutput::TextToken { id, .. }) => assert_eq!(id, expected),
                other => panic!("expected token {expected}, received {other:?}"),
            }
        }
        assert!(matches!(
            events.recv().await,
            Some(EngineCoreOutput::Finished {
                reason: uniserve_core::FinishReason::Aborted,
                ..
            })
        ));
        assert!(events.recv().await.is_none());
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
}
