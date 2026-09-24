//! Directional wake services for the iceoryx2 request-response boundary.
//!
//! Request-response ports do not expose a file descriptor suitable for parking.
//! A host wake service reports results and worker death, while a worker wake
//! service reports submitted requests and asynchronous completions. Separate
//! directions prevent broadcast notifications from accumulating on the
//! sender's listener.
//!
//! Callers supply call or liveness deadlines to all waits.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use iceoryx2::port::listener::Listener;
use iceoryx2::port::notifier::Notifier;
use iceoryx2::prelude::*;
use iceoryx2::service::port_factory::event::PortFactory as EventFactory;

use super::{IpcError, IpcResult, IxService};

macro_rules! ipc_error {
    ($($arg:tt)*) => {
        IpcError::transport(format!($($arg)*))
    };
}

/// Park on the event descriptor while preserving the caller's deadline across
/// signal interruptions. iceoryx2's timed receive erases EINTR into a terminal
/// InternalFailure, so use its descriptor and then its nonblocking drain.
///
/// Returns `Ok(())` both when the descriptor is readable and when `timeout`
/// elapses; callers learn what fired from the drain that follows. Fails when
/// poll fails with anything other than EINTR or reports the descriptor in an
/// error state.
fn wait_readable(listener: &Listener<IxService>, timeout: Duration) -> IpcResult<()> {
    let started = Instant::now();
    let mut descriptor = libc::pollfd {
        // SAFETY: the listener owns the descriptor throughout this wait.
        fd: unsafe { listener.file_descriptor().native_handle() },
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Ok(());
        }
        // Round up to whole milliseconds so a sub-millisecond remainder parks
        // instead of becoming a zero-timeout poll that spins until the
        // deadline.
        let milliseconds =
            remaining.as_millis() + u128::from(!remaining.subsec_nanos().is_multiple_of(1_000_000));
        // SAFETY: poll borrows one initialized descriptor and retains no pointer.
        let result = unsafe {
            libc::poll(
                &raw mut descriptor,
                1,
                milliseconds.min(i32::MAX as u128) as i32,
            )
        };
        if result < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(ipc_error!("waiting on worker event descriptor: {error}"));
        }
        if descriptor.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
            return Err(ipc_error!("worker event descriptor is unavailable"));
        }
        // A zero result before the deadline polls again for the remainder.
        if result > 0 || started.elapsed() >= timeout {
            return Ok(());
        }
    }
}

/// `evt_host_wake`: the worker signals the host that a response is available.
pub const EVT_RESULT: usize = 2;
/// `evt_host_wake`: the worker-death watcher signals worker exit to the host.
pub const EVT_DEATH: usize = 4;
/// `evt_worker_wake`: the host signals that an IPC request was queued.
pub const EVT_REQUEST: usize = 5;
/// `evt_worker_wake`: asynchronous worker progress became observable.
pub const EVT_COMPLETION: usize = 6;

/// Builds the event-service name listened to by the host.
fn host_wake_event_name(svc: &str) -> String {
    format!("{svc}/evt_host_wake")
}

/// Builds the event-service name listened to by the worker.
fn worker_wake_event_name(svc: &str) -> String {
    format!("{svc}/evt_worker_wake")
}

/// Opens or creates one named iceoryx2 event service.
fn open_event_service(node: &Node<IxService>, name: &str) -> IpcResult<EventFactory<IxService>> {
    let service_name =
        ServiceName::new(name).map_err(|e| ipc_error!("event service name {name:?}: {e:?}"))?;
    node.service_builder(&service_name)
        .event()
        .open_or_create()
        .map_err(|e| ipc_error!("opening iceoryx2 event service {name:?}: {e:?}"))
}

/// Creates a notifier whose ordinary wake uses `default_id`.
fn make_notifier(
    factory: &EventFactory<IxService>,
    default_id: usize,
) -> IpcResult<Notifier<IxService>> {
    factory
        .notifier_builder()
        .default_event_id(EventId::new(default_id))
        .create()
        .map_err(|e| ipc_error!("creating iceoryx2 notifier: {e:?}"))
}

/// Creates the listener paired with an event service.
fn make_listener(factory: &EventFactory<IxService>) -> IpcResult<Listener<IxService>> {
    factory
        .listener_builder()
        .create()
        .map_err(|e| ipc_error!("creating iceoryx2 listener: {e:?}"))
}

/// A cloneable, thread-safe wake source plus the `EventId` to stamp.
///
/// It backs the host's worker-death wake (`ClientEvents::death_wake`) and the
/// worker's completion wake (`ServerEvents::completion_wake`). Both fire from
/// threads other than the endpoint's owner, such as frontend, watcher, and
/// completion-callback threads. Clones share one pending bit, so wakes fired
/// between two listener drains coalesce into one notification.
#[derive(Clone)]
pub struct WakeSender {
    /// Shared event notifier safe to call from callback and watcher threads.
    notifier: Arc<Notifier<IxService>>,
    /// Event identity stamped onto each notification.
    event_id: usize,
    /// Coalescing bit cleared when the corresponding listener drains; set
    /// while a notification is outstanding.
    pending: Arc<AtomicBool>,
}

impl WakeSender {
    /// Fires one coalesced wake. A notifier failure clears the pending bit so a
    /// later producer can retry; endpoint teardown remains intentionally
    /// non-panicking for callback and watcher threads.
    pub fn wake(&self) {
        if self.pending.swap(true, Ordering::AcqRel) {
            return;
        }
        if self
            .notifier
            .notify_with_custom_event_id(EventId::new(self.event_id))
            .is_err()
        {
            self.pending.store(false, Ordering::Release);
        }
    }
}

/// Which wake sources fired during a [`ClientEvents::wait`].
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct WakeEvents {
    /// A worker response became available.
    pub result: bool,
    /// A monitored worker process exited.
    pub death: bool,
    /// A notification carried an unrecognized event identity.
    pub other: bool,
}

impl WakeEvents {
    /// Returns whether at least one wake category was observed.
    pub fn any(&self) -> bool {
        self.result || self.death || self.other
    }
}

/// Host-side event ports: a listener for {result, death}, a host-local
/// notifier, and a distinct notifier for the worker's request listener.
pub(crate) struct ClientEvents {
    /// Host-facing listener for results and process death.
    wake_listener: Listener<IxService>,
    /// Host-local notifier used by death wake senders.
    wake_notifier: Arc<Notifier<IxService>>,
    /// Notifier targeting the worker's request listener.
    request_notifier: Notifier<IxService>,
    /// Coalescing state for process-death wakes.
    death_pending: Arc<AtomicBool>,
}

impl ClientEvents {
    /// Opens host-facing result and death wake ports.
    pub(crate) fn open(node: &Node<IxService>, service: &str) -> IpcResult<Self> {
        let wake = open_event_service(node, &host_wake_event_name(service))?;
        let request_wake = open_event_service(node, &worker_wake_event_name(service))?;
        let wake_notifier = Arc::new(make_notifier(&wake, EVT_DEATH)?);
        let request_notifier = make_notifier(&request_wake, EVT_REQUEST)?;
        let wake_listener = make_listener(&wake)?;
        let death_pending = Arc::new(AtomicBool::new(false));
        Ok(Self {
            wake_listener,
            wake_notifier,
            request_notifier,
            death_pending,
        })
    }

    /// Parks until a wake fires or `timeout` elapses, draining every pending
    /// event id so a backlog cannot cause an immediate re-wake spin.
    pub(crate) fn wait(&self, timeout: Duration) -> IpcResult<WakeEvents> {
        wait_readable(&self.wake_listener, timeout)?;
        self.drain()
    }

    /// Drains pending host wakes and classifies their event identifiers.
    pub(crate) fn drain(&self) -> IpcResult<WakeEvents> {
        let mut ev = WakeEvents::default();

        // Re-arm coalesced local producers before draining event identities, so
        // a death wake fired during or after this drain sends a fresh
        // notification instead of being absorbed by one already drained.
        self.death_pending.store(false, Ordering::Release);
        self.wake_listener
            .try_wait_all(|id| match id.as_value() {
                EVT_RESULT => ev.result = true,
                EVT_DEATH => ev.death = true,
                _ => ev.other = true,
            })
            .map_err(|e| ipc_error!("draining iceoryx2 host wake listener: {e:?}"))?;
        Ok(ev)
    }

    /// Returns the listener descriptor used by external poll loops.
    pub(crate) fn file_descriptor(&self) -> i32 {
        // SAFETY: the listener owns this descriptor for at least as long as the
        // endpoint exposing it. Callers borrow it only while the endpoint lives.
        unsafe { self.wake_listener.file_descriptor().native_handle() }
    }

    /// Returns a cloneable wake source for worker-process exit.
    pub(crate) fn death_wake(&self) -> WakeSender {
        WakeSender {
            notifier: Arc::clone(&self.wake_notifier),
            event_id: EVT_DEATH,
            pending: Arc::clone(&self.death_pending),
        }
    }

    /// Tells the worker that one or more requests are available in the ring.
    ///
    /// A notify failure is ignored; the request stays in the ring, where a
    /// parked worker finds it only when it re-checks after its wait times out.
    pub(crate) fn notify_request(&self) {
        let _ = self
            .request_notifier
            .notify_with_custom_event_id(EventId::new(EVT_REQUEST));
    }
}

/// Worker-side event ports: a notifier fired after each response on the host
/// service and a listener on the separate request service.
pub(crate) struct ServerEvents {
    /// Notifier targeting the host's result listener.
    wake_notifier: Notifier<IxService>,
    /// Worker-facing listener for requests and asynchronous completions.
    wake_listener: Listener<IxService>,
    /// Shared notifier for completion callbacks.
    completion_notifier: Arc<Notifier<IxService>>,
    /// Coalescing state for asynchronous completion wakes.
    completion_pending: Arc<AtomicBool>,
}

impl ServerEvents {
    /// Opens worker-facing request and completion wake ports.
    pub(crate) fn open(node: &Node<IxService>, service: &str) -> IpcResult<Self> {
        let wake = open_event_service(node, &host_wake_event_name(service))?;
        let request_wake = open_event_service(node, &worker_wake_event_name(service))?;
        let wake_notifier = make_notifier(&wake, EVT_RESULT)?;
        let completion_notifier = Arc::new(make_notifier(&request_wake, EVT_COMPLETION)?);
        let wake_listener = make_listener(&request_wake)?;
        Ok(Self {
            wake_notifier,
            wake_listener,
            completion_notifier,
            completion_pending: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Tells the host a response is available in the request-response ring.
    ///
    /// A notify failure is ignored; the response stays in the ring, where a
    /// parked host finds it only when it re-checks after its wait times out.
    pub(crate) fn notify_response(&self) {
        let _ = self
            .wake_notifier
            .notify_with_custom_event_id(EventId::new(EVT_RESULT));
    }

    /// Parks until an inbound request or completion wake fires or `timeout`
    /// elapses, draining every pending event id. The IPC client fires
    /// `EVT_REQUEST` after send, so an idle server wakes immediately. The
    /// timeout belongs to the caller's liveness or shutdown deadline.
    pub(crate) fn wait_request(&self, timeout: Duration) -> IpcResult<()> {
        wait_readable(&self.wake_listener, timeout)?;
        self.drain_worker_wakes()
    }

    /// Drains wake hints after consuming directly from the request ring. A busy
    /// worker may never need to park, so keeping the listener aligned with ring
    /// consumption prevents bounded event capacity from becoming backpressure.
    pub(crate) fn drain_worker_wakes(&self) -> IpcResult<()> {
        self.wake_listener
            .try_wait_all(|id| {
                // Re-arm the completion producer once its notification is
                // drained, so the next completion wake notifies again.
                if id.as_value() == EVT_COMPLETION {
                    self.completion_pending.store(false, Ordering::Release);
                }
            })
            .map_err(|e| ipc_error!("draining iceoryx2 worker wake listener: {e:?}"))?;
        Ok(())
    }

    /// Returns a sender for notifying asynchronous completion readiness.
    pub(crate) fn completion_wake(&self) -> WakeSender {
        WakeSender {
            notifier: Arc::clone(&self.completion_notifier),
            event_id: EVT_COMPLETION,
            pending: Arc::clone(&self.completion_pending),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn wake_sources_deliver_across_busy_periods_and_listener_drains() {
        // Request and completion wakes reach the worker's listener, and result
        // and death wakes reach the host's.
        let node = NodeBuilder::new()
            .create::<IxService>()
            .expect("create event test node");
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let service = format!("uniserve/test/events/{}/{nonce}", std::process::id());
        let events = ClientEvents::open(&node, &service).expect("open client events");
        let server = ServerEvents::open(&node, &service).expect("open server events");
        let death = events.death_wake();

        events.notify_request();
        server
            .wait_request(Duration::from_secs(1))
            .expect("drain worker request wake");
        server.completion_wake().wake();
        server
            .wait_request(Duration::from_secs(1))
            .expect("receive worker completion wake");

        server.notify_response();
        assert!(
            events
                .wait(Duration::from_secs(1))
                .expect("receive result wake")
                .result
        );

        // A burst of death wakes before one drain coalesces into a single
        // pending notification that the host still observes.
        for _ in 0..10_000 {
            death.wake();
        }
        assert!(
            events
                .wait(Duration::from_secs(1))
                .expect("receive death wake")
                .death
        );
    }
}
