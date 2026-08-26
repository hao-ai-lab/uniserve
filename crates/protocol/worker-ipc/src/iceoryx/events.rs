//! iceoryx2 event-service companions for the request-response boundary.
//!
//! The request-response ports (`Client`/`Server`) carry no file descriptor, so
//! they cannot be parked on directly — only an event `Listener` implements
//! `SynchronousMultiplexing`. Each request-response service therefore gets a
//! companion `<svc>/evt_host_wake` event service: the worker notifies after
//! sending a response, and the host's command ingress / worker-death watcher
//! notify here too; the host (client) parks here for {result, command, death}
//! (the scheduler park). A separate `<svc>/evt_worker_wake` service carries
//! request-ring notifications to the worker. Keeping the two directions separate is
//! essential: an event notifier broadcasts to every listener on its service, so
//! a shared service would enqueue every result on the worker's own listener
//! while it is busy on the GPU and eventually overflow that listener.
//!
//! Wait deadlines are supplied by the caller's operation or liveness contract.
//! They are not transport polling intervals: progress is signalled by an event.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use iceoryx2::port::listener::Listener;
use iceoryx2::port::notifier::Notifier;
use iceoryx2::prelude::*;
use iceoryx2::service::port_factory::event::PortFactory as EventFactory;

use super::IxService;

/// `evt_host_wake`: the worker signals the host that a response is available.
pub const EVT_RESULT: usize = 2;
/// `evt_host_wake`: command ingress signals that work was enqueued for the host.
pub const EVT_COMMAND: usize = 3;
/// `evt_host_wake`: the worker-death watcher signals worker exit to the host.
pub const EVT_DEATH: usize = 4;
/// `evt_worker_wake`: the host signals that an IPC request was queued.
pub const EVT_REQUEST: usize = 5;
/// `evt_worker_wake`: asynchronous worker progress became observable.
pub const EVT_COMPLETION: usize = 6;

fn host_wake_event_name(svc: &str) -> String {
    format!("{svc}/evt_host_wake")
}

fn worker_wake_event_name(svc: &str) -> String {
    format!("{svc}/evt_worker_wake")
}

fn open_event_service(
    node: &Node<IxService>,
    name: &str,
) -> anyhow::Result<EventFactory<IxService>> {
    let service_name = ServiceName::new(name)
        .map_err(|e| anyhow::anyhow!("event service name {name:?}: {e:?}"))?;
    node.service_builder(&service_name)
        .event()
        .open_or_create()
        .map_err(|e| anyhow::anyhow!("opening iceoryx2 event service {name:?}: {e:?}"))
}

fn make_notifier(
    factory: &EventFactory<IxService>,
    default_id: usize,
) -> anyhow::Result<Notifier<IxService>> {
    factory
        .notifier_builder()
        .default_event_id(EventId::new(default_id))
        .create()
        .map_err(|e| anyhow::anyhow!("creating iceoryx2 notifier: {e:?}"))
}

fn make_listener(factory: &EventFactory<IxService>) -> anyhow::Result<Listener<IxService>> {
    factory
        .listener_builder()
        .create()
        .map_err(|e| anyhow::anyhow!("creating iceoryx2 listener: {e:?}"))
}

/// A cloneable, thread-safe host wake source plus the `EventId` to stamp.
/// Commands and worker death can fire from arbitrary frontend / watcher threads.
#[derive(Clone)]
pub struct WakeSender {
    notifier: Arc<Notifier<IxService>>,
    event_id: usize,
    pending: Arc<AtomicBool>,
}

impl WakeSender {
    /// Fire one coalesced wake. A notifier failure clears the pending bit so a
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
    pub result: bool,
    pub command: bool,
    pub death: bool,
    /// A notification with an unrecognized id (forward-compat / spurious wake).
    pub other: bool,
}

impl WakeEvents {
    pub fn any(&self) -> bool {
        self.result || self.command || self.death || self.other
    }
}

/// Host-side event ports: a listener for {result, command, death}, a host-local
/// notifier, and a distinct notifier for the worker's request listener.
pub(crate) struct ClientEvents {
    wake_listener: Listener<IxService>,
    wake_notifier: Arc<Notifier<IxService>>,
    request_notifier: Notifier<IxService>,
    command_pending: Arc<AtomicBool>,
    death_pending: Arc<AtomicBool>,
}

impl ClientEvents {
    pub(crate) fn open(node: &Node<IxService>, service: &str) -> anyhow::Result<Self> {
        let wake = open_event_service(node, &host_wake_event_name(service))?;
        let request_wake = open_event_service(node, &worker_wake_event_name(service))?;
        let wake_notifier = Arc::new(make_notifier(&wake, EVT_COMMAND)?);
        let request_notifier = make_notifier(&request_wake, EVT_REQUEST)?;
        let wake_listener = make_listener(&wake)?;
        let command_pending = Arc::new(AtomicBool::new(false));
        let death_pending = Arc::new(AtomicBool::new(false));
        Ok(Self {
            wake_listener,
            wake_notifier,
            request_notifier,
            command_pending,
            death_pending,
        })
    }

    /// Park until a wake fires or `timeout` elapses, draining every pending
    /// event id so a backlog cannot cause an immediate re-wake spin.
    pub(crate) fn wait(&self, timeout: Duration) -> anyhow::Result<WakeEvents> {
        let mut ev = WakeEvents::default();
        self.command_pending.store(false, Ordering::Release);
        self.death_pending.store(false, Ordering::Release);
        self.wake_listener
            .timed_wait_all(
                |id| match id.as_value() {
                    EVT_RESULT => ev.result = true,
                    EVT_COMMAND => ev.command = true,
                    EVT_DEATH => ev.death = true,
                    _ => ev.other = true,
                },
                timeout,
            )
            .map_err(|e| anyhow::anyhow!("waiting on iceoryx2 wake listener: {e:?}"))?;
        Ok(ev)
    }

    pub(crate) fn drain(&self) -> anyhow::Result<WakeEvents> {
        let mut ev = WakeEvents::default();
        self.command_pending.store(false, Ordering::Release);
        self.death_pending.store(false, Ordering::Release);
        self.wake_listener
            .try_wait_all(|id| match id.as_value() {
                EVT_RESULT => ev.result = true,
                EVT_COMMAND => ev.command = true,
                EVT_DEATH => ev.death = true,
                _ => ev.other = true,
            })
            .map_err(|e| anyhow::anyhow!("draining iceoryx2 host wake listener: {e:?}"))?;
        Ok(ev)
    }

    pub(crate) fn file_descriptor(&self) -> i32 {
        // SAFETY: the listener owns this descriptor for at least as long as the
        // endpoint exposing it. Callers borrow it only while the endpoint lives.
        unsafe { self.wake_listener.file_descriptor().native_handle() }
    }

    /// A cloneable wake source the command ingress fires after enqueuing a
    /// command.
    pub(crate) fn command_wake(&self) -> WakeSender {
        WakeSender {
            notifier: Arc::clone(&self.wake_notifier),
            event_id: EVT_COMMAND,
            pending: Arc::clone(&self.command_pending),
        }
    }

    /// A cloneable wake source the worker-death watcher fires on child exit, so
    /// an idle host detects death immediately.
    pub(crate) fn death_wake(&self) -> WakeSender {
        WakeSender {
            notifier: Arc::clone(&self.wake_notifier),
            event_id: EVT_DEATH,
            pending: Arc::clone(&self.death_pending),
        }
    }

    /// Tell the worker that one or more requests are available in the ring.
    pub(crate) fn notify_request(&self) {
        let _ = self
            .request_notifier
            .notify_with_custom_event_id(EventId::new(EVT_REQUEST));
    }
}

/// Worker-side event ports: a notifier fired after each response on the host
/// service and a listener on the separate request service.
pub(crate) struct ServerEvents {
    wake_notifier: Notifier<IxService>,
    wake_listener: Listener<IxService>,
    completion_notifier: Arc<Notifier<IxService>>,
    completion_pending: Arc<AtomicBool>,
}

impl ServerEvents {
    pub(crate) fn open(node: &Node<IxService>, service: &str) -> anyhow::Result<Self> {
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

    /// Tell the host a response is available in the request-response ring.
    pub(crate) fn notify_response(&self) {
        let _ = self
            .wake_notifier
            .notify_with_custom_event_id(EventId::new(EVT_RESULT));
    }

    /// Park until an inbound request wake fires or `timeout` elapses, draining
    /// every pending event id. The IPC client fires `EVT_REQUEST` after send, so
    /// an idle server wakes immediately. The timeout belongs to the caller's
    /// liveness or shutdown deadline.
    pub(crate) fn wait_request(&self, timeout: Duration) -> anyhow::Result<()> {
        self.wake_listener
            .timed_wait_all(
                |id| {
                    if id.as_value() == EVT_COMPLETION {
                        self.completion_pending.store(false, Ordering::Release);
                    }
                },
                timeout,
            )
            .map_err(|e| anyhow::anyhow!("waiting on iceoryx2 server wake listener: {e:?}"))?;
        Ok(())
    }

    /// Drain wake hints after consuming directly from the request ring. A busy
    /// worker may never need to park, so keeping the listener aligned with ring
    /// consumption prevents bounded event capacity from becoming backpressure.
    pub(crate) fn drain_worker_wakes(&self) -> anyhow::Result<()> {
        self.wake_listener
            .try_wait_all(|id| {
                if id.as_value() == EVT_COMPLETION {
                    self.completion_pending.store(false, Ordering::Release);
                }
            })
            .map_err(|e| anyhow::anyhow!("draining iceoryx2 worker wake listener: {e:?}"))?;
        Ok(())
    }

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
        let command = events.command_wake();
        let death = events.death_wake();

        for _ in 0..10_000 {
            command.wake();
        }
        assert!(
            events
                .wait(Duration::from_secs(1))
                .expect("drain command wake")
                .command
        );
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

        command.wake();
        assert!(
            events
                .wait(Duration::from_secs(1))
                .expect("receive next command wake")
                .command
        );
        death.wake();
        assert!(
            events
                .wait(Duration::from_secs(1))
                .expect("receive death wake")
                .death
        );
    }
}
