//! iceoryx2 event-service companions that make the request-response boundary
//! event-driven instead of polled.
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
//! Every wait carries a bounded safety-net timeout ([`EVENT_WAIT_SAFETY_NET`]),
//! so a missed notification degrades to the old poll latency instead of
//! hanging: correctness is identical to polling, only the common-case latency
//! drops toward the raw shm transfer time. This is what makes the event path
//! safe to enable by default and trivially reversible.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use iceoryx2::port::listener::Listener;
use iceoryx2::port::notifier::Notifier;
use iceoryx2::prelude::*;
use iceoryx2::service::port_factory::event::PortFactory as EventFactory;

use crate::IxService;

/// `evt_host_wake`: the worker signals the host that a response is available.
pub const EVT_RESULT: usize = 2;
/// `evt_host_wake`: command ingress signals that work was enqueued for the host.
pub const EVT_COMMAND: usize = 3;
/// `evt_host_wake`: the worker-death watcher signals worker exit to the host.
pub const EVT_DEATH: usize = 4;
/// `evt_worker_wake`: the host signals that an IPC request was queued.
pub const EVT_REQUEST: usize = 5;

/// Env switch for the event-driven boundary. Default on; set to `0`/`false`/
/// `off` to fall back to the fixed-interval poll on both ends. The host
/// process and the worker it spawns inherit the same value, so the two ends
/// always agree; even if they did not, a mismatch only costs poll latency
/// (the safety-net timeout), never correctness.
pub const EVENT_DRIVEN_ENV: &str = "UNISERVE_IPC_EVENT_DRIVEN";

/// Upper bound on how long any event wait blocks before re-checking the
/// transport directly. Matches the 1ms poll interval, so the worst case
/// when a notification is missed is exactly the poll-based behavior.
pub const EVENT_WAIT_SAFETY_NET: Duration = Duration::from_millis(1);

/// Whether the event-driven boundary is enabled (see [`EVENT_DRIVEN_ENV`]).
pub fn event_driven_enabled() -> bool {
    match std::env::var(EVENT_DRIVEN_ENV) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off" | ""
        ),
        Err(_) => true,
    }
}

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
    /// Fire the wake. Errors are swallowed: a wake is a best-effort latency
    /// optimization over the parked listener's safety-net timeout, never a
    /// correctness requirement, so a transient notify failure must not surface
    /// as a command/teardown error.
    pub fn wake(&self) {
        if self.pending.swap(true, Ordering::AcqRel) {
            return;
        }
        let _ = self
            .notifier
            .notify_with_custom_event_id(EventId::new(self.event_id));
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

    /// A cloneable wake source the command ingress fires after enqueuing a
    /// command, so the parked host wakes immediately instead of after the
    /// safety-net timeout.
    pub(crate) fn command_wake(&self) -> WakeSender {
        WakeSender {
            notifier: Arc::clone(&self.wake_notifier),
            event_id: EVT_COMMAND,
            pending: Arc::clone(&self.command_pending),
        }
    }

    /// A cloneable wake source the worker-death watcher fires on child exit, so
    /// an idle host detects death immediately (no liveness-poll floor).
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
}

impl ServerEvents {
    pub(crate) fn open(node: &Node<IxService>, service: &str) -> anyhow::Result<Self> {
        let wake = open_event_service(node, &host_wake_event_name(service))?;
        let request_wake = open_event_service(node, &worker_wake_event_name(service))?;
        let wake_notifier = make_notifier(&wake, EVT_RESULT)?;
        let wake_listener = make_listener(&request_wake)?;
        Ok(Self {
            wake_notifier,
            wake_listener,
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
    /// an idle server wakes immediately instead of polling; the timeout is the
    /// safety-net re-check floor for a missed notification.
    pub(crate) fn wait_request(&self, timeout: Duration) -> anyhow::Result<()> {
        self.wake_listener
            .timed_wait_all(|_id| {}, timeout)
            .map_err(|e| anyhow::anyhow!("waiting on iceoryx2 server wake listener: {e:?}"))?;
        Ok(())
    }

    /// Drain request notifications after consuming from the ring. This keeps a
    /// busy worker's event queue aligned with the ring it is already servicing.
    pub(crate) fn drain_requests(&self) -> anyhow::Result<()> {
        self.wake_listener
            .try_wait_all(|_id| {})
            .map_err(|e| anyhow::anyhow!("draining iceoryx2 worker wake listener: {e:?}"))?;
        Ok(())
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
