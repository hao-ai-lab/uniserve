//! Shared IPC primitives for the iceoryx2 worker request-response boundary.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use iceoryx2::active_request::ActiveRequest;
use iceoryx2::pending_response::PendingResponse;
use iceoryx2::port::client::Client;
use iceoryx2::port::server::Server;
use iceoryx2::prelude::*;
use iceoryx2_bb_elementary_traits::zero_copy_send::ZeroCopySend;
use uniserve_worker_wire::flat::{
    decode_request, decode_response, encode_request, encode_response,
};
use uniserve_worker_wire::{RequestKind, ResponseKind, WorkerRequest, WorkerResponse};

mod events;
use events::{ClientEvents, ServerEvents};
pub use events::{
    EVENT_DRIVEN_ENV, EVENT_WAIT_SAFETY_NET, EVT_COMMAND, EVT_DEATH, EVT_RESULT, WakeEvents,
    WakeSender, event_driven_enabled,
};

pub mod transfer_agent;
pub use transfer_agent::{
    AgentConfig, InProcessAgent, LocalAddr, MemoryRegion, RegisteredRegion, RemoteAddr,
    RemoteSegment, TransferAgent, TransferOp, TransferReq, TransferStatus, TransferTicket,
    make_transfer_agent,
};

pub const DEFAULT_SERVICE_PREFIX: &str = "uniserve/worker";

/// Wire protocol version this build emits on every [`Header`].
pub const WIRE_VERSION: u16 = 5;

/// Whether a peer-advertised wire `version` is one this build can decode.
pub fn is_supported_wire_version(version: u16) -> bool {
    version == WIRE_VERSION
}

/// Fixed-size IPC frame header sent zero-copy across the worker process
/// boundary.
///
/// Fields are ordered largest-alignment-first so the `#[repr(C)]` layout is
/// completely free of implicit padding. This matters because the header is
/// shipped via [`ZeroCopySend`]: any padding bytes the compiler inserted would
/// be copied verbatim across the process boundary while uninitialized, leaking
/// stale stack/heap contents and making byte-for-byte comparison of two
/// logically-equal headers unreliable. The current layout is exactly
/// `3*u64 + 3*u32 + u16 + 2*u8 = 40` bytes with no holes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub step_id: u64,
    pub op_id: u64,
    pub call_id: u64,
    pub len: u32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub version: u16,
    pub kind: u8,
    pub flags: u8,
}

impl Default for Header {
    fn default() -> Self {
        Self {
            step_id: 0,
            op_id: 0,
            call_id: 0,
            len: 0,
            reserved0: 0,
            reserved1: 0,
            version: WIRE_VERSION,
            kind: 0,
            flags: 0,
        }
    }
}

unsafe impl ZeroCopySend for Header {}

// Guard the padding-free invariant: if a future field change reintroduces
// implicit padding, `size_of::<Header>` will no longer equal the sum of the
// field sizes and this assertion fails the build, signalling that
// uninitialized padding bytes would otherwise be sent zero-copy.
const _: () = assert!(
    std::mem::size_of::<Header>() == 3 * 8 + 3 * 4 + 2 + 2,
    "Header must be padding-free for zero-copy transmission"
);

pub struct Frame {
    pub header: Header,
    pub payload: Vec<u8>,
}

impl Frame {
    pub fn decode_request(&self) -> anyhow::Result<WorkerRequest> {
        decode_request(&self.payload)
    }

    pub fn decode_response(&self) -> anyhow::Result<WorkerResponse> {
        decode_response(&self.payload)
    }
}

pub(crate) type IxService = iceoryx2::service::ipc_threadsafe::Service;
type IxClient = Client<IxService, [u8], Header, [u8], Header>;
type IxServer = Server<IxService, [u8], Header, [u8], Header>;
type IxActive = ActiveRequest<IxService, [u8], Header, [u8], Header>;
pub type Pending = PendingResponse<IxService, [u8], Header, [u8], Header>;

/// Convert an encoded payload length to the `u32` carried in [`Header::len`],
/// rejecting payloads that do not fit instead of silently truncating with an
/// `as u32` cast (which would corrupt the length on the receiving side).
fn payload_len_u32(len: usize) -> anyhow::Result<u32> {
    u32::try_from(len).map_err(|_| {
        anyhow::anyhow!(
            "IPC payload of {len} bytes exceeds the maximum frame size of {} bytes",
            u32::MAX
        )
    })
}

pub fn service_name(id: &str) -> String {
    if id.contains('/') {
        id.to_string()
    } else {
        format!("{DEFAULT_SERVICE_PREFIX}/{id}")
    }
}

pub struct ClientEndpoint {
    _node: Node<IxService>,
    client: IxClient,
    connect_timeout: Duration,
    /// Companion event ports for the event-driven boundary (None when polling).
    events: Option<ClientEvents>,
}

impl ClientEndpoint {
    pub fn connect(
        service: &str,
        initial_max_slice_len: usize,
        max_inflight: usize,
    ) -> anyhow::Result<Self> {
        Self::connect_with(
            service,
            initial_max_slice_len,
            max_inflight,
            event_driven_enabled(),
        )
    }

    /// As [`Self::connect`], but with an explicit event-driven choice (the host
    /// is authoritative: it spawns the worker with the matching setting).
    pub fn connect_with(
        service: &str,
        initial_max_slice_len: usize,
        max_inflight: usize,
        event_driven: bool,
    ) -> anyhow::Result<Self> {
        let service_name = ServiceName::new(service)?;
        let node = NodeBuilder::new()
            .create::<IxService>()
            .context("creating iceoryx2 client node")?;
        let factory = node
            .service_builder(&service_name)
            .request_response::<[u8], [u8]>()
            .request_user_header::<Header>()
            .response_user_header::<Header>()
            .max_active_requests_per_client(max_inflight.max(1))
            .max_response_buffer_size(max_inflight.max(1))
            .max_loaned_requests(max_inflight.max(1))
            .enable_safe_overflow_for_requests(false)
            .enable_safe_overflow_for_responses(false)
            .open_or_create()
            .context("opening iceoryx2 request-response service")?;
        let client = factory
            .client_builder()
            .initial_max_slice_len(initial_max_slice_len.max(1))
            .allocation_strategy(AllocationStrategy::PowerOfTwo)
            .create()
            .context("creating iceoryx2 client port")?;
        let events = if event_driven {
            Some(ClientEvents::open(&node, service).context("opening client event companions")?)
        } else {
            None
        };
        Ok(Self {
            _node: node,
            client,
            connect_timeout: Duration::from_secs(300),
            events,
        })
    }

    /// Whether the event-driven boundary is active on this endpoint.
    pub fn is_event_driven(&self) -> bool {
        self.events.is_some()
    }

    /// Park for {result, command, death} until a wake fires or `timeout`
    /// elapses. Returns which sources fired. Only valid on an event-driven
    /// endpoint; pollers must use [`Self::try_recv_response`] on a deadline.
    pub fn wait_wake(&self, timeout: Duration) -> anyhow::Result<WakeEvents> {
        match &self.events {
            Some(ev) => ev.wait(timeout),
            None => {
                std::thread::sleep(timeout.min(EVENT_WAIT_SAFETY_NET));
                Ok(WakeEvents::default())
            }
        }
    }

    /// A cloneable wake source the command ingress fires after enqueuing a
    /// command (None when polling).
    pub fn command_wake(&self) -> Option<WakeSender> {
        self.events.as_ref().map(ClientEvents::command_wake)
    }

    /// A cloneable wake source the worker-death watcher fires on child exit
    /// (None when polling).
    pub fn death_wake(&self) -> Option<WakeSender> {
        self.events.as_ref().map(ClientEvents::death_wake)
    }

    pub fn send_request(&self, req: &WorkerRequest) -> anyhow::Result<Pending> {
        let payload = encode_request(req)?;
        let mut header = header_for_request(req);
        header.len = payload_len_u32(payload.len())?;
        self.send_raw(header, &payload)
    }

    pub fn send_request_attempt(&self, req: &WorkerRequest) -> anyhow::Result<Pending> {
        let payload = encode_request(req)?;
        let mut header = header_for_request(req);
        header.len = payload_len_u32(payload.len())?;
        self.send_raw_attempt(header, &payload)
    }

    pub fn send_raw(&self, header: Header, payload: &[u8]) -> anyhow::Result<Pending> {
        let deadline = Instant::now() + self.connect_timeout;
        loop {
            let pending = self.send_raw_attempt(header, payload)?;
            if pending.number_of_server_connections() > 0 {
                return Ok(pending);
            }
            drop(pending);
            if Instant::now() >= deadline {
                bail!("iceoryx2 worker service has no connected server");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    pub fn send_raw_attempt(&self, header: Header, payload: &[u8]) -> anyhow::Result<Pending> {
        let mut request = self
            .client
            .loan_slice_uninit(payload.len())
            .context("loaning iceoryx2 request slice")?;
        *request.user_header_mut() = header;
        let request = request.write_from_slice(payload);
        let pending = request.send().context("sending iceoryx2 request")?;
        Ok(pending)
    }

    pub fn try_recv_response(&self, pending: &Pending) -> anyhow::Result<Option<Frame>> {
        let Some(response) = pending.receive().context("receiving iceoryx2 response")? else {
            return Ok(None);
        };
        let header = *response.user_header();
        let payload = response.payload().to_vec();
        verify_header_len(header, payload.len())?;
        Ok(Some(Frame { header, payload }))
    }

    pub fn recv_response_timeout(
        &self,
        pending: &Pending,
        timeout: Duration,
    ) -> anyhow::Result<Option<Frame>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(frame) = self.try_recv_response(pending)? {
                return Ok(Some(frame));
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            let slice = (deadline - now).min(EVENT_WAIT_SAFETY_NET);
            // Event-driven: park on the wake listener (wakes on a real response,
            // falls back to the safety-net slice otherwise). Polling: sleep the slice.
            match &self.events {
                Some(events) => {
                    events.wait(slice)?;
                }
                None => std::thread::sleep(slice),
            }
        }
    }
}

pub struct ServerEndpoint {
    _node: Node<IxService>,
    server: IxServer,
    active: VecDeque<(u64, IxActive)>,
    /// Companion event ports for the event-driven boundary (None when polling).
    events: Option<ServerEvents>,
}

impl ServerEndpoint {
    pub fn bind(
        service: &str,
        initial_max_slice_len: usize,
        max_inflight: usize,
    ) -> anyhow::Result<Self> {
        Self::bind_with(
            service,
            initial_max_slice_len,
            max_inflight,
            event_driven_enabled(),
        )
    }

    /// As [`Self::bind`], but with an explicit event-driven choice (so the
    /// worker can be forced to match a host that disabled it).
    pub fn bind_with(
        service: &str,
        initial_max_slice_len: usize,
        max_inflight: usize,
        event_driven: bool,
    ) -> anyhow::Result<Self> {
        let service_name = ServiceName::new(service)?;
        let node = NodeBuilder::new()
            .create::<IxService>()
            .context("creating iceoryx2 server node")?;
        let factory = node
            .service_builder(&service_name)
            .request_response::<[u8], [u8]>()
            .request_user_header::<Header>()
            .response_user_header::<Header>()
            .max_active_requests_per_client(max_inflight.max(1))
            .max_response_buffer_size(max_inflight.max(1))
            .max_loaned_requests(max_inflight.max(1))
            .enable_safe_overflow_for_requests(false)
            .enable_safe_overflow_for_responses(false)
            .open_or_create()
            .context("opening iceoryx2 request-response service")?;
        let server = factory
            .server_builder()
            .initial_max_slice_len(initial_max_slice_len.max(1))
            .allocation_strategy(AllocationStrategy::PowerOfTwo)
            .create()
            .context("creating iceoryx2 server port")?;
        let events = if event_driven {
            Some(ServerEvents::open(&node, service).context("opening server event companions")?)
        } else {
            None
        };
        Ok(Self {
            _node: node,
            server,
            active: VecDeque::new(),
            events,
        })
    }

    /// Whether the event-driven boundary is active on this endpoint.
    pub fn is_event_driven(&self) -> bool {
        self.events.is_some()
    }

    pub fn try_recv(&mut self) -> anyhow::Result<Option<Frame>> {
        let Some(active) = self
            .server
            .receive()
            .context("receiving iceoryx2 request")?
        else {
            return Ok(None);
        };
        let header = *active.user_header();
        let payload = active.payload().to_vec();
        verify_header_len(header, payload.len())?;
        self.active.push_back((header.call_id, active));
        Ok(Some(Frame { header, payload }))
    }

    pub fn recv(&mut self) -> anyhow::Result<Frame> {
        loop {
            if let Some(frame) = self.try_recv()? {
                return Ok(frame);
            }
            std::thread::sleep(EVENT_WAIT_SAFETY_NET);
        }
    }

    pub fn respond(&mut self, resp: &WorkerResponse) -> anyhow::Result<()> {
        let payload = encode_response(resp)?;
        let mut header = header_for_response(resp);
        header.len = payload_len_u32(payload.len())?;
        self.respond_raw(header, &payload)
    }

    pub fn respond_raw(&mut self, header: Header, payload: &[u8]) -> anyhow::Result<()> {
        let pos = self
            .active
            .iter()
            .position(|(call_id, _)| *call_id == header.call_id)
            .with_context(|| {
                format!(
                    "respond called for unknown active request call_id {}",
                    header.call_id
                )
            })?;
        let active = self
            .active
            .remove(pos)
            .map(|(_, active)| active)
            .context("active request position disappeared")?;
        let mut response = active
            .loan_slice_uninit(payload.len())
            .context("loaning iceoryx2 response slice")?;
        *response.user_header_mut() = header;
        let response = response.write_from_slice(payload);
        response.send().context("sending iceoryx2 response")?;
        // Wake the host the instant the response is queued, so its result wait
        // returns from the event listener rather than the safety-net poll.
        if let Some(events) = &self.events {
            events.notify_response();
        }
        Ok(())
    }
}

/// Build the IPC header that accompanies `req`.
///
/// `call_id` is the authoritative request/response correlation key. The
/// `step_id`/`op_id` fields are populated from the batch's `step_id` and the
/// *first* op only, as a cheap at-a-glance diagnostic hint (e.g. for tracing);
/// they are **not** a per-op index. A batch may carry many ops with distinct
/// `op_id`s, so consumers must read the full decoded payload rather than
/// treating `header.op_id` as identifying every op in the frame.
pub fn header_for_request(req: &WorkerRequest) -> Header {
    let mut h = Header {
        kind: request_kind_code(req.kind),
        call_id: req.call_id.unwrap_or_default(),
        ..Default::default()
    };
    if let Some(batch) = &req.batch {
        h.step_id = batch.step_id;
        // Hint only: first op's id. See the doc comment above.
        if let Some(operation) = batch.operations().next() {
            h.op_id = operation.op_id.0;
        }
    }
    h
}

/// Build the IPC header that accompanies `resp`.
///
/// As in [`header_for_request`], `call_id` is the authoritative correlation
/// key. `step_id`/`op_id` are populated from the result's `step_id` and the
/// *first* per-seq entry only, as a diagnostic hint; a result may aggregate
/// many sequences with distinct `op_id`s, so consumers must use the decoded
/// payload to correlate individual sequences.
pub fn header_for_response(resp: &WorkerResponse) -> Header {
    let mut h = Header {
        kind: response_kind_code(resp.kind),
        call_id: resp.call_id.unwrap_or_default(),
        ..Default::default()
    };
    if let Some(report) = &resp.completion_report {
        h.step_id = report.step_id;
        // Hint only: first completion's op id. See the doc comment above.
        if let Some(completion) = report.completions().next() {
            h.op_id = completion.op_id.0;
        }
    }
    h
}

fn verify_header_len(header: Header, actual: usize) -> anyhow::Result<()> {
    if !is_supported_wire_version(header.version) {
        bail!(
            "unsupported IPC wire version {}: this build requires {}",
            header.version,
            WIRE_VERSION
        );
    }
    if header.len as usize != actual {
        bail!(
            "IPC payload length mismatch: header={} actual={actual}",
            header.len
        );
    }
    Ok(())
}

// Diagnostic-only header byte; the authoritative kind travels in the payload.
fn request_kind_code(kind: RequestKind) -> u8 {
    match kind {
        RequestKind::GetCapabilities => 1,
        RequestKind::Execute => 2,
        RequestKind::PollCompletions => 3,
        RequestKind::DropSession => 4,
        RequestKind::Shutdown => 5,
        RequestKind::CopyKv => 6,
        RequestKind::ReleaseProducts => 7,
        RequestKind::GetMetrics => 8,
        RequestKind::GetPressure => 9,
        RequestKind::SnapshotSession => 10,
        RequestKind::RestoreSession => 11,
    }
}

fn response_kind_code(kind: ResponseKind) -> u8 {
    match kind {
        ResponseKind::Capabilities => 1,
        ResponseKind::Result => 2,
        ResponseKind::Ok => 3,
        ResponseKind::Error => 4,
        ResponseKind::Metrics => 5,
        ResponseKind::Pressure => 6,
        ResponseKind::Snapshot => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_len_matches_encoded_request() {
        let req = WorkerRequest::get_capabilities();
        let bytes = encode_request(&req).unwrap();
        let mut h = header_for_request(&req);
        h.len = payload_len_u32(bytes.len()).unwrap();
        verify_header_len(h, bytes.len()).unwrap();
        assert_eq!(h.kind, 1);
        assert_eq!(h.version, WIRE_VERSION);
    }

    #[test]
    fn request_kind_codes_cover_every_wire_kind_uniquely() {
        // Every payload kind has a distinct nonzero diagnostic code.
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for kind in RequestKind::ALL {
            let code = request_kind_code(kind);
            assert_ne!(code, 0, "request_kind_code missing for {kind:?}");
            assert!(
                seen.insert(code),
                "duplicate request_kind_code for {kind:?}"
            );
        }
    }

    #[test]
    fn payload_len_u32_rejects_oversized() {
        assert_eq!(payload_len_u32(0).unwrap(), 0);
        assert_eq!(payload_len_u32(u32::MAX as usize).unwrap(), u32::MAX);
        // Lengths that do not fit in u32 must error instead of silently
        // truncating via `as u32`.
        if (u32::MAX as usize) < usize::MAX {
            assert!(payload_len_u32(u32::MAX as usize + 1).is_err());
        }
    }

    #[test]
    fn wire_version_is_exact() {
        assert!(is_supported_wire_version(WIRE_VERSION));
        assert!(!is_supported_wire_version(WIRE_VERSION + 1));
        assert!(!is_supported_wire_version(WIRE_VERSION - 1));
    }

    #[test]
    fn verify_header_len_rejects_unsupported_version() {
        let h = Header {
            version: WIRE_VERSION + 1,
            len: 0,
            ..Default::default()
        };
        assert!(verify_header_len(h, 0).is_err());
    }

    #[test]
    fn header_is_padding_free() {
        // 3 u64 + 3 u32 + 1 u16 + 2 u8, with no implicit padding.
        assert_eq!(std::mem::size_of::<Header>(), 3 * 8 + 3 * 4 + 2 + 2);
    }
}
