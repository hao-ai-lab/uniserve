//! Shared IPC primitives for the iceoryx2 worker request-response boundary.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::codec::{CodecError, decode_request, decode_response, encode_request, encode_response};
use crate::{RequestKind, ResponseKind, WorkerRequest, WorkerResponse};
use iceoryx2::active_request::ActiveRequest;
use iceoryx2::pending_response::PendingResponse;
use iceoryx2::port::client::Client;
use iceoryx2::port::server::Server;
use iceoryx2::prelude::*;
use iceoryx2_bb_elementary_traits::zero_copy_send::ZeroCopySend;

mod events;
use events::{ClientEvents, ServerEvents};
pub use events::{
    EVT_COMMAND, EVT_COMPLETION, EVT_DEATH, EVT_REQUEST, EVT_RESULT, WakeEvents, WakeSender,
};

pub const DEFAULT_SERVICE_PREFIX: &str = "uniserve/worker";

pub type IpcResult<T> = std::result::Result<T, IpcError>;

#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    #[error(transparent)]
    Codec(#[from] CodecError),
    #[error("worker IPC error: {0}")]
    Transport(String),
}

impl IpcError {
    pub(crate) fn transport(message: impl Into<String>) -> Self {
        Self::Transport(message.into())
    }
}

trait IpcContext<T> {
    fn context(self, message: &str) -> IpcResult<T>;
    fn with_context<F, D>(self, message: F) -> IpcResult<T>
    where
        F: FnOnce() -> D,
        D: std::fmt::Display;
}

impl<T, E> IpcContext<T> for std::result::Result<T, E>
where
    E: std::fmt::Debug,
{
    fn context(self, message: &str) -> IpcResult<T> {
        self.map_err(|error| IpcError::transport(format!("{message}: {error:?}")))
    }

    fn with_context<F, D>(self, message: F) -> IpcResult<T>
    where
        F: FnOnce() -> D,
        D: std::fmt::Display,
    {
        self.map_err(|error| IpcError::transport(format!("{}: {error:?}", message())))
    }
}

impl<T> IpcContext<T> for Option<T> {
    fn context(self, message: &str) -> IpcResult<T> {
        self.ok_or_else(|| IpcError::transport(message))
    }

    fn with_context<F, D>(self, message: F) -> IpcResult<T>
    where
        F: FnOnce() -> D,
        D: std::fmt::Display,
    {
        self.ok_or_else(|| IpcError::transport(message().to_string()))
    }
}

macro_rules! ipc_bail {
    ($($arg:tt)*) => {
        return Err(IpcError::transport(format!($($arg)*)))
    };
}

macro_rules! ipc_error {
    ($($arg:tt)*) => {
        IpcError::transport(format!($($arg)*))
    };
}

/// Wire protocol version this build emits on every [`Header`].
pub const WIRE_VERSION: u16 = 10;

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
    pub fn decode_request(&self) -> IpcResult<WorkerRequest> {
        decode_request(&self.payload).map_err(Into::into)
    }

    pub fn decode_response(&self) -> IpcResult<WorkerResponse> {
        decode_response(&self.payload).map_err(Into::into)
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
fn payload_len_u32(len: usize) -> IpcResult<u32> {
    u32::try_from(len).map_err(|_| {
        ipc_error!(
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
    events: ClientEvents,
}

impl ClientEndpoint {
    pub fn connect(
        service: &str,
        initial_max_slice_len: usize,
        max_inflight: usize,
    ) -> IpcResult<Self> {
        let service_name = ServiceName::new(service).context("invalid iceoryx2 service name")?;
        let node = NodeBuilder::new()
            .create::<IxService>()
            .context("creating iceoryx2 client node")?;
        let factory = node
            .service_builder(&service_name)
            .request_response::<[u8], [u8]>()
            .request_user_header::<Header>()
            .response_user_header::<Header>()
            .max_active_requests_per_client(max_inflight.max(1))
            .max_response_buffer_size(1)
            .max_loaned_requests(max_inflight.max(1))
            .max_servers(1)
            .max_clients(1)
            .max_borrowed_responses_per_pending_response(1)
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
        let events =
            ClientEvents::open(&node, service).context("opening client event companions")?;
        Ok(Self {
            _node: node,
            client,
            connect_timeout: Duration::from_secs(300),
            events,
        })
    }

    /// Park for {result, command, death} until a wake fires or `timeout`
    /// elapses. Returns which sources fired.
    pub fn wait_wake(&self, timeout: Duration) -> IpcResult<WakeEvents> {
        self.events.wait(timeout)
    }

    /// Drain queued wake ids after a composite executor parked on this
    /// listener's descriptor.
    pub fn drain_wakes(&self) -> IpcResult<WakeEvents> {
        self.events.drain()
    }

    /// Native descriptor used by a composite executor's single multi-worker
    /// park. It remains owned by this endpoint.
    pub fn wake_file_descriptor(&self) -> i32 {
        self.events.file_descriptor()
    }

    /// A cloneable wake source the command ingress fires after enqueuing a command.
    pub fn command_wake(&self) -> WakeSender {
        self.events.command_wake()
    }

    /// A cloneable wake source the worker-death watcher fires on child exit.
    pub fn death_wake(&self) -> WakeSender {
        self.events.death_wake()
    }

    pub fn send_request(&self, req: &WorkerRequest) -> IpcResult<Pending> {
        let payload = encode_request(req)?;
        let mut header = header_for_request(req);
        header.len = payload_len_u32(payload.len())?;
        self.send_raw(header, &payload)
    }

    pub fn send_request_attempt(&self, req: &WorkerRequest) -> IpcResult<Pending> {
        let payload = encode_request(req)?;
        let mut header = header_for_request(req);
        header.len = payload_len_u32(payload.len())?;
        self.send_raw_attempt(header, &payload)
    }

    pub fn send_raw(&self, header: Header, payload: &[u8]) -> IpcResult<Pending> {
        let deadline = Instant::now() + self.connect_timeout;
        loop {
            let pending = self.send_raw_attempt(header, payload)?;
            if pending.number_of_server_connections() > 0 {
                return Ok(pending);
            }
            drop(pending);
            if Instant::now() >= deadline {
                ipc_bail!("iceoryx2 worker service has no connected server");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    pub fn send_raw_attempt(&self, header: Header, payload: &[u8]) -> IpcResult<Pending> {
        let mut request = self
            .client
            .loan_slice_uninit(payload.len())
            .context("loaning iceoryx2 request slice")?;
        *request.user_header_mut() = header;
        let request = request.write_from_slice(payload);
        let pending = request.send().context("sending iceoryx2 request")?;
        self.events.notify_request();
        Ok(pending)
    }

    pub fn try_recv_response(&self, pending: &Pending) -> IpcResult<Option<Frame>> {
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
    ) -> IpcResult<Option<Frame>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(frame) = self.try_recv_response(pending)? {
                return Ok(Some(frame));
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            self.events.wait(deadline - now)?;
        }
    }
}

pub struct ServerEndpoint {
    _node: Node<IxService>,
    server: IxServer,
    active: VecDeque<(u64, IxActive)>,
    events: ServerEvents,
}

impl ServerEndpoint {
    pub fn bind(
        service: &str,
        initial_max_slice_len: usize,
        max_inflight: usize,
    ) -> IpcResult<Self> {
        let service_name = ServiceName::new(service).context("invalid iceoryx2 service name")?;
        let node = NodeBuilder::new()
            .create::<IxService>()
            .context("creating iceoryx2 server node")?;
        let factory = node
            .service_builder(&service_name)
            .request_response::<[u8], [u8]>()
            .request_user_header::<Header>()
            .response_user_header::<Header>()
            .max_active_requests_per_client(max_inflight.max(1))
            .max_response_buffer_size(1)
            .max_loaned_requests(max_inflight.max(1))
            .max_servers(1)
            .max_clients(1)
            .max_borrowed_responses_per_pending_response(1)
            .enable_safe_overflow_for_requests(false)
            .enable_safe_overflow_for_responses(false)
            .open_or_create()
            .context("opening iceoryx2 request-response service")?;
        let server = factory
            .server_builder()
            .initial_max_slice_len(initial_max_slice_len.max(1))
            .allocation_strategy(AllocationStrategy::PowerOfTwo)
            .max_loaned_responses_per_request(1)
            .create()
            .context("creating iceoryx2 server port")?;
        let events =
            ServerEvents::open(&node, service).context("opening server event companions")?;
        Ok(Self {
            _node: node,
            server,
            active: VecDeque::new(),
            events,
        })
    }

    pub fn try_recv(&mut self) -> IpcResult<Option<Frame>> {
        let Some(active) = self
            .server
            .receive()
            .context("receiving iceoryx2 request")?
        else {
            return Ok(None);
        };
        self.events.drain_worker_wakes()?;
        let header = *active.user_header();
        let payload = active.payload().to_vec();
        verify_header_len(header, payload.len())?;
        self.active.push_back((header.call_id, active));
        Ok(Some(Frame { header, payload }))
    }

    pub fn recv(&mut self) -> IpcResult<Frame> {
        loop {
            if let Some(frame) = self.try_recv()? {
                return Ok(frame);
            }
            self.wait_incoming(self.connect_timeout())?;
        }
    }

    /// Park until an inbound request or asynchronous completion wake fires, or
    /// `timeout` elapses. The caller re-checks all progress sources after return.
    pub fn wait_incoming(&self, timeout: Duration) -> IpcResult<()> {
        self.events.wait_request(timeout)
    }

    /// A thread-safe signal for device, transfer, and CPU completion callbacks.
    pub fn completion_wake(&self) -> WakeSender {
        self.events.completion_wake()
    }

    pub fn respond(&mut self, resp: &WorkerResponse) -> IpcResult<()> {
        let payload = encode_response(resp)?;
        let mut header = header_for_response(resp);
        header.len = payload_len_u32(payload.len())?;
        self.respond_raw(header, &payload)
    }

    pub fn respond_raw(&mut self, header: Header, payload: &[u8]) -> IpcResult<()> {
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
        // Wake the host the instant the response is queued.
        self.events.notify_response();
        Ok(())
    }

    fn connect_timeout(&self) -> Duration {
        Duration::from_secs(300)
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
        kind: request_kind_code(req.kind()),
        call_id: req.call_id().unwrap_or_default(),
        ..Default::default()
    };
    if let Some(batch) = req.batch() {
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
        kind: response_kind_code(resp.kind()),
        call_id: resp.call_id().unwrap_or_default(),
        ..Default::default()
    };
    if let Some(report) = resp.report() {
        h.step_id = report.step_id;
        // Hint only: first completion's op id. See the doc comment above.
        if let Some(completion) = report.completions().next() {
            h.op_id = completion.op_id.0;
        }
    }
    h
}

fn verify_header_len(header: Header, actual: usize) -> IpcResult<()> {
    if !is_supported_wire_version(header.version) {
        ipc_bail!(
            "unsupported IPC wire version {}: this build requires {}",
            header.version,
            WIRE_VERSION
        );
    }
    if header.len as usize != actual {
        ipc_bail!(
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
        RequestKind::ReleaseProducts => 7,
        RequestKind::GetPressure => 8,
    }
}

fn response_kind_code(kind: ResponseKind) -> u8 {
    match kind {
        ResponseKind::Capabilities => 1,
        ResponseKind::Result => 2,
        ResponseKind::Ok => 3,
        ResponseKind::Error => 4,
        ResponseKind::Pressure => 5,
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
