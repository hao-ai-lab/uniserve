//! Request-response endpoints built on iceoryx2 shared memory.
//!
//! Frames carry a fixed header and a FlatBuffers payload. Companion event
//! services provide blocking wakeups without polling the request rings.

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

/// Default namespace prefix for per-worker iceoryx2 services.
pub const DEFAULT_SERVICE_PREFIX: &str = "uniserve/worker";

/// Result type returned by worker transport calls.
pub type IpcResult<T> = std::result::Result<T, IpcError>;

/// Codec, transport, timeout, and protocol failures at the IPC boundary.
#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    /// Payload encoding, decoding, or semantic validation failed.
    #[error(transparent)]
    Codec(#[from] CodecError),
    /// A rank channel's setup or one of its calls failed.
    #[error("worker IPC error: {0}")]
    Transport(String),
}

impl IpcError {
    /// Constructs a transport error with contextual text.
    pub(crate) fn transport(message: impl Into<String>) -> Self {
        Self::Transport(message.into())
    }
}

/// Extension methods for attaching transport context to fallible calls.
trait IpcContext<T> {
    /// Replaces a missing value or source error with fixed transport context.
    fn context(self, message: &str) -> IpcResult<T>;

    /// Adds lazily constructed transport context to a missing value or error.
    fn with_context<F, D>(self, message: F) -> IpcResult<T>
    where
        F: FnOnce() -> D,
        D: std::fmt::Display;
}

impl<T, E> IpcContext<T> for std::result::Result<T, E>
where
    E: std::fmt::Debug,
{
    /// Wraps a source error with fixed transport context.
    fn context(self, message: &str) -> IpcResult<T> {
        self.map_err(|error| IpcError::transport(format!("{message}: {error:?}")))
    }

    /// Wraps a source error with lazily constructed transport context.
    fn with_context<F, D>(self, message: F) -> IpcResult<T>
    where
        F: FnOnce() -> D,
        D: std::fmt::Display,
    {
        self.map_err(|error| IpcError::transport(format!("{}: {error:?}", message())))
    }
}

impl<T> IpcContext<T> for Option<T> {
    /// Converts absence into a transport error with fixed context.
    fn context(self, message: &str) -> IpcResult<T> {
        self.ok_or_else(|| IpcError::transport(message))
    }

    /// Converts absence into a transport error with lazily constructed context.
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

/// IPC version this build emits on every [`Header`].
pub const IPC_VERSION: u16 = 67;

/// Returns whether this build can decode a peer-advertised IPC `version`.
pub fn is_supported_ipc_version(version: u16) -> bool {
    version == IPC_VERSION
}

/// Fixed-size IPC frame header sent zero-copy across the worker process
/// boundary.
///
/// Fields are ordered largest-alignment-first so the `#[repr(C)]` layout is
/// completely free of implicit padding. The header is shipped via
/// [`ZeroCopySend`], so padding would copy uninitialized bytes across the
/// process boundary and make logically equal headers bytewise unstable. The
/// layout is exactly
/// `2*u64 + 3*u32 + u16 + 2*u8 = 32` bytes with no holes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Batch identity, or zero for frames that carry no batch.
    pub batch_id: u64,
    /// Request-response correlation identity.
    pub message_id: u64,
    /// Encoded payload length in bytes.
    pub len: u32,
    /// Reserved protocol word, emitted as zero.
    pub reserved0: u32,
    /// Reserved protocol word, emitted as zero.
    pub reserved1: u32,
    /// Worker IPC protocol version.
    pub version: u16,
    /// Request or response kind discriminator.
    pub kind: u8,
    /// Kind-specific protocol flags.
    pub flags: u8,
}

impl Default for Header {
    /// Returns an empty header stamped with the emitted IPC version.
    fn default() -> Self {
        Self {
            batch_id: 0,
            message_id: 0,
            len: 0,
            reserved0: 0,
            reserved1: 0,
            version: IPC_VERSION,
            kind: 0,
            flags: 0,
        }
    }
}

unsafe impl ZeroCopySend for Header {}

// This assertion binds the C representation to the sum of its field sizes so
// zero-copy transmission cannot include implicit padding.
const _: () = assert!(
    std::mem::size_of::<Header>() == 2 * 8 + 3 * 4 + 2 + 2,
    "Header must be padding-free for zero-copy transmission"
);

/// Received frame header and owned FlatBuffers payload.
pub struct Frame {
    /// Zero-copy transport header.
    pub header: Header,
    /// Owned encoded message payload.
    pub payload: Vec<u8>,
}

impl Frame {
    /// Decodes this frame as a worker request.
    pub fn decode_request(&self) -> IpcResult<WorkerRequest> {
        decode_request(&self.payload).map_err(Into::into)
    }

    /// Decodes this frame as a worker response.
    pub fn decode_response(&self) -> IpcResult<WorkerResponse> {
        decode_response(&self.payload).map_err(Into::into)
    }
}

/// Thread-safe iceoryx2 IPC service type used by endpoint aliases.
pub(crate) type IxService = iceoryx2::service::ipc_threadsafe::Service;
/// Concrete iceoryx2 client port used by the host endpoint.
type IxClient = Client<IxService, [u8], Header, [u8], Header>;
/// Concrete iceoryx2 server port used by the worker endpoint.
type IxServer = Server<IxService, [u8], Header, [u8], Header>;
/// Active request handle retained until the worker publishes its response.
type IxActive = ActiveRequest<IxService, [u8], Header, [u8], Header>;
/// Pending response handle returned after a request is sent.
pub type Pending = PendingResponse<IxService, [u8], Header, [u8], Header>;

/// Converts an encoded payload length to the checked `u32` in [`Header::len`].
///
/// Truncation would make the receiver interpret a different frame boundary.
fn payload_len_u32(len: usize) -> IpcResult<u32> {
    u32::try_from(len).map_err(|_| {
        ipc_error!(
            "IPC payload of {len} bytes exceeds the maximum frame size of {} bytes",
            u32::MAX
        )
    })
}

/// Builds the iceoryx2 service name for one worker identifier.
pub fn service_name(id: &str) -> String {
    if id.contains('/') {
        id.to_string()
    } else {
        format!("{DEFAULT_SERVICE_PREFIX}/{id}")
    }
}

/// Host-side endpoint for sending requests and receiving worker responses.
pub struct ClientEndpoint {
    /// Node owner that keeps all client ports alive.
    _node: Node<IxService>,
    /// Request-response port used to loan and send request frames.
    client: IxClient,
    /// Deadline for establishing or awaiting worker connectivity.
    connect_timeout: Duration,
    /// Directional wake ports paired with the request-response service.
    events: ClientEvents,
}

impl ClientEndpoint {
    /// Connects a host endpoint to an existing worker service.
    pub fn connect(
        service: &str,
        initial_max_slice_len: usize,
        max_inflight: usize,
    ) -> IpcResult<Self> {
        // Create the request-response service with bounded, non-overflowing
        // capacity so every pending handle names one retained request.
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
        // Power-of-two growth accommodates variable FlatBuffers frame sizes.
        let client = factory
            .client_builder()
            .initial_max_slice_len(initial_max_slice_len.max(1))
            .allocation_strategy(AllocationStrategy::PowerOfTwo)
            .create()
            .context("creating iceoryx2 client port")?;
        // Companion event ports provide file-descriptor-based wakeups.
        let events =
            ClientEvents::open(&node, service).context("opening client event companions")?;
        Ok(Self {
            _node: node,
            client,
            connect_timeout: Duration::from_secs(300),
            events,
        })
    }

    /// Parks for {result, command, death} until a wake fires or `timeout`
    /// elapses. Returns which sources fired.
    pub fn wait_wake(&self, timeout: Duration) -> IpcResult<WakeEvents> {
        self.events.wait(timeout)
    }

    /// Drains queued wake ids after a composite executor parked on this
    /// listener's descriptor.
    pub fn drain_wakes(&self) -> IpcResult<WakeEvents> {
        self.events.drain()
    }

    /// Returns the borrowed descriptor used by an external poll loop.
    pub fn wake_file_descriptor(&self) -> i32 {
        self.events.file_descriptor()
    }

    /// Returns a cloneable wake source for queued host commands.
    pub fn command_wake(&self) -> WakeSender {
        self.events.command_wake()
    }

    /// Returns a cloneable wake source for worker-process exit.
    pub fn death_wake(&self) -> WakeSender {
        self.events.death_wake()
    }

    /// Encodes and sends a request, waiting for ring capacity until the deadline.
    pub fn send_request(&self, req: &WorkerRequest) -> IpcResult<Pending> {
        let payload = encode_request(req)?;
        let mut header = header_for_request(req);
        header.len = payload_len_u32(payload.len())?;
        self.send_raw(header, &payload)
    }

    /// Attempts one non-blocking request submission.
    pub fn send_request_attempt(&self, req: &WorkerRequest) -> IpcResult<Pending> {
        let payload = encode_request(req)?;
        let mut header = header_for_request(req);
        header.len = payload_len_u32(payload.len())?;
        self.send_raw_attempt(header, &payload)
    }

    /// Sends a pre-encoded frame, waiting for ring capacity until the deadline.
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

    /// Attempts one non-blocking pre-encoded frame submission.
    pub fn send_raw_attempt(&self, header: Header, payload: &[u8]) -> IpcResult<Pending> {
        // Loan exact shared-memory capacity, initialize the header and payload,
        // then transfer ownership to the request-response service.
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

    /// Attempts to receive the response associated with `pending`.
    pub fn try_recv_response(&self, pending: &Pending) -> IpcResult<Option<Frame>> {
        let Some(response) = pending.receive().context("receiving iceoryx2 response")? else {
            return Ok(None);
        };
        let header = *response.user_header();
        let payload = response.payload().to_vec();
        verify_header_len(header, payload.len())?;
        Ok(Some(Frame { header, payload }))
    }

    /// Waits up to `timeout` for the response associated with `pending`.
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

/// Worker-side endpoint for receiving requests and publishing responses.
pub struct ServerEndpoint {
    /// Node owner that keeps all server ports alive.
    _node: Node<IxService>,
    /// Request-response port used to receive and answer request frames.
    server: IxServer,
    /// Active requests retained until their matching `message_id` is answered.
    active: VecDeque<(u64, IxActive)>,
    /// Directional wake ports paired with the request-response service.
    events: ServerEvents,
}

impl ServerEndpoint {
    /// Creates the worker endpoint and its directional wake services.
    pub fn bind(
        service: &str,
        initial_max_slice_len: usize,
        max_inflight: usize,
    ) -> IpcResult<Self> {
        // Mirror client capacity on a single-server, single-client service and
        // disable overflow so request ownership remains explicit.
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
        // Allocate response buffers with the same variable-size strategy.
        let server = factory
            .server_builder()
            .initial_max_slice_len(initial_max_slice_len.max(1))
            .allocation_strategy(AllocationStrategy::PowerOfTwo)
            .max_loaned_responses_per_request(1)
            .create()
            .context("creating iceoryx2 server port")?;
        // Companion event ports wake request and asynchronous-completion paths.
        let events =
            ServerEvents::open(&node, service).context("opening server event companions")?;
        Ok(Self {
            _node: node,
            server,
            active: VecDeque::new(),
            events,
        })
    }

    /// Attempts to receive one request without blocking.
    pub fn try_recv(&mut self) -> IpcResult<Option<Frame>> {
        let Some(active) = self
            .server
            .receive()
            .context("receiving iceoryx2 request")?
        else {
            return Ok(None);
        };
        // Receiving from the ring consumes the work represented by wake hints.
        self.events.drain_worker_wakes()?;
        let header = *active.user_header();
        let payload = active.payload().to_vec();
        verify_header_len(header, payload.len())?;
        // Retain transport ownership until a response with this message id arrives.
        self.active.push_back((header.message_id, active));
        Ok(Some(Frame { header, payload }))
    }

    /// Waits for and receives one request before the call deadline.
    pub fn recv(&mut self) -> IpcResult<Frame> {
        loop {
            if let Some(frame) = self.try_recv()? {
                return Ok(frame);
            }
            self.wait_incoming(self.connect_timeout())?;
        }
    }

    /// Parks until an inbound request or asynchronous completion wake fires, or
    /// `timeout` elapses. The caller re-checks all progress sources after return.
    pub fn wait_incoming(&self, timeout: Duration) -> IpcResult<()> {
        self.events.wait_request(timeout)
    }

    /// Returns a thread-safe signal for asynchronous worker completion.
    pub fn completion_wake(&self) -> WakeSender {
        self.events.completion_wake()
    }

    /// Encodes and publishes a response for the active request.
    pub fn respond(&mut self, resp: &WorkerResponse) -> IpcResult<()> {
        let payload = encode_response(resp)?;
        let mut header = header_for_response(resp);
        header.len = payload_len_u32(payload.len())?;
        self.respond_raw(header, &payload)
    }

    /// Publishes a pre-encoded response for the active request.
    pub fn respond_raw(&mut self, header: Header, payload: &[u8]) -> IpcResult<()> {
        // Resolve and remove the exact active transport request before loaning
        // its response buffer.
        let pos = self
            .active
            .iter()
            .position(|(message_id, _)| *message_id == header.message_id)
            .with_context(|| {
                format!(
                    "respond called for unknown active request message_id {}",
                    header.message_id
                )
            })?;
        let active = self
            .active
            .remove(pos)
            .map(|(_, active)| active)
            .context("active request position disappeared")?;
        // Initialize the loaned response completely before publishing it.
        let mut response = active
            .loan_slice_uninit(payload.len())
            .context("loaning iceoryx2 response slice")?;
        *response.user_header_mut() = header;
        let response = response.write_from_slice(payload);
        response.send().context("sending iceoryx2 response")?;
        // Notify only after the ring owns the fully initialized response.
        self.events.notify_response();
        Ok(())
    }

    /// Returns the worker-side deadline used by blocking receive calls.
    fn connect_timeout(&self) -> Duration {
        Duration::from_secs(300)
    }
}

/// Builds the IPC header that accompanies `req`.
///
/// `message_id` is the authoritative request/response correlation key. The
/// `batch_id` field is a diagnostic hint populated from the batch; the decoded
/// payload is authoritative for the call identities it carries.
pub fn header_for_request(req: &WorkerRequest) -> Header {
    let mut h = Header {
        kind: request_kind_code(req.kind()),
        message_id: req.message_id().unwrap_or_default(),
        ..Default::default()
    };
    if let Some(batch) = req.batch() {
        h.batch_id = batch.batch_id;
    }
    h
}

/// Builds the IPC header that accompanies `resp`.
///
/// `message_id` is the authoritative request/response correlation key. The
/// `batch_id` field is a diagnostic hint populated from the result; the decoded
/// payload is authoritative for the call identities it carries.
pub fn header_for_response(resp: &WorkerResponse) -> Header {
    let mut h = Header {
        kind: response_kind_code(resp.kind()),
        message_id: resp.message_id().unwrap_or_default(),
        ..Default::default()
    };
    if let Some(report) = resp.report() {
        h.batch_id = report.batch_id;
    }
    h
}

/// Verifies protocol version and payload length before frame decoding.
fn verify_header_len(header: Header, actual: usize) -> IpcResult<()> {
    if !is_supported_ipc_version(header.version) {
        ipc_bail!(
            "unsupported IPC IPC version {}: this build requires {}",
            header.version,
            IPC_VERSION
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

/// Maps a request kind to its diagnostic header byte.
///
/// The FlatBuffers payload remains authoritative for request dispatch.
fn request_kind_code(kind: RequestKind) -> u8 {
    match kind {
        RequestKind::Info => 1,
        RequestKind::Submit => 2,
        RequestKind::Close => 3,
    }
}

/// Maps a response kind to its diagnostic header byte.
///
/// The FlatBuffers payload remains authoritative for response dispatch.
fn response_kind_code(kind: ResponseKind) -> u8 {
    match kind {
        ResponseKind::Info => 1,
        ResponseKind::Result => 2,
        ResponseKind::Ok => 3,
        ResponseKind::Error => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_len_matches_encoded_request() {
        let req = WorkerRequest::info();
        let bytes = encode_request(&req).unwrap();
        let mut h = header_for_request(&req);
        h.len = payload_len_u32(bytes.len()).unwrap();
        verify_header_len(h, bytes.len()).unwrap();
        assert_eq!(h.version, IPC_VERSION);
    }

    #[test]
    fn payload_len_u32_rejects_oversized() {
        assert_eq!(payload_len_u32(0).unwrap(), 0);
        assert_eq!(payload_len_u32(u32::MAX as usize).unwrap(), u32::MAX);
        // Header lengths are exact `u32` values; oversized payloads are invalid.
        if (u32::MAX as usize) < usize::MAX {
            assert!(payload_len_u32(u32::MAX as usize + 1).is_err());
        }
    }
}
