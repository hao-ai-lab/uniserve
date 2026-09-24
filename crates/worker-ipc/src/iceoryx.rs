//! Request-response endpoints built on iceoryx2 shared storage.
//!
//! This is the rank channel for a rank on the head's host: the engine holds a
//! [`ClientEndpoint`] and the rank a [`ServerEndpoint`], both opened on one
//! iceoryx2 request-response service named by [`service_name`]. Frames carry a
//! fixed [`Header`] and a FlatBuffers payload; the same header and payload
//! travel over the socket transport in `crate::socket`. Companion event
//! services provide blocking wakeups without polling the request rings.
//!
//! The module also owns the transport vocabulary shared by both transports:
//! [`IpcError`], [`Frame`], [`IPC_VERSION`] and header validation.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::codec::{CodecError, decode_request, decode_response, encode_request, encode_response};
use crate::{WorkerRequest, WorkerResponse};
use iceoryx2::active_request::ActiveRequest;
use iceoryx2::pending_response::PendingResponse;
use iceoryx2::port::client::Client;
use iceoryx2::port::server::Server;
use iceoryx2::prelude::*;
use iceoryx2_bb_elementary_traits::zero_copy_send::ZeroCopySend;

mod events;
use events::{ClientEvents, ServerEvents};
pub use events::{EVT_COMPLETION, EVT_DEATH, EVT_REQUEST, EVT_RESULT, WakeEvents, WakeSender};

/// Default namespace prefix for per-worker iceoryx2 services.
pub const DEFAULT_SERVICE_PREFIX: &str = "uniserve/worker";

/// Result type returned by worker transport calls.
pub type IpcResult<T> = std::result::Result<T, IpcError>;

/// Codec, transport, timeout, and protocol failures at the IPC boundary.
#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    /// Submission was not accepted because the bounded send queue is full.
    ///
    /// Only the socket transport's send queue produces this. The shared-storage
    /// transport reports an exhausted iceoryx2 request limit as `Transport`.
    #[error("rank channel send queue is full")]
    WouldBlock,
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
///
/// A wire-protocol change bumps this value. Receivers accept only this exact
/// version, so the engine and the Python worker extension must be built with
/// the same value.
pub const IPC_VERSION: u16 = 69;

/// Returns whether this build can decode a peer-advertised IPC `version`.
pub fn is_supported_ipc_version(version: u16) -> bool {
    version == IPC_VERSION
}

/// Frame correlation, size, and protocol version. Message semantics live in the payload.
/// The explicit reserved word keeps this zero-copy C layout padding-free.
///
/// Shared storage hands this struct across as a typed value, while the socket
/// transport serializes the same fields in the same order by hand
/// (`socket::encode_header`). A field change must update both; the size
/// assertion below catches a size change but not a reordering or a type
/// change of equal size.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Request-response correlation identity.
    ///
    /// Zero marks a frame without a correlation identity. The engine allocates
    /// identities from one, and when it routes a batch response it accepts a
    /// zero header identity without comparing it with the request's.
    pub message_id: u64,
    /// Encoded payload length in bytes.
    pub len: u32,
    /// Worker IPC protocol version.
    pub version: u16,
    /// Reserved word, always zero.
    pub reserved: u16,
}

impl Default for Header {
    fn default() -> Self {
        Self {
            message_id: 0,
            len: 0,
            version: IPC_VERSION,
            reserved: 0,
        }
    }
}

unsafe impl ZeroCopySend for Header {}
const _: () = assert!(std::mem::size_of::<Header>() == 16);

impl Header {
    /// Checks framing before allocating or decoding a payload on either transport.
    ///
    /// Rejects a foreign protocol version, a nonzero reserved word, and a
    /// length above `bound` bytes. It does not compare `len` with a payload.
    pub(crate) fn validate(&self, bound: usize) -> IpcResult<()> {
        if !is_supported_ipc_version(self.version) {
            ipc_bail!(
                "unsupported IPC version {}: this build requires {}",
                self.version,
                IPC_VERSION
            );
        }
        if self.reserved != 0 {
            ipc_bail!("IPC reserved header word must be zero");
        }
        if self.len as usize > bound {
            ipc_bail!(
                "rank channel frame of {} bytes exceeds its {bound} byte bound",
                self.len
            );
        }
        Ok(())
    }

    /// Stamps the actual length and enforces the common send limit.
    ///
    /// Both transports call this before sending, so the sender refuses a frame
    /// the receiver would reject under the same bound.
    pub(crate) fn for_payload(mut self, len: usize, bound: usize) -> IpcResult<Self> {
        self.len = payload_len_u32(len)?;
        self.validate(bound)?;
        Ok(self)
    }
}

/// Received frame header and owned FlatBuffers payload.
pub struct Frame {
    /// Transport header as received and validated.
    pub header: Header,
    /// Owned encoded message payload.
    pub payload: Vec<u8>,
}

impl Frame {
    /// Decodes this frame as a worker request.
    pub fn decode_request(&self) -> IpcResult<WorkerRequest> {
        verify_header_len(self.header, self.payload.len())?;
        decode_request(&self.payload).map_err(Into::into)
    }

    /// Decodes this frame as a worker response.
    pub fn decode_response(&self) -> IpcResult<WorkerResponse> {
        verify_header_len(self.header, self.payload.len())?;
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
///
/// An identifier containing `/` is taken as a complete service name; any other
/// identifier is placed under [`DEFAULT_SERVICE_PREFIX`]. The event companions
/// append their own suffixes to the service name an endpoint opens.
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
    /// Largest payload in bytes accepted in either direction.
    max_payload: usize,
    /// How long `send_raw` keeps retrying while no server is connected.
    connect_timeout: Duration,
    /// Directional wake ports paired with the request-response service.
    events: ClientEvents,
}

impl ClientEndpoint {
    /// Connects a host endpoint to a worker service, creating it if absent.
    ///
    /// `max_payload` bounds every payload in bytes. `max_inflight` is the number
    /// of [`Pending`] handles this client may hold at once; the engine passes
    /// its per-rank queue depth, which `RankProcess::finish_startup` requires
    /// to equal the worker's advertised `queue_depth`. A send while that many
    /// handles are alive fails with a transport error.
    pub fn connect(service: &str, max_payload: usize, max_inflight: usize) -> IpcResult<Self> {
        // Size the service for one client and one server, `max_inflight`
        // outstanding requests and one response per request, and disable
        // safe overflow so a full buffer never discards an unconsumed frame.
        // `ServerEndpoint::bind` requests identical settings: either side may
        // create the service through `open_or_create`, and iceoryx2 refuses to
        // open an existing service whose settings are incompatible.
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

        // Start with a slice of at most 64 KiB; power-of-two growth
        // accommodates larger, variable FlatBuffers frames.
        let client = factory
            .client_builder()
            .initial_max_slice_len(max_payload.clamp(1, 64 * 1024))
            .allocation_strategy(AllocationStrategy::PowerOfTwo)
            .create()
            .context("creating iceoryx2 client port")?;

        // Companion event ports provide file-descriptor-based wakeups.
        let events =
            ClientEvents::open(&node, service).context("opening client event companions")?;

        Ok(Self {
            _node: node,
            client,
            max_payload,
            connect_timeout: Duration::from_secs(300),
            events,
        })
    }

    /// Parks for {result, death} until a wake fires or `timeout`
    /// elapses. Returns which sources fired; none set means no wake was
    /// pending when the wait ended.
    pub fn wait_wake(&self, timeout: Duration) -> IpcResult<WakeEvents> {
        self.events.wait(timeout)
    }

    /// Drains queued wake ids after a composite executor parked on this
    /// listener's descriptor.
    pub fn drain_wakes(&self) -> IpcResult<WakeEvents> {
        self.events.drain()
    }

    /// Returns the borrowed descriptor used by an external poll loop.
    ///
    /// The descriptor stays owned by this endpoint and is valid only while it
    /// lives. After it polls readable, call [`Self::drain_wakes`].
    pub fn wake_file_descriptor(&self) -> i32 {
        self.events.file_descriptor()
    }

    /// Returns a cloneable wake source for worker-process exit.
    pub fn death_wake(&self) -> WakeSender {
        self.events.death_wake()
    }

    /// Encodes and sends a request, retrying while no server is connected.
    ///
    /// See [`Self::send_raw`] for the retry and failure behavior.
    pub fn send_request(&self, req: &WorkerRequest) -> IpcResult<Pending> {
        let payload = encode_request(req)?;
        let mut header = header_for_request(req);
        header.len = payload_len_u32(payload.len())?;
        self.send_raw(header, &payload)
    }

    /// Encodes and sends a request once, whether or not a server is connected.
    ///
    /// See [`Self::send_raw_attempt`]; the caller checks connectivity on the
    /// returned handle.
    pub fn send_request_attempt(&self, req: &WorkerRequest) -> IpcResult<Pending> {
        let payload = encode_request(req)?;
        let mut header = header_for_request(req);
        header.len = payload_len_u32(payload.len())?;
        self.send_raw_attempt(header, &payload)
    }

    /// Sends a pre-encoded frame, retrying while no server is connected.
    ///
    /// A request sent before the worker's server port exists reaches no one, so
    /// such a send is discarded and repeated every 20 ms until a server is
    /// connected or `connect_timeout` elapses. Any other send failure returns
    /// immediately. The engine uses [`Self::send_request_attempt`] and its own
    /// connectivity loop instead; this blocking form serves tests.
    pub fn send_raw(&self, header: Header, payload: &[u8]) -> IpcResult<Pending> {
        let deadline = Instant::now() + self.connect_timeout;
        loop {
            let pending = self.send_raw_attempt(header, payload)?;
            if pending.number_of_server_connections() > 0 {
                return Ok(pending);
            }

            // Release the unanswerable request's active-request slot before
            // retrying.
            drop(pending);
            if Instant::now() >= deadline {
                ipc_bail!("iceoryx2 worker service has no connected server");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Sends a pre-encoded frame once and wakes the worker's request listener.
    ///
    /// Stamps `header.len` from `payload`. Fails with a transport error when the
    /// header is invalid, the payload exceeds `max_payload`, or iceoryx2 refuses
    /// the loan or send, for example while `max_inflight` pending handles are
    /// alive. Success does not imply that a server received the request; see
    /// `Pending::number_of_server_connections`.
    pub fn send_raw_attempt(&self, header: Header, payload: &[u8]) -> IpcResult<Pending> {
        let header = header.for_payload(payload.len(), self.max_payload)?;
        // Loan exact shared-storage capacity, initialize the header and payload,
        // then transfer ownership to the request-response service.
        let mut request = self
            .client
            .loan_slice_uninit(payload.len())
            .context("loaning iceoryx2 request slice")?;
        *request.user_header_mut() = header;
        let request = request.write_from_slice(payload);
        let pending = request.send().context("sending iceoryx2 request")?;

        // Notify only after the request is in the ring.
        self.events.notify_request();
        Ok(pending)
    }

    /// Attempts to receive the response associated with `pending`.
    ///
    /// Returns `Ok(None)` when no response has arrived. The payload is copied
    /// out of shared storage, so the returned frame outlives the received
    /// sample.
    pub fn try_recv_response(&self, pending: &Pending) -> IpcResult<Option<Frame>> {
        let Some(response) = pending.receive().context("receiving iceoryx2 response")? else {
            return Ok(None);
        };
        let header = *response.user_header();
        header.validate(self.max_payload)?;
        verify_header_len(header, response.payload().len())?;
        let payload = response.payload().to_vec();
        Ok(Some(Frame { header, payload }))
    }

    /// Waits up to `timeout` for the response associated with `pending`.
    ///
    /// Returns `Ok(None)` at the deadline. Host wakes observed while waiting,
    /// including a worker-death wake, are drained without being reported, so a
    /// caller that tracks death must not rely on this method to surface it.
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
    /// Largest payload in bytes accepted in either direction.
    max_payload: usize,
    /// Active requests retained until their matching `message_id` is answered.
    active: VecDeque<(u64, IxActive)>,
    /// Directional wake ports paired with the request-response service.
    events: ServerEvents,
}

impl ServerEndpoint {
    /// Creates the worker endpoint and its directional wake services.
    ///
    /// `max_payload` and `max_inflight` size the service as in
    /// [`ClientEndpoint::connect`]; each side enforces its own `max_payload` on
    /// the frames it sends and receives.
    pub fn bind(service: &str, max_payload: usize, max_inflight: usize) -> IpcResult<Self> {
        // Mirror `ClientEndpoint::connect` exactly; iceoryx2 refuses to open a
        // service whose existing settings are incompatible with the request.
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
            .initial_max_slice_len(max_payload.clamp(1, 64 * 1024))
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
            max_payload,
            active: VecDeque::new(),
            events,
        })
    }

    /// Attempts to receive one request without blocking.
    ///
    /// A received frame's active request is retained until
    /// [`Self::respond_raw`] answers its `message_id`. A frame that fails header
    /// validation is dropped together with its active request, so no response
    /// can be sent for it.
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
        header.validate(self.max_payload)?;
        verify_header_len(header, active.payload().len())?;
        let payload = active.payload().to_vec();
        // Retain transport ownership until a response with this message id arrives.
        self.active.push_back((header.message_id, active));
        Ok(Some(Frame { header, payload }))
    }

    /// Blocks until one request arrives; there is no overall deadline.
    ///
    /// Each park is bounded by `connect_timeout`, after which the ring is
    /// checked again. Returns only with a frame or an error.
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
    ///
    /// Returns `Ok(())` on timeout as well as on a wake.
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
    ///
    /// `header.message_id` selects the oldest retained request with that
    /// identity. Answering removes the request, so a further response under
    /// that identity fails as unknown unless another retained request carries
    /// it. An oversized payload fails before any request is consumed, but once
    /// the request is resolved a failed loan or send still consumes it and no
    /// response can follow.
    pub fn respond_raw(&mut self, header: Header, payload: &[u8]) -> IpcResult<()> {
        let header = header.for_payload(payload.len(), self.max_payload)?;

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

    /// Returns the bound on one park inside [`Self::recv`].
    fn connect_timeout(&self) -> Duration {
        Duration::from_secs(300)
    }
}

/// Builds the request correlation header; send stamps the encoded length.
///
/// A request without a `message_id` gets header identity zero.
pub fn header_for_request(req: &WorkerRequest) -> Header {
    Header {
        message_id: req.message_id().unwrap_or_default(),
        ..Default::default()
    }
}

/// Builds the response correlation header; send stamps the encoded length.
///
/// A response without a `message_id` gets header identity zero.
pub fn header_for_response(resp: &WorkerResponse) -> Header {
    Header {
        message_id: resp.message_id().unwrap_or_default(),
        ..Default::default()
    }
}

/// Verifies protocol version and payload length before frame decoding.
///
/// Checks the version and reserved word and that `header.len` equals `actual`.
/// The payload bound is the receiving endpoint's check, so none is applied
/// here beyond the `u32` range.
fn verify_header_len(header: Header, actual: usize) -> IpcResult<()> {
    header.validate(u32::MAX as usize)?;
    if header.len as usize != actual {
        ipc_bail!(
            "IPC payload length mismatch: header={} actual={actual}",
            header.len
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_storage_enforces_the_payload_bound_in_both_directions() {
        // A frame one byte over the bound is refused by the sender on each
        // side, and a frame exactly at the bound round-trips.
        let service = format!("payload-bound-{}", std::process::id());
        let mut server = ServerEndpoint::bind(&service, 256, 2).unwrap();
        let client = ClientEndpoint::connect(&service, 256, 2).unwrap();

        assert!(client.send_raw(Header::default(), &[0; 257]).is_err());
        let pending = client
            .send_raw(
                Header {
                    message_id: 7,
                    ..Header::default()
                },
                &[9; 256],
            )
            .unwrap();
        let request = server.recv().unwrap();
        assert_eq!(request.payload, vec![9; 256]);

        // The oversized response is refused before it consumes the request,
        // so the in-bound response can still answer it.
        assert!(server.respond_raw(request.header, &[0; 257]).is_err());
        server.respond_raw(request.header, &[3; 256]).unwrap();
        let response = client
            .recv_response_timeout(&pending, Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert_eq!(response.payload, vec![3; 256]);
    }

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
