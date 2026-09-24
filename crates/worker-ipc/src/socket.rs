//! Rank channel over a stream socket.
//!
//! A rank on the head's host reaches the engine through shared storage; a rank
//! on another host has no shared storage to reach, so it carries the same frames
//! over a stream socket. The frames are identical: a [`Header`] followed by its
//! encoded payload, so the codec, the protocol version and every message shape
//! are the ones the shared-storage channel uses.
//!
//! Socket readiness advances incoming and outgoing frames. Local process-death
//! notifications use a separate descriptor.
//!
//! Both ends are non-blocking and make progress only when called: receive,
//! wait and send calls flush queued writes as far as the socket accepts, so a
//! frame larger than the kernel's send buffer finishes sending across later
//! calls.

use std::collections::{HashMap, VecDeque};
use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::codec::encode_request;
use crate::iceoryx::{Frame, Header, IpcError, IpcResult, WakeEvents, header_for_request};
use crate::request::WorkerRequest;

/// Bytes of one encoded frame header, the same size and field order as the
/// `#[repr(C)]` [`Header`].
const HEADER_BYTES: usize = 16;

/// Encodes a header in a fixed little-endian layout.
///
/// The shared-storage channel hands the header across as a typed value in one
/// address space. A socket carries bytes between two, so the layout is stated
/// here rather than borrowed from the compiler's struct layout.
fn encode_header(header: &Header) -> [u8; HEADER_BYTES] {
    let mut out = [0u8; HEADER_BYTES];
    out[0..8].copy_from_slice(&header.message_id.to_le_bytes());
    out[8..12].copy_from_slice(&header.len.to_le_bytes());
    out[12..14].copy_from_slice(&header.version.to_le_bytes());
    out[14..16].copy_from_slice(&header.reserved.to_le_bytes());
    out
}

/// Decodes a header from its fixed little-endian layout.
fn decode_header(bytes: &[u8; HEADER_BYTES]) -> Header {
    Header {
        message_id: u64::from_le_bytes(std::array::from_fn(|i| bytes[i])),
        len: u32::from_le_bytes(std::array::from_fn(|i| bytes[8 + i])),
        version: u16::from_le_bytes([bytes[12], bytes[13]]),
        reserved: u16::from_le_bytes([bytes[14], bytes[15]]),
    }
}

/// Reads whole frames from a non-blocking stream without losing a partial one.
///
/// A stream delivers bytes, not messages, so a reader that abandons a partial
/// frame loses the bytes it already took. This keeps the incomplete frame and
/// resumes it on the next read.
struct FrameReader {
    /// Bytes taken from the stream and not yet formed into a frame.
    buffer: Vec<u8>,
    /// Frames completed and waiting to be taken.
    ready: VecDeque<Frame>,
}

impl FrameReader {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(64 * 1024),
            ready: VecDeque::new(),
        }
    }

    /// Takes whatever the stream has without blocking, forming whole frames.
    ///
    /// Returns whether the peer closed the connection. Fails on a read error,
    /// on a header that fails validation, and on a close that leaves a partial
    /// frame; frames completed before the failure stay in `ready`.
    fn fill(&mut self, stream: &mut TcpStream, bound: usize) -> IpcResult<bool> {
        let mut chunk = [0u8; 64 * 1024];
        let mut closed = false;
        loop {
            match stream.read(&mut chunk) {
                // A peer that answers and then exits closes the stream with
                // its last frames still unread, so the bytes already taken are
                // formed into frames before the close is reported.
                Ok(0) => {
                    closed = true;
                    break;
                }
                Ok(read) => {
                    self.buffer.extend_from_slice(&chunk[..read]);
                    self.split(bound)?;
                }
                Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => {
                    return Err(IpcError::Transport(format!(
                        "reading a rank channel: {error}"
                    )));
                }
            }
        }
        if closed && !self.buffer.is_empty() {
            return Err(IpcError::Transport(
                "rank channel closed with an incomplete frame".into(),
            ));
        }
        Ok(closed)
    }

    /// Forms every whole frame the buffer now holds.
    fn split(&mut self, bound: usize) -> IpcResult<()> {
        loop {
            if self.buffer.len() < HEADER_BYTES {
                return Ok(());
            }
            let mut head = [0u8; HEADER_BYTES];
            head.copy_from_slice(&self.buffer[..HEADER_BYTES]);
            let header = decode_header(&head);
            let len = header.len as usize;

            // Validate as soon as the header is complete, before waiting for its
            // payload, so a foreign version or an over-bound length fails
            // without buffering the bytes it announces.
            header.validate(bound)?;
            if self.buffer.len() < HEADER_BYTES + len {
                return Ok(());
            }
            let payload = self.buffer[HEADER_BYTES..HEADER_BYTES + len].to_vec();
            self.buffer.drain(..HEADER_BYTES + len);
            self.ready.push_back(Frame { header, payload });
        }
    }
}

/// Retains partial writes until the socket becomes writable. Queue capacity is
/// the channel's in-flight depth, so a stalled peer cannot grow storage use without bound.
struct FrameWriter {
    /// Encoded frames, header then payload, not yet fully written.
    pending: VecDeque<Vec<u8>>,
    /// Bytes of the front frame already written.
    offset: usize,
    /// Maximum queued frames, counting a partially written one.
    depth: usize,
}

impl FrameWriter {
    fn new(depth: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            offset: 0,
            depth: depth.max(1),
        }
    }

    /// Queues one frame after stamping and checking its header.
    ///
    /// Fails with a transport error when the header is invalid or the payload
    /// exceeds `bound`, and otherwise with [`IpcError::WouldBlock`] when
    /// `depth` frames are already queued.
    fn enqueue(&mut self, header: Header, payload: &[u8], bound: usize) -> IpcResult<()> {
        let header = header.for_payload(payload.len(), bound)?;
        if self.pending.len() >= self.depth {
            return Err(IpcError::WouldBlock);
        }
        let mut frame = Vec::with_capacity(HEADER_BYTES + payload.len());
        frame.extend_from_slice(&encode_header(&header));
        frame.extend_from_slice(payload);
        self.pending.push_back(frame);
        Ok(())
    }

    /// Writes queued bytes until the queue empties or the socket would block.
    fn flush(&mut self, stream: &mut TcpStream) -> IpcResult<()> {
        while let Some(frame) = self.pending.front() {
            match stream.write(&frame[self.offset..]) {
                Ok(0) => {
                    return Err(IpcError::Transport(
                        "rank channel closed while sending".into(),
                    ));
                }
                Ok(count) => {
                    self.offset += count;
                    if self.offset == frame.len() {
                        self.pending.pop_front();
                        self.offset = 0;
                    }
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(()),
                Err(error) => {
                    return Err(IpcError::Transport(format!(
                        "sending on a rank channel: {error}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Poll interests for the stream: always readable, and writable only while
    /// bytes are queued, since an idle writable socket would end every poll
    /// at once.
    fn events(&self) -> i16 {
        libc::POLLIN
            | if self.pending.is_empty() {
                0
            } else {
                libc::POLLOUT
            }
    }
}

/// Parks on current readiness interests, preserving the deadline across signals.
///
/// Returns `Ok(())` after any poll completes, whether it saw readiness or timed
/// out; callers re-check their own state rather than reading `revents`.
fn wait_ready(descriptors: &mut [libc::pollfd], timeout: Duration) -> IpcResult<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());

        // Round up to whole milliseconds so poll does not return before the
        // deadline.
        let millis = left.as_millis() + u128::from(!left.subsec_nanos().is_multiple_of(1_000_000));
        // SAFETY: poll borrows this live array only for the duration of the call.
        let result = unsafe {
            libc::poll(
                descriptors.as_mut_ptr(),
                descriptors.len() as libc::nfds_t,
                millis.min(i32::MAX as u128) as i32,
            )
        };
        if result >= 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != ErrorKind::Interrupted {
            return Err(IpcError::Transport(format!(
                "waiting for rank channel: {error}"
            )));
        }
    }
}

/// A local wake: one descriptor this process both fires and polls.
///
/// Process death and local worker completion can wake the owner independently
/// of incoming network traffic.
struct LocalWake {
    /// Non-blocking end the owner polls and drains.
    reader: UnixStream,
    /// Non-blocking end that senders write one byte to per coalesced wake.
    writer: UnixStream,
    /// Cleared when the reader drains, so repeated wakes coalesce.
    pending: Arc<AtomicBool>,
}

impl LocalWake {
    fn new() -> IpcResult<Self> {
        let (reader, writer) = UnixStream::pair()
            .map_err(|error| IpcError::Transport(format!("creating a local wake: {error}")))?;
        reader.set_nonblocking(true).map_err(|error| {
            IpcError::Transport(format!("making a local wake pollable: {error}"))
        })?;
        writer.set_nonblocking(true).map_err(|error| {
            IpcError::Transport(format!("making a local wake pollable: {error}"))
        })?;
        Ok(Self {
            reader,
            writer,
            pending: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Returns a sender that fires this wake from any thread.
    fn sender(&self) -> Arc<dyn Fn() + Send + Sync> {
        let writer = match self.writer.try_clone() {
            Ok(writer) => writer,
            // Without a cloned writer the returned sender does nothing, so the
            // owner observes the event only through its own re-checks after a
            // wait returns.
            Err(_) => return Arc::new(|| {}),
        };
        let pending = Arc::clone(&self.pending);
        Arc::new(move || {
            if pending.swap(true, Ordering::AcqRel) {
                return;
            }
            // A shared reference writes because a wake fires from arbitrary
            // frontend and watcher threads.
            let _ = (&writer).write(&[1]);
        })
    }

    /// Drains the descriptor and reports whether it had fired.
    fn take(&mut self) -> bool {
        // Clear the pending bit before draining, so a wake fired during the
        // drain sets it again and is reported by the next `take`.
        let fired = self.pending.swap(false, Ordering::AcqRel);
        let mut sink = [0u8; 64];
        while self.reader.read(&mut sink).is_ok_and(|read| read > 0) {}
        fired
    }
}

/// The engine's end of a socket rank channel.
pub struct SocketClient {
    stream: TcpStream,
    reader: FrameReader,
    writer: FrameWriter,
    /// Responses read from the stream before the caller asked for them.
    ///
    /// Keyed by header identity, so outstanding requests need distinct
    /// identities: a later response with the same identity replaces an
    /// untaken one.
    inbox: HashMap<u64, Frame>,
    /// Largest payload either direction may carry.
    bound: usize,
    /// Terminal IO failure, reported after complete responses have been delivered.
    failure: Option<String>,
    /// Descriptor the engine's worker-death watcher fires on.
    local: LocalWake,
}

impl SocketClient {
    /// Connects to the address a rank reported at registration.
    ///
    /// `bound` is the largest payload in bytes in either direction, `depth`
    /// the number of frames the send queue holds, and `timeout` bounds the TCP
    /// connect only.
    pub fn connect(
        address: &str,
        bound: usize,
        depth: usize,
        timeout: Duration,
    ) -> IpcResult<Self> {
        let target: SocketAddr = address.parse().map_err(|_| {
            IpcError::Transport(format!("rank channel address {address} is not an address"))
        })?;
        let stream = TcpStream::connect_timeout(&target, timeout)
            .map_err(|error| IpcError::Transport(format!("connecting to rank channel: {error}")))?;

        // Disable Nagle so a small segment is sent without waiting for earlier
        // data to be acknowledged, and make the stream non-blocking for the
        // poll-driven reader and writer.
        stream
            .set_nodelay(true)
            .map_err(|error| IpcError::Transport(format!("disabling Nagle: {error}")))?;
        stream.set_nonblocking(true).map_err(|error| {
            IpcError::Transport(format!("making a rank channel pollable: {error}"))
        })?;
        Ok(Self {
            stream,
            reader: FrameReader::new(),
            writer: FrameWriter::new(depth),
            inbox: HashMap::new(),
            bound,
            failure: None,
            local: LocalWake::new()?,
        })
    }

    /// Returns the sender a worker-death watcher fires.
    pub fn death_wake(&self) -> Arc<dyn Fn() + Send + Sync> {
        self.local.sender()
    }

    /// Returns the descriptor the engine's own wakes fire on.
    pub fn local_wake_fd(&self) -> RawFd {
        self.local.reader.as_raw_fd()
    }

    /// Encodes and sends one request, returning the identity of its response.
    ///
    /// See [`Self::send_raw`] for the queueing and failure behavior.
    pub fn send_request(&mut self, request: &WorkerRequest) -> IpcResult<u64> {
        let header = header_for_request(request);
        let payload = encode_request(request)?;
        self.send_raw(header, &payload)
    }

    /// Returns the descriptor that becomes readable when a result arrives.
    pub fn wake_file_descriptor(&self) -> RawFd {
        self.stream.as_raw_fd()
    }

    /// Sends one frame and returns the identity its response will carry.
    ///
    /// The frame is queued and written as far as the socket accepts; the rest
    /// is written by later send, receive and wait calls. Fails after a terminal
    /// channel failure, when the header is invalid or the payload exceeds the
    /// bound, with [`IpcError::WouldBlock`] when the send queue is full, and on
    /// a write error.
    pub fn send_raw(&mut self, header: Header, payload: &[u8]) -> IpcResult<u64> {
        if let Some(error) = &self.failure {
            return Err(IpcError::Transport(error.clone()));
        }

        // Flush first so frames the socket has since accepted free queue
        // capacity before the depth check.
        self.writer.flush(&mut self.stream)?;
        self.writer.enqueue(header, payload, self.bound)?;
        self.writer.flush(&mut self.stream)?;
        Ok(header.message_id)
    }

    /// Takes the response to one outstanding message, if it has arrived.
    ///
    /// A terminal channel failure is returned only once no received response
    /// remains in the inbox, so complete responses are delivered first.
    pub fn try_recv_response(&mut self, message_id: u64) -> IpcResult<Option<Frame>> {
        if let Some(frame) = self.inbox.remove(&message_id) {
            return Ok(Some(frame));
        }
        self.pump()?;
        if let Some(frame) = self.inbox.remove(&message_id) {
            return Ok(Some(frame));
        }
        if self.inbox.is_empty()
            && let Some(error) = &self.failure
        {
            return Err(IpcError::Transport(error.clone()));
        }
        Ok(None)
    }

    /// Waits for one outstanding response until the deadline passes.
    ///
    /// Returns `Ok(None)` at the deadline, and also once the channel has failed
    /// while other responses still wait in the inbox.
    pub fn recv_response_timeout(
        &mut self,
        message_id: u64,
        timeout: Duration,
    ) -> IpcResult<Option<Frame>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(frame) = self.try_recv_response(message_id)? {
                return Ok(Some(frame));
            }
            if self.failure.is_some() || Instant::now() >= deadline {
                return Ok(None);
            }
            self.wait_wake(deadline.saturating_duration_since(Instant::now()))?;
        }
    }

    /// Waits until one of this channel's wake sources fires or the deadline passes.
    pub fn wait_wake(&mut self, timeout: Duration) -> IpcResult<WakeEvents> {
        let wakes = self.drain_wakes()?;
        if wakes.any() {
            return Ok(wakes);
        }
        wait_ready(&mut self.progress_fds(), timeout)?;
        self.drain_wakes()
    }

    /// Readiness interests include writes only while an accepted frame is pending.
    ///
    /// Returns the stream and the local wake descriptor. After either polls
    /// ready, call [`Self::drain_wakes`], which also advances pending writes.
    pub fn progress_fds(&self) -> Vec<libc::pollfd> {
        vec![
            libc::pollfd {
                fd: self.stream.as_raw_fd(),
                events: self.writer.events(),
                revents: 0,
            },
            libc::pollfd {
                fd: self.local_wake_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ]
    }

    /// Reports the wakes already observed without waiting.
    ///
    /// Buffered results and peer death may be reported together; callers drain
    /// complete responses before failing the remaining work.
    pub fn drain_wakes(&mut self) -> IpcResult<WakeEvents> {
        self.pump()?;
        let local = self.local.take();
        Ok(WakeEvents {
            result: !self.inbox.is_empty(),
            death: local || self.failure.is_some(),
            other: false,
        })
    }

    /// Moves every frame the stream holds into the inbox, keyed by identity.
    ///
    /// Also flushes queued writes while the stream remains open. Transport
    /// failures are recorded in `failure` rather than returned, so this returns
    /// `Ok` in every case.
    fn pump(&mut self) -> IpcResult<()> {
        if self.failure.is_none() {
            self.failure = match self.reader.fill(&mut self.stream, self.bound) {
                Ok(false) => self
                    .writer
                    .flush(&mut self.stream)
                    .err()
                    .map(|error| error.to_string()),
                Ok(true) => Some("rank channel closed before its response".into()),
                Err(error) => Some(error.to_string()),
            };
        }
        // A final read may contain complete responses followed by a truncated
        // frame or EOF. Preserve those responses before failing outstanding work.
        while let Some(frame) = self.reader.ready.pop_front() {
            self.inbox.insert(frame.header.message_id, frame);
        }
        Ok(())
    }
}

/// A rank's end of a socket rank channel.
pub struct SocketServer {
    /// Listening socket until the engine's connection is accepted.
    listener: Option<TcpListener>,
    /// The engine's connection once accepted; the channel accepts only one.
    stream: Option<TcpStream>,
    reader: FrameReader,
    writer: FrameWriter,
    /// Largest payload either direction may carry.
    bound: usize,
    /// Terminal IO failure, reported after buffered requests have been taken.
    failure: Option<String>,
    /// The address the engine connects to, as it travels in the registration.
    address: String,
    /// Descriptor this rank's own completion callbacks fire on.
    local: LocalWake,
}

impl SocketServer {
    /// Binds the address this rank will report, without awaiting the engine.
    ///
    /// Registration reports an address that must already exist, so binding and
    /// accepting are separate: the rank reports as soon as the address is real
    /// and accepts when the engine connects.
    pub fn bind(host: &str, bound: usize, depth: usize) -> IpcResult<Self> {
        let listener = TcpListener::bind((host, 0))
            .map_err(|error| IpcError::Transport(format!("binding a rank channel: {error}")))?;
        let address = listener
            .local_addr()
            .map_err(|error| {
                IpcError::Transport(format!("reading a rank channel address: {error}"))
            })?
            .to_string();
        Ok(Self {
            listener: Some(listener),
            stream: None,
            reader: FrameReader::new(),
            writer: FrameWriter::new(depth),
            bound,
            failure: None,
            address,
            local: LocalWake::new()?,
        })
    }

    /// Returns the sender this rank's completion callbacks fire.
    ///
    /// A device, transfer or host-lane completion is raised inside the rank,
    /// so it wakes the rank's own service loop and never reaches the engine.
    pub fn completion_wake(&self) -> Arc<dyn Fn() + Send + Sync> {
        self.local.sender()
    }

    /// Returns the address the engine connects to.
    pub fn address(&self) -> &str {
        &self.address
    }

    /// Accepts the engine's connection if one is waiting, without blocking.
    ///
    /// A rank reports its address and then serves; the engine connects once,
    /// some time later. The rank notices that arrival on its next receive
    /// rather than stopping to wait for it.
    pub fn accept_if_pending(&mut self) -> IpcResult<()> {
        if self.stream.is_some() {
            return Ok(());
        }
        let Some(listener) = self.listener.as_ref() else {
            return Ok(());
        };
        listener
            .set_nonblocking(true)
            .map_err(|error| IpcError::Transport(format!("polling for a rank channel: {error}")))?;
        match listener.accept() {
            Ok((stream, _)) => {
                stream
                    .set_nodelay(true)
                    .map_err(|error| IpcError::Transport(format!("disabling Nagle: {error}")))?;
                stream.set_nonblocking(true).map_err(|error| {
                    IpcError::Transport(format!("making a rank channel pollable: {error}"))
                })?;
                self.stream = Some(stream);
                self.listener = None;
                Ok(())
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => Ok(()),
            Err(error) => Err(IpcError::Transport(format!(
                "accepting a rank channel: {error}"
            ))),
        }
    }

    /// Accepts the engine's connection, waiting up to the deadline.
    pub fn accept(&mut self, timeout: Duration) -> IpcResult<()> {
        let deadline = Instant::now() + timeout;
        loop {
            self.accept_if_pending()?;
            if self.stream.is_some() {
                return Ok(());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(IpcError::Transport(
                    "rank channel connection timed out".into(),
                ));
            }
            let listener = self
                .listener
                .as_ref()
                .ok_or_else(|| IpcError::Transport("rank channel has no listener".into()))?;
            wait_ready(
                &mut [libc::pollfd {
                    fd: listener.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                }],
                remaining,
            )?;
        }
    }

    /// Takes one request, if the engine has sent one.
    ///
    /// Returns `Ok(None)` before the engine has connected; it does not accept a
    /// waiting connection itself. When no request is already buffered and no
    /// failure is recorded, it reads the stream and, unless the read fails or
    /// finds the stream closed, flushes queued responses. A terminal failure
    /// is returned once no complete request remains buffered.
    pub fn try_recv(&mut self) -> IpcResult<Option<Frame>> {
        let Some(stream) = self.stream.as_mut() else {
            return Ok(None);
        };
        if let Some(frame) = self.reader.ready.pop_front() {
            return Ok(Some(frame));
        }
        if self.failure.is_none() {
            self.failure = match self.reader.fill(stream, self.bound) {
                Ok(false) => self
                    .writer
                    .flush(stream)
                    .err()
                    .map(|error| error.to_string()),
                Ok(true) => Some("rank channel is closed".into()),
                Err(error) => Some(error.to_string()),
            };
        }
        if let Some(frame) = self.reader.ready.pop_front() {
            return Ok(Some(frame));
        }
        if let Some(error) = &self.failure {
            return Err(IpcError::Transport(error.clone()));
        }
        Ok(None)
    }

    /// Waits until a request is readable, a completion fires, or time passes.
    ///
    /// Returns at once when a request is buffered or, failing that, when a
    /// completion wake is already pending, which it consumes. Otherwise it
    /// polls and then flushes queued responses; it never reads the stream,
    /// which the caller's next [`Self::try_recv`] does. Before the engine
    /// connects, the listener stands in for the stream.
    pub fn wait_incoming(&mut self, timeout: Duration) -> IpcResult<()> {
        self.accept_if_pending()?;
        if !self.reader.ready.is_empty() || self.local.take() {
            return Ok(());
        }

        let mut descriptors = vec![libc::pollfd {
            fd: self.local.reader.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        }];
        if let Some(stream) = &self.stream {
            descriptors.push(libc::pollfd {
                fd: stream.as_raw_fd(),
                events: self.writer.events(),
                revents: 0,
            });
        } else if let Some(listener) = &self.listener {
            descriptors.push(libc::pollfd {
                fd: listener.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            });
        }
        wait_ready(&mut descriptors, timeout)?;

        self.accept_if_pending()?;
        if let Some(stream) = &mut self.stream {
            self.writer.flush(stream)?;
        }
        Ok(())
    }

    /// Answers one request under the identity it carried.
    ///
    /// Queues the frame and writes as far as the socket accepts. Fails before
    /// the engine has connected, when the header is invalid or the payload
    /// exceeds the bound, with [`IpcError::WouldBlock`] when the send queue is
    /// full, and on a write error. Unlike the shared-storage server, it does
    /// not check `header.message_id` against received requests.
    pub fn respond_raw(&mut self, header: Header, payload: &[u8]) -> IpcResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| IpcError::Transport("rank channel is not connected".into()))?;
        self.writer.flush(stream)?;
        self.writer.enqueue(header, payload, self.bound)?;
        self.writer.flush(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connected(bound: usize, depth: usize) -> (SocketClient, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = SocketClient::connect(
            &listener.local_addr().unwrap().to_string(),
            bound,
            depth,
            Duration::from_secs(5),
        )
        .unwrap();
        let (peer, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        (client, peer)
    }

    #[test]
    fn completed_responses_survive_a_truncated_last_frame() {
        // The peer writes one complete frame a byte at a time, then a frame cut
        // short, then closes. The complete response must still be delivered;
        // only the truncated one fails, and the channel then reports death.
        let (mut client, mut peer) = connected(1024, 2);
        let header = Header {
            message_id: 1,
            len: 5,
            ..Header::default()
        };
        for byte in encode_header(&header).into_iter().chain(*b"hello") {
            peer.write_all(&[byte]).unwrap();
        }
        peer.write_all(&encode_header(&Header {
            message_id: 2,
            len: 9,
            ..Header::default()
        }))
        .unwrap();
        peer.write_all(b"partial").unwrap();
        peer.shutdown(std::net::Shutdown::Write).unwrap();

        let response = client
            .recv_response_timeout(1, Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert_eq!(response.payload, b"hello");
        let error = client
            .recv_response_timeout(2, Duration::from_secs(5))
            .err()
            .expect("truncation must fail pending work");
        assert!(error.to_string().contains("incomplete frame"), "{error}");
        assert!(client.drain_wakes().unwrap().death);
    }

    #[test]
    fn socket_rejects_incompatible_or_oversized_frames() {
        for header in [
            Header {
                version: crate::IPC_VERSION + 1,
                ..Header::default()
            },
            Header {
                len: 1025,
                ..Header::default()
            },
        ] {
            // Only the header is sent: it must be refused before any payload
            // arrives.
            let (mut client, mut peer) = connected(1024, 1);
            peer.write_all(&encode_header(&header)).unwrap();
            assert!(
                client
                    .recv_response_timeout(0, Duration::from_secs(5))
                    .is_err()
            );
        }
    }

    #[test]
    fn a_stalled_peer_applies_backpressure_and_partial_writes_resume() {
        let size = 1 << 20;
        let (mut client, mut peer) = connected(size, 1);
        let capacity: libc::c_int = 4096;
        // Restrict the real socket send buffer so this workload must span writes.
        assert_eq!(
            unsafe {
                libc::setsockopt(
                    client.stream.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_SNDBUF,
                    (&capacity as *const libc::c_int).cast(),
                    std::mem::size_of_val(&capacity) as libc::socklen_t,
                )
            },
            0
        );
        // With depth 1, the partially written first frame fills the queue.
        let payload = vec![7; size];
        client
            .send_raw(
                Header {
                    message_id: 1,
                    ..Header::default()
                },
                &payload,
            )
            .unwrap();
        assert!(matches!(
            client.send_raw(
                Header {
                    message_id: 2,
                    ..Header::default()
                },
                b"next"
            ),
            Err(IpcError::WouldBlock)
        ));

        // Once the peer reads, driving the client's wait loop resumes the
        // partial write until the whole frame arrives.
        let (tx, rx) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            let mut bytes = vec![0; HEADER_BYTES + size];
            peer.read_exact(&mut bytes).unwrap();
            tx.send(bytes).unwrap();
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        let wire = loop {
            if let Ok(bytes) = rx.try_recv() {
                break bytes;
            }
            assert!(
                Instant::now() < deadline,
                "accepted frame did not finish sending"
            );
            client.wait_wake(Duration::from_millis(10)).unwrap();
        };
        reader.join().unwrap();
        assert_eq!(&wire[HEADER_BYTES..], payload);
        assert_eq!(
            decode_header(wire[..HEADER_BYTES].try_into().unwrap()).message_id,
            1
        );
    }

    #[test]
    fn a_batch_and_its_result_cross_a_socket_unchanged() {
        // The channel's contract is that a frame arrives as it was sent, under
        // the identity it carried, so the engine can match a result to the
        // batch it answers.
        let mut server = SocketServer::bind("127.0.0.1", 1 << 20, 8).expect("rank binds");
        let address = server.address().to_string();

        let accepted = std::thread::spawn(move || {
            server
                .accept(Duration::from_secs(5))
                .expect("engine connects");
            let request = loop {
                if let Some(frame) = server.try_recv().expect("request readable") {
                    break frame;
                }
            };
            let answer = Header {
                message_id: request.header.message_id,
                ..Header::default()
            };
            server
                .respond_raw(answer, b"result-payload")
                .expect("rank answers");
            request
        });

        let mut client = SocketClient::connect(&address, 1 << 20, 8, Duration::from_secs(5))
            .expect("engine connects");
        let sent = Header {
            message_id: 77,
            ..Header::default()
        };
        let identity = client
            .send_raw(sent, b"batch-payload")
            .expect("engine sends");
        assert_eq!(identity, 77);

        let received = accepted.join().expect("rank thread");
        assert_eq!(received.payload, b"batch-payload");

        let deadline = Instant::now() + Duration::from_secs(5);
        let result = loop {
            if let Some(frame) = client.try_recv_response(77).expect("result readable") {
                break frame;
            }
            assert!(Instant::now() < deadline, "result did not arrive");
        };
        assert_eq!(result.payload, b"result-payload");
    }
}
