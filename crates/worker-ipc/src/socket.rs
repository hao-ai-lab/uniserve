//! Rank channel over a stream socket.
//!
//! A rank on the head's host reaches the engine through shared memory; a rank
//! on another host has no shared memory to reach, so it carries the same frames
//! over a stream socket. The frames are identical: a [`Header`] followed by its
//! encoded payload, so the codec, the protocol version and every message shape
//! are the ones the shared-memory channel uses.
//!
//! Only the result wake crosses the process boundary. A command or a worker
//! death is raised by a thread inside the engine, so those wakes stay local to
//! the process that raises them and never travel.

use std::collections::{HashMap, VecDeque};
use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::fd::{AsRawFd, RawFd};
use std::time::{Duration, Instant};

use crate::iceoryx::{Frame, Header, IpcError, IpcResult, WakeEvents};

/// Bytes of one encoded frame header.
const HEADER_BYTES: usize = 32;

/// Encodes a header in a fixed little-endian layout.
///
/// The shared-memory channel hands the header across as a typed value in one
/// address space. A socket carries bytes between two, so the layout is stated
/// here rather than borrowed from the compiler's struct layout.
fn encode_header(header: &Header) -> [u8; HEADER_BYTES] {
    let mut out = [0u8; HEADER_BYTES];
    out[0..8].copy_from_slice(&header.batch_id.to_le_bytes());
    out[8..16].copy_from_slice(&header.message_id.to_le_bytes());
    out[16..20].copy_from_slice(&header.len.to_le_bytes());
    out[20..24].copy_from_slice(&header.reserved0.to_le_bytes());
    out[24..28].copy_from_slice(&header.reserved1.to_le_bytes());
    out[28..30].copy_from_slice(&header.version.to_le_bytes());
    out[30] = header.kind;
    out[31] = header.flags;
    out
}

/// Decodes a header from its fixed little-endian layout.
fn decode_header(bytes: &[u8; HEADER_BYTES]) -> Header {
    Header {
        batch_id: u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")),
        message_id: u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")),
        len: u32::from_le_bytes(bytes[16..20].try_into().expect("4 bytes")),
        reserved0: u32::from_le_bytes(bytes[20..24].try_into().expect("4 bytes")),
        reserved1: u32::from_le_bytes(bytes[24..28].try_into().expect("4 bytes")),
        version: u16::from_le_bytes(bytes[28..30].try_into().expect("2 bytes")),
        kind: bytes[30],
        flags: bytes[31],
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
    /// Returns whether the peer closed the connection.
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
                Ok(read) => self.buffer.extend_from_slice(&chunk[..read]),
                Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => {
                    return Err(IpcError::Transport(format!(
                        "reading a rank channel: {error}"
                    )));
                }
            }
        }
        self.split(bound)?;
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
            if len > bound {
                return Err(IpcError::Transport(format!(
                    "rank channel frame of {len} bytes exceeds its {bound} byte bound"
                )));
            }
            if self.buffer.len() < HEADER_BYTES + len {
                return Ok(());
            }
            let payload = self.buffer[HEADER_BYTES..HEADER_BYTES + len].to_vec();
            self.buffer.drain(..HEADER_BYTES + len);
            self.ready.push_back(Frame { header, payload });
        }
    }
}

/// Writes one whole frame, blocking only for the bytes it has already begun.
fn write_frame(stream: &mut TcpStream, header: &Header, payload: &[u8]) -> IpcResult<()> {
    let mut framed = Vec::with_capacity(HEADER_BYTES + payload.len());
    let mut stamped = *header;
    stamped.len = payload.len() as u32;
    framed.extend_from_slice(&encode_header(&stamped));
    framed.extend_from_slice(payload);

    let mut written = 0;
    while written < framed.len() {
        match stream.write(&framed[written..]) {
            Ok(0) => {
                return Err(IpcError::Transport(
                    "rank channel closed while sending a frame".into(),
                ));
            }
            Ok(count) => written += count,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                // The frame is partly on the wire and must finish, so this
                // waits for capacity rather than abandoning it.
                std::thread::yield_now();
            }
            Err(error) => {
                return Err(IpcError::Transport(format!(
                    "sending on a rank channel: {error}"
                )));
            }
        }
    }
    stream
        .flush()
        .map_err(|error| IpcError::Transport(format!("flushing a rank channel: {error}")))
}

/// The engine's end of a socket rank channel.
pub struct SocketClient {
    stream: TcpStream,
    reader: FrameReader,
    /// Responses read from the stream before the caller asked for them.
    inbox: HashMap<u64, Frame>,
    /// Largest payload either direction may carry.
    bound: usize,
    /// Set when the rank closed its end.
    closed: bool,
}

impl SocketClient {
    /// Connects to the address a rank reported at registration.
    pub fn connect(address: &str, bound: usize, timeout: Duration) -> IpcResult<Self> {
        let target: SocketAddr = address.parse().map_err(|_| {
            IpcError::Transport(format!("rank channel address {address} is not an address"))
        })?;
        let stream = TcpStream::connect_timeout(&target, timeout)
            .map_err(|error| IpcError::Transport(format!("connecting to rank channel: {error}")))?;
        stream
            .set_nodelay(true)
            .map_err(|error| IpcError::Transport(format!("disabling Nagle: {error}")))?;
        stream.set_nonblocking(true).map_err(|error| {
            IpcError::Transport(format!("making a rank channel pollable: {error}"))
        })?;
        Ok(Self {
            stream,
            reader: FrameReader::new(),
            inbox: HashMap::new(),
            bound,
            closed: false,
        })
    }

    /// Returns the descriptor that becomes readable when a result arrives.
    pub fn wake_file_descriptor(&self) -> RawFd {
        self.stream.as_raw_fd()
    }

    /// Sends one frame and returns the identity its response will carry.
    pub fn send_raw(&mut self, header: Header, payload: &[u8]) -> IpcResult<u64> {
        write_frame(&mut self.stream, &header, payload)?;
        Ok(header.message_id)
    }

    /// Takes the response to one outstanding message, if it has arrived.
    pub fn try_recv_response(&mut self, message_id: u64) -> IpcResult<Option<Frame>> {
        if let Some(frame) = self.inbox.remove(&message_id) {
            return Ok(Some(frame));
        }
        self.pump()?;
        Ok(self.inbox.remove(&message_id))
    }

    /// Waits until one of this channel's wake sources fires or the deadline passes.
    pub fn wait_wake(&mut self, timeout: Duration) -> IpcResult<WakeEvents> {
        let deadline = Instant::now() + timeout;
        loop {
            self.pump()?;
            if !self.inbox.is_empty() {
                return Ok(WakeEvents {
                    result: true,
                    ..WakeEvents::default()
                });
            }
            if self.closed || Instant::now() >= deadline {
                return Ok(WakeEvents::default());
            }
            std::thread::sleep(Duration::from_micros(50).min(timeout));
        }
    }

    /// Reports the wakes already observed without waiting.
    pub fn drain_wakes(&mut self) -> IpcResult<WakeEvents> {
        self.pump()?;
        Ok(WakeEvents {
            result: !self.inbox.is_empty(),
            ..WakeEvents::default()
        })
    }

    /// Moves every frame the stream holds into the inbox, keyed by identity.
    fn pump(&mut self) -> IpcResult<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = self.reader.fill(&mut self.stream, self.bound)?;
        while let Some(frame) = self.reader.ready.pop_front() {
            self.inbox.insert(frame.header.message_id, frame);
        }
        Ok(())
    }
}

/// A rank's end of a socket rank channel.
pub struct SocketServer {
    listener: Option<TcpListener>,
    stream: Option<TcpStream>,
    reader: FrameReader,
    bound: usize,
    /// The address the engine connects to, as it travels in the registration.
    address: String,
}

impl SocketServer {
    /// Binds the address this rank will report, without awaiting the engine.
    ///
    /// Registration reports an address that must already exist, so binding and
    /// accepting are separate: the rank reports as soon as the address is real
    /// and accepts when the engine connects.
    pub fn bind(host: &str, bound: usize) -> IpcResult<Self> {
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
            bound,
            address,
        })
    }

    /// Returns the address the engine connects to.
    pub fn address(&self) -> &str {
        &self.address
    }

    /// Accepts the engine's connection, waiting up to the deadline.
    pub fn accept(&mut self, timeout: Duration) -> IpcResult<()> {
        if self.stream.is_some() {
            return Ok(());
        }
        let listener = self
            .listener
            .as_ref()
            .ok_or_else(|| IpcError::Transport("rank channel has no listener".into()))?;
        listener
            .set_nonblocking(false)
            .map_err(|error| IpcError::Transport(format!("awaiting a rank channel: {error}")))?;
        let deadline = Instant::now() + timeout;
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nodelay(true).map_err(|error| {
                        IpcError::Transport(format!("disabling Nagle: {error}"))
                    })?;
                    stream.set_nonblocking(true).map_err(|error| {
                        IpcError::Transport(format!("making a rank channel pollable: {error}"))
                    })?;
                    self.stream = Some(stream);
                    self.listener = None;
                    return Ok(());
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) if Instant::now() < deadline => {
                    return Err(IpcError::Transport(format!(
                        "accepting a rank channel: {error}"
                    )));
                }
                Err(error) => {
                    return Err(IpcError::Transport(format!(
                        "rank channel was not connected before its deadline: {error}"
                    )));
                }
            }
        }
    }

    /// Takes one request, if the engine has sent one.
    pub fn try_recv(&mut self) -> IpcResult<Option<Frame>> {
        let Some(stream) = self.stream.as_mut() else {
            return Ok(None);
        };
        if let Some(frame) = self.reader.ready.pop_front() {
            return Ok(Some(frame));
        }
        self.reader.fill(stream, self.bound)?;
        Ok(self.reader.ready.pop_front())
    }

    /// Waits until a request is readable or the deadline passes.
    pub fn wait_incoming(&mut self, timeout: Duration) -> IpcResult<()> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if !self.reader.ready.is_empty() {
                return Ok(());
            }
            if let Some(stream) = self.stream.as_mut() {
                self.reader.fill(stream, self.bound)?;
                if !self.reader.ready.is_empty() {
                    return Ok(());
                }
            }
            std::thread::sleep(Duration::from_micros(50));
        }
        Ok(())
    }

    /// Answers one request under the identity it carried.
    pub fn respond_raw(&mut self, header: Header, payload: &[u8]) -> IpcResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| IpcError::Transport("rank channel is not connected".into()))?;
        write_frame(stream, &header, payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_header_survives_its_encoding() {
        let header = Header {
            batch_id: 9_876_543_210,
            message_id: 1_234_567_890,
            len: 4096,
            reserved0: 0,
            reserved1: 0,
            version: 61,
            kind: 2,
            flags: 3,
        };
        assert_eq!(decode_header(&encode_header(&header)), header);
    }

    #[test]
    fn a_reader_forms_frames_from_arbitrary_byte_boundaries() {
        // A stream may deliver a frame in any number of pieces, so the reader
        // is fed one byte at a time and must still produce both frames whole.
        let mut wire = Vec::new();
        for (id, payload) in [(1u64, vec![7u8; 5]), (2, vec![9u8; 3])] {
            let header = Header {
                message_id: id,
                len: payload.len() as u32,
                ..Header::default()
            };
            wire.extend_from_slice(&encode_header(&header));
            wire.extend_from_slice(&payload);
        }

        let mut reader = FrameReader::new();
        for byte in wire {
            reader.buffer.push(byte);
            reader.split(1024).expect("frames split");
        }

        let first = reader.ready.pop_front().expect("first frame");
        let second = reader.ready.pop_front().expect("second frame");
        assert_eq!((first.header.message_id, first.payload), (1, vec![7u8; 5]));
        assert_eq!(
            (second.header.message_id, second.payload),
            (2, vec![9u8; 3])
        );
        assert!(reader.ready.is_empty());
    }

    #[test]
    fn a_batch_and_its_result_cross_a_socket_unchanged() {
        // The channel's contract is that a frame arrives as it was sent, under
        // the identity it carried, so the engine can match a result to the
        // batch it answers.
        let mut server = SocketServer::bind("127.0.0.1", 1 << 20).expect("rank binds");
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
                batch_id: request.header.batch_id,
                kind: 9,
                ..Header::default()
            };
            server
                .respond_raw(answer, b"result-payload")
                .expect("rank answers");
            request
        });

        let mut client = SocketClient::connect(&address, 1 << 20, Duration::from_secs(5))
            .expect("engine connects");
        let sent = Header {
            message_id: 77,
            batch_id: 4321,
            kind: 2,
            ..Header::default()
        };
        let identity = client
            .send_raw(sent, b"batch-payload")
            .expect("engine sends");
        assert_eq!(identity, 77);

        let received = accepted.join().expect("rank thread");
        assert_eq!(received.payload, b"batch-payload");
        assert_eq!(received.header.batch_id, 4321);
        assert_eq!(received.header.kind, 2);

        let deadline = Instant::now() + Duration::from_secs(5);
        let result = loop {
            if let Some(frame) = client.try_recv_response(77).expect("result readable") {
                break frame;
            }
            assert!(Instant::now() < deadline, "result did not arrive");
        };
        assert_eq!(result.payload, b"result-payload");
        assert_eq!(result.header.batch_id, 4321);
    }

    #[test]
    fn a_frame_above_the_bound_is_refused_by_size() {
        let header = Header {
            len: 8192,
            ..Header::default()
        };
        let mut reader = FrameReader::new();
        reader.buffer.extend_from_slice(&encode_header(&header));
        let error = reader.split(4096).expect_err("oversized frame is refused");
        assert!(
            error.to_string().contains("exceeds its 4096 byte bound"),
            "{error}"
        );
    }
}
