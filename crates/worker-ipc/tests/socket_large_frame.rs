//! A frame far larger than any kernel socket buffer crosses a loopback TCP
//! connection between two processes and arrives byte for byte.
//!
//! The rank channel of a worker on another host carries the engine's batches
//! and the worker's results over a stream socket, so a payload larger than
//! one send buffer must finish across later calls and reassemble exactly. The
//! test binary runs a second time as the worker side: the parent connects to
//! the child's server, sends the payload, and requires the child's echo to
//! equal it, while the child verifies what it received against the same
//! deterministic pattern.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use uniserve_worker_ipc::{Header, SocketClient, SocketServer};

/// Payload bytes of the exchanged frame: above 64 MiB, so the frame spans
/// many socket buffers in each direction.
const PAYLOAD_BYTES: usize = 72 << 20;
const MESSAGE_ID: u64 = 7;
const SERVER_ROLE: &str = "UNISERVE_SOCKET_LARGE_FRAME_SERVER";
const DEADLINE: Duration = Duration::from_secs(120);

/// A payload with no short period, so a dropped, repeated or reordered chunk
/// changes the content.
fn pattern() -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    (0..PAYLOAD_BYTES)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

/// The worker side: binds, announces its address, receives one frame, checks
/// it, and answers with the same bytes.
fn serve() {
    let mut server = SocketServer::bind("127.0.0.1", 1).unwrap();
    println!("{}", server.address());
    server.accept(DEADLINE).unwrap();

    let deadline = Instant::now() + DEADLINE;
    let frame = loop {
        if let Some(frame) = server.try_recv().unwrap() {
            break frame;
        }
        assert!(Instant::now() < deadline, "no frame arrived");
        server.wait_incoming(Duration::from_millis(50)).unwrap();
    };
    assert!(frame.payload == pattern(), "the frame arrived altered");

    // The response frame is queued whole and finishes across later calls.
    server
        .respond_raw(
            Header {
                message_id: frame.header.message_id,
                ..Header::default()
            },
            &frame.payload,
        )
        .unwrap();
    let deadline = Instant::now() + DEADLINE;
    while !server.try_recv().is_err() && Instant::now() < deadline {
        server.wait_incoming(Duration::from_millis(50)).unwrap();
    }
}

#[test]
fn a_frame_above_64_mib_crosses_loopback_tcp_between_two_processes_unaltered() {
    if std::env::var_os(SERVER_ROLE).is_some() {
        serve();
        return;
    }

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "a_frame_above_64_mib_crosses_loopback_tcp_between_two_processes_unaltered",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(SERVER_ROLE, "1")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    // The child prints its listening address; the harness may precede it on
    // the same line.
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let address = loop {
        let line = lines
            .next()
            .expect("the server announces its address")
            .unwrap();
        if let Some(start) = line.find("127.0.0.1:") {
            break line[start..].trim().to_owned();
        }
    };

    let mut client = SocketClient::connect(&address, 1, Duration::from_secs(30)).unwrap();
    let payload = pattern();
    client
        .send_raw(
            Header {
                message_id: MESSAGE_ID,
                ..Header::default()
            },
            &payload,
        )
        .unwrap();
    let echo = client
        .recv_response_timeout(MESSAGE_ID, DEADLINE)
        .unwrap()
        .expect("the server answers within the deadline");
    assert_eq!(echo.payload.len(), payload.len());
    assert!(echo.payload == payload, "the echoed frame arrived altered");

    drop(client);
    assert!(
        child.wait().unwrap().success(),
        "the server saw the frame altered"
    );
}
