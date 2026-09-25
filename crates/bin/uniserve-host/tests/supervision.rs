//! Host supervision through its public command stream and real child processes.
//!
//! Each test binds a TCP listener that stands in for the head, starts the
//! built `uniserve-host` binary against it, and exchanges the launcher's
//! newline-delimited JSON messages: instructions to the launcher, and its
//! presentation, reservation replies, and rank exit reports back.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

/// Kills and reaps the launcher process when a test ends, including when it
/// panics.
struct Launcher(Child);

impl Drop for Launcher {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Reads one newline-terminated JSON message from the launcher, panicking if
/// the read fails, including on the socket's read timeout, or if the line is
/// not JSON, as at end of stream.
fn receive(reader: &mut BufReader<TcpStream>) -> serde_json::Value {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    serde_json::from_str(&line).unwrap()
}

/// Starts a launcher against a head bound here and reads its presentation.
///
/// Returns the launcher guard, the head's end of the connection for sending
/// instructions, and a reader over a clone of it for receiving messages. The
/// read timeout is set on the socket, so the clone shares it.
fn start_launcher() -> (Launcher, TcpStream, BufReader<TcpStream>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let launcher = Launcher(
        Command::new(env!("CARGO_BIN_EXE_uniserve-host"))
            .args([
                "--head",
                &listener.local_addr().unwrap().to_string(),
                "--host-identity",
                "host",
            ])
            .stdout(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let (stream, _) = listener.accept().unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    assert_eq!(receive(&mut reader)["host"], "host");
    (launcher, stream, reader)
}

#[test]
fn groups_reserve_distinct_ports_and_exits_arrive_without_another_instruction() {
    let (_launcher, mut stream, mut reader) = start_launcher();

    // A partial instruction must survive the supervisor's child-exit polling.
    // The pause outlasts the read timeout `supervise` sets on the head
    // connection, so the partial line spans more than one poll.
    stream.write_all(b"{\"reserve\":").unwrap();
    std::thread::sleep(Duration::from_millis(300));
    stream.write_all(b"{\"worker_id\":\"first\"}}\n").unwrap();
    let first = receive(&mut reader);
    writeln!(
        stream,
        "{}",
        serde_json::json!({"reserve": {"worker_id": "second"}})
    )
    .unwrap();
    let second = receive(&mut reader);
    assert_ne!(first["port"], second["port"]);

    // /bin/false is a real process with a known exit status; numerical worker
    // initialization is irrelevant to the host's independent exit reporting.
    writeln!(
        stream,
        "{}",
        serde_json::json!({"spawn": {
            "rank": 0, "worker_id": "first", "generation": 3, "world_size": 1,
            "python": "/bin/false", "descriptor": {}, "environment": {}
        }})
    )
    .unwrap();

    // No further instruction is sent: the report must come from the
    // launcher's own polling of its ranks. It names the generation the
    // rank's spawn carried, which is how the head tells it from an exit of
    // the same rank before its group was relaunched.
    let exit = receive(&mut reader);
    assert_eq!(exit["worker_id"], "first");
    assert_eq!(exit["generation"], 3);
    assert_eq!(exit["rank"], 0);
    assert_eq!(exit["status"], "exit status: 1");
}

#[test]
fn the_first_rank_serves_its_store_on_the_reserved_port() {
    let (_launcher, mut stream, mut reader) = start_launcher();
    writeln!(
        stream,
        "{}",
        serde_json::json!({"reserve": {"worker_id": "group"}})
    )
    .unwrap();
    let port = receive(&mut reader)["port"].as_u64().unwrap() as u16;

    // The rank's module accepts one peer on the socket its descriptor names,
    // which is how a collective store serves on an inherited socket. The
    // package is regular so it shadows any installed worker package.
    let modules = tempfile::tempdir().unwrap();
    let package = modules.path().join("uniserve_worker");
    std::fs::create_dir(&package).unwrap();
    std::fs::write(package.join("__init__.py"), "").unwrap();
    std::fs::write(
        package.join("main.py"),
        "import json, socket, sys\n\
         path = sys.argv[sys.argv.index('--launch-descriptor') + 1]\n\
         descriptor = json.load(open(path))\n\
         store = socket.socket(fileno=descriptor['rendezvous_listen_fd'])\n\
         peer, _ = store.accept()\n\
         peer.sendall(b'served\\n')\n",
    )
    .unwrap();
    writeln!(
        stream,
        "{}",
        serde_json::json!({"spawn": {
            "rank": 0, "worker_id": "group", "generation": 0, "world_size": 2,
            "python": "python3", "descriptor": {},
            "environment": {"PYTHONPATH": modules.path()}
        }})
    )
    .unwrap();

    // The reserved socket is already listening, so this connection waits in
    // its backlog until the rank starts and accepts it; the generous read
    // timeout covers the interpreter's startup.
    let peer = TcpStream::connect(("127.0.0.1", port)).unwrap();
    peer.set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let mut served = String::new();
    BufReader::new(peer).read_line(&mut served).unwrap();
    assert_eq!(served, "served\n");

    let exit = receive(&mut reader);
    assert_eq!(
        (
            exit["worker_id"].as_str(),
            exit["generation"].as_u64(),
            exit["rank"].as_u64()
        ),
        (Some("group"), Some(0), Some(0))
    );
    assert_eq!(exit["status"], "exit status: 0");
    // The launcher kept no copy of the socket it handed over, so the port
    // closes with the rank that served it.
    assert_eq!(
        TcpStream::connect(("127.0.0.1", port)).unwrap_err().kind(),
        std::io::ErrorKind::ConnectionRefused
    );
}
