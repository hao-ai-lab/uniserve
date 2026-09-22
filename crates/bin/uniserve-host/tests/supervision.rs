//! Host supervision through its public command stream and real child processes.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

struct Launcher(Child);

impl Drop for Launcher {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn receive(reader: &mut BufReader<TcpStream>) -> serde_json::Value {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    serde_json::from_str(&line).unwrap()
}

#[test]
fn groups_reserve_distinct_ports_and_exits_arrive_without_another_instruction() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let _launcher = Launcher(
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
    let (mut stream, _) = listener.accept().unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    assert_eq!(receive(&mut reader)["host"], "host");

    // A partial instruction must survive the supervisor's child-exit polling.
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
            "rank": 0, "worker_id": "first", "world_size": 1,
            "python": "/bin/false", "descriptor": {}, "environment": {}
        }})
    )
    .unwrap();
    let exit = receive(&mut reader);
    assert_eq!(exit["worker_id"], "first");
    assert_eq!(exit["rank"], 0);
    assert_eq!(exit["status"], "exit status: 1");
}
