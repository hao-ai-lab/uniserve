//! Host supervision through its public command stream and real child processes.
//!
//! Each test binds a TCP listener that stands in for the head, starts the
//! built `uniserve-host` binary against it, and exchanges the launcher's
//! messages: instructions to the launcher, and its presentation, reservation
//! replies, and rank exit reports back.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use uniserve_core::launch::{Instruction, Presentation, RankExit, RankLaunch, Report};

/// Kills and reaps the launcher process when a test ends, including when it
/// panics.
struct Launcher(Child);

impl Drop for Launcher {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Reads one newline-terminated message line from the launcher, panicking if
/// the read fails, including on the socket's read timeout.
fn receive_line(reader: &mut BufReader<TcpStream>) -> String {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    line
}

/// Reads one report from the launcher, panicking if it cannot be read or does
/// not decode, as at end of stream.
fn receive(reader: &mut BufReader<TcpStream>) -> Report {
    serde_json::from_str(&receive_line(reader)).unwrap()
}

/// Sends the launcher one instruction as a line of JSON.
fn send(stream: &mut TcpStream, instruction: &Instruction) {
    writeln!(stream, "{}", serde_json::to_string(instruction).unwrap()).unwrap();
}

/// Returns the port of a reservation reply.
fn reserved_port(report: Report) -> u16 {
    match report {
        Report::Reservation { port } => port,
        other => panic!("expected a reservation, received {other:?}"),
    }
}

/// Returns a spawn of `rank` of group `worker_id` under `generation`, run by
/// `python` with `environment` and an empty descriptor.
fn spawn(
    worker_id: &str,
    generation: u64,
    rank: u32,
    world_size: u32,
    python: &str,
    environment: &[(&str, &std::path::Path)],
) -> Instruction {
    Instruction::Spawn {
        generation,
        launch: RankLaunch {
            rank,
            worker_id: worker_id.to_owned(),
            world_size,
            python: python.to_owned(),
            descriptor: serde_json::json!({}),
            environment: environment
                .iter()
                .map(|(name, value)| ((*name).to_owned(), value.to_str().unwrap().to_owned()))
                .collect::<HashMap<_, _>>(),
        },
    }
}

/// Starts a launcher against a head bound here and reads its presentation.
///
/// Returns the launcher guard, the head's end of the connection for sending
/// instructions, and a reader over a clone of it for receiving messages. The
/// read timeout is set on the socket, so the clone shares it.
fn start_launcher() -> (Launcher, TcpStream, BufReader<TcpStream>) {
    start_launcher_with(Stdio::null(), Stdio::inherit())
}

/// `start_launcher` with the launcher's log output sent to `stdout` and
/// `stderr`.
fn start_launcher_with(
    stdout: Stdio,
    stderr: Stdio,
) -> (Launcher, TcpStream, BufReader<TcpStream>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let launcher = Launcher(
        Command::new(env!("CARGO_BIN_EXE_uniserve-host"))
            .args([
                "--head",
                &listener.local_addr().unwrap().to_string(),
                "--host-identity",
                "host",
            ])
            .stdout(stdout)
            .stderr(stderr)
            .spawn()
            .unwrap(),
    );
    let (stream, _) = listener.accept().unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let presentation: Presentation = serde_json::from_str(&receive_line(&mut reader)).unwrap();
    assert_eq!(presentation.host, "host");
    (launcher, stream, reader)
}

#[test]
fn groups_reserve_distinct_ports_and_exits_arrive_without_another_instruction() {
    let (_launcher, mut stream, mut reader) = start_launcher();

    // A partial instruction must survive the supervisor's child-exit polling.
    // The pause outlasts the read timeout `supervise` sets on the head
    // connection, so the partial line spans more than one poll.
    let reserve = serde_json::to_string(&Instruction::Reserve {
        worker_id: "first".to_owned(),
    })
    .unwrap();
    let (head, tail) = reserve.split_at(reserve.len() / 2);
    stream.write_all(head.as_bytes()).unwrap();
    std::thread::sleep(Duration::from_millis(300));
    writeln!(stream, "{tail}").unwrap();
    let first = reserved_port(receive(&mut reader));
    send(
        &mut stream,
        &Instruction::Reserve {
            worker_id: "second".to_owned(),
        },
    );
    let second = reserved_port(receive(&mut reader));
    assert_ne!(first, second);

    // /bin/false is a real process with a known exit status; numerical worker
    // initialization is irrelevant to the host's independent exit reporting.
    send(&mut stream, &spawn("first", 3, 0, 1, "/bin/false", &[]));

    // No further instruction is sent: the report must come from the
    // launcher's own polling of its ranks. It names the generation the
    // rank's spawn carried, which is how the head tells it from an exit of
    // the same rank before its group was relaunched.
    assert_eq!(
        receive(&mut reader),
        Report::Exit(RankExit {
            worker_id: "first".to_owned(),
            generation: 3,
            rank: 0,
            status: "exit status: 1".to_owned(),
        })
    );
}

#[test]
fn the_first_rank_serves_its_store_on_the_reserved_port() {
    let (_launcher, mut stream, mut reader) = start_launcher();
    send(
        &mut stream,
        &Instruction::Reserve {
            worker_id: "group".to_owned(),
        },
    );
    let port = reserved_port(receive(&mut reader));

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
    send(
        &mut stream,
        &spawn(
            "group",
            0,
            0,
            2,
            "python3",
            &[("PYTHONPATH", modules.path())],
        ),
    );

    // The reserved socket is already listening, so this connection waits in
    // its backlog until the rank starts and accepts it; the generous read
    // timeout covers the interpreter's startup.
    let peer = TcpStream::connect(("127.0.0.1", port)).unwrap();
    peer.set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let mut served = String::new();
    BufReader::new(peer).read_line(&mut served).unwrap();
    assert_eq!(served, "served\n");

    assert_eq!(
        receive(&mut reader),
        Report::Exit(RankExit {
            worker_id: "group".to_owned(),
            generation: 0,
            rank: 0,
            status: "exit status: 0".to_owned(),
        })
    );
    // The launcher kept no copy of the socket it handed over, so the port
    // closes with the rank that served it.
    assert_eq!(
        TcpStream::connect(("127.0.0.1", port)).unwrap_err().kind(),
        std::io::ErrorKind::ConnectionRefused
    );
}

/// A long-running stand-in rank started by a launcher, killed when a test
/// ends so a failing test leaves no process behind.
#[cfg(target_os = "linux")]
struct StandIn {
    pid: libc::pid_t,
    _directory: tempfile::TempDir,
}

#[cfg(target_os = "linux")]
impl StandIn {
    /// Has the launcher behind `stream` spawn rank 1 of group `group` as a
    /// shell script that records its PID and then sleeps under that PID, and
    /// waits for it to start.
    fn spawn(stream: &mut TcpStream, group: &str) -> Self {
        use std::os::unix::fs::PermissionsExt;

        // The launcher runs `<python> -m uniserve_worker.main ...`; the
        // script ignores those arguments.
        let directory = tempfile::tempdir().unwrap();
        let script = directory.path().join("rank");
        let pid_file = directory.path().join("rank.pid");
        std::fs::write(
            &script,
            "#!/bin/sh\necho $$ > \"$RANK_PID_FILE\"\nexec sleep 600\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        send(
            stream,
            &spawn(
                group,
                0,
                1,
                2,
                script.to_str().unwrap(),
                &[("RANK_PID_FILE", &pid_file)],
            ),
        );

        let mut pid = None;
        assert!(
            eventually(|| {
                pid = std::fs::read_to_string(&pid_file)
                    .ok()
                    .and_then(|text| text.trim().parse().ok());
                pid.is_some()
            }),
            "the launcher did not start the rank"
        );
        Self {
            pid: pid.unwrap(),
            _directory: directory,
        }
    }

    /// Whether the rank exists and is not a zombie awaiting its reaper.
    fn alive(&self) -> bool {
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{}/stat", self.pid)) else {
            return false;
        };
        // The state letter follows the parenthesized command name.
        let state = stat.rsplit(')').next().unwrap_or("").trim_start();
        !state.starts_with('Z')
    }
}

#[cfg(target_os = "linux")]
impl Drop for StandIn {
    fn drop(&mut self) {
        if self.alive() {
            // SAFETY: signal delivery to the stand-in this test started.
            unsafe { libc::kill(self.pid, libc::SIGKILL) };
        }
    }
}

/// Polls `condition` for up to 20 s, the bound for a launcher or rank to
/// react, and returns its last value.
#[cfg(target_os = "linux")]
fn eventually(mut condition: impl FnMut() -> bool) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_secs(20) {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    condition()
}

#[cfg(target_os = "linux")]
#[test]
fn a_killed_launcher_takes_its_ranks_with_it() {
    let (mut launcher, mut stream, _reader) = start_launcher();
    let rank = StandIn::spawn(&mut stream, "group");
    assert!(rank.alive());

    // SIGKILL skips every exit path of the launcher's own.
    launcher.0.kill().unwrap();
    launcher.0.wait().unwrap();

    assert!(
        eventually(|| !rank.alive()),
        "the rank outlived its launcher"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn a_closed_log_pipe_does_not_end_supervision() {
    let (mut launcher, mut stream, reader) = start_launcher_with(Stdio::piped(), Stdio::piped());
    let rank = StandIn::spawn(&mut stream, "group");

    // Closing the read ends, as a dropped SSH session does, makes every
    // later log write fail.
    drop(launcher.0.stdout.take());
    drop(launcher.0.stderr.take());

    // The launcher still follows the head: a stop ends the group's rank, and
    // the head closing the connection ends the launcher cleanly.
    send(
        &mut stream,
        &Instruction::Stop {
            worker_id: "group".to_owned(),
        },
    );
    assert!(
        eventually(|| !rank.alive()),
        "the stop did not end the rank"
    );
    assert_eq!(
        launcher.0.try_wait().unwrap(),
        None,
        "the launcher ended with its log pipe"
    );

    drop(stream);
    drop(reader);
    let mut status = None;
    assert!(eventually(|| {
        status = launcher.0.try_wait().unwrap();
        status.is_some()
    }));
    assert!(
        status.unwrap().success(),
        "the launcher failed: {:?}",
        status.unwrap()
    );
}
