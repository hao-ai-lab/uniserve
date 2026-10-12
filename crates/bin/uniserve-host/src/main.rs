//! The per-host launcher.
//!
//! Each host of a deployment runs one of these, started by the cluster; exactly
//! one host runs the head. The engine does not start processes on other hosts,
//! so a launcher connects out to the head, presents its host identity, receives
//! the launch descriptors of the ranks placed on its host, of every worker
//! group the deployment places there, spawns them, reports their exits by
//! worker, generation and rank, stops one group's ranks on instruction ahead
//! of that group's relaunch, and terminates them all when the head's
//! connection closes. For a group whose first rank it runs, it also holds the
//! bound socket that rank serves the group's collective store on, from the
//! head's reservation until the rank inherits it at spawn.
//!
//! Its ranks never outlive it. An orderly end of supervision terminates them,
//! and on Linux the kernel kills each rank when the launcher dies by any other
//! means. Its log output is diagnostics only: a launcher is often started over
//! an SSH session whose pipe closes before the deployment stops, and a failed
//! log write must not end supervision.
//!
//! It does nothing else. The head derives every launch value once, so a
//! launcher's command line is the head's address and its own host identity.
//!
//! The connection carries the messages of `uniserve_core::launch`, one JSON
//! message per line; the head's side is the engine's `LauncherRegistry`.
//!
//! Every spawn names the generation of its worker group that the head is
//! launching, and the rank's exit report echoes it. The head relaunches a
//! group under a new generation after stopping it, so an exit this launcher
//! reported just before it read the stop is recognized as the stopped
//! generation's rather than the relaunch's.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::process::{Child, Command};

use anyhow::Context;
use clap::Parser;
use uniserve_core::launch::{Instruction, Presentation, RankExit, RankLaunch, Report};

/// Spawns and supervises the ranks an instance places on this host.
#[derive(Parser, Debug)]
#[command(
    name = "uniserve-host",
    about = "Launch and supervise one host's ranks"
)]
struct Args {
    /// Address of the head's launcher registration endpoint.
    #[arg(long)]
    head: String,
    /// This host's identity, as the placement names it.
    #[arg(long = "host-identity")]
    host_identity: String,
}

/// A rank's identity on this host: its worker group and its rank within it.
type RankKey = (String, u32);

/// One rank this launcher owns.
struct Rank {
    child: Child,
    /// The group generation the rank's spawn named.
    generation: u64,
    /// Retains the descriptor file until the rank has read it; dropping the
    /// `Rank` deletes the directory.
    _descriptor: tempfile::TempDir,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        // tracing-subscriber reports a failed write with `eprintln!`, which
        // panics, and so aborts this process, when stderr is the same closed
        // pipe; dropping the event keeps supervision independent of it.
        .log_internal_errors(false)
        .init();
    let args = Args::parse();

    let stream = TcpStream::connect(&args.head)
        .with_context(|| format!("connecting to the head at {}", args.head))?;
    stream
        .set_nodelay(true)
        .context("disabling Nagle on the head connection")?;
    let mut writer = stream
        .try_clone()
        .context("splitting the head connection for writing")?;
    let mut reader = BufReader::new(stream);

    let presentation = serde_json::to_string(&Presentation {
        host: args.host_identity.clone(),
    })?;
    writer
        .write_all(format!("{presentation}\n").as_bytes())
        .context("presenting this host to the head")?;
    writer
        .flush()
        .context("flushing this host's presentation")?;
    tracing::info!(host = %args.host_identity, head = %args.head, "presented this host");

    let mut ranks: HashMap<RankKey, Rank> = HashMap::new();
    let outcome = supervise(&mut reader, &mut writer, &args, &mut ranks);

    // However supervision ends (the head closing the connection because the
    // instance stopped or the head was lost, a `Terminate`, or an error), this
    // host's ranks are terminated before the launcher exits.
    terminate(&mut ranks);
    outcome
}

/// Follows the head's instructions until it closes the connection or sends
/// `Terminate`, reporting rank exits as they happen.
///
/// Fails when the read timeout cannot be set, on a read error other than a
/// timeout or interruption, an undecodable instruction, a failed reservation
/// or rank start, or a failed write to the head. The caller terminates the
/// remaining ranks in every case.
fn supervise(
    reader: &mut BufReader<TcpStream>,
    writer: &mut TcpStream,
    args: &Args,
    ranks: &mut HashMap<RankKey, Rank>,
) -> anyhow::Result<()> {
    // The read timeout bounds how long an exit waits to be reported: exit
    // reporting must not depend on another instruction arriving, including
    // while that instruction is partial. `read_until` keeps the bytes it read
    // before a timeout in `line`, so a partial instruction survives it.
    reader
        .get_ref()
        .set_read_timeout(Some(std::time::Duration::from_millis(100)))?;
    let mut line = Vec::new();
    // Bound store sockets by worker group, each held until that group's first
    // rank is spawned here and inherits it.
    let mut rendezvous: HashMap<String, TcpListener> = HashMap::new();
    loop {
        report_exits(writer, ranks)?;
        let read = match reader.read_until(b'\n', &mut line) {
            Ok(read) => read,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::Interrupted
                ) =>
            {
                continue;
            }
            Err(error) => return Err(error).context("reading an instruction from the head"),
        };
        if read == 0 {
            tracing::info!("head connection closed; terminating this host's ranks");
            return Ok(());
        }
        let instruction: Instruction =
            serde_json::from_slice(&line).context("decoding an instruction from the head")?;
        line.clear();
        match instruction {
            Instruction::Reserve { worker_id } => {
                // This launcher holds the bound socket until it spawns the
                // group's first rank, which inherits it, so no other process
                // on this host, and no other group's reservation, can take the
                // port meanwhile.
                let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, 0))
                    .context("reserving a collective store address")?;
                let port = listener.local_addr()?.port();
                let response = serde_json::to_string(&Report::Reservation { port })?;
                rendezvous.insert(worker_id, listener);
                writeln!(writer, "{response}")?;
            }
            Instruction::Spawn { generation, launch } => {
                // The group's first rank serves its store on the reserved
                // socket; the others connect to it and bind nothing.
                let store_listener = if launch.rank == 0 {
                    rendezvous.remove(&launch.worker_id)
                } else {
                    None
                };
                let key = (launch.worker_id.clone(), launch.rank);
                let started =
                    start_rank(args, generation, launch, store_listener).with_context(|| {
                        format!("starting rank {} of worker {} on this host", key.1, key.0)
                    })?;
                ranks.insert(key.clone(), started);
                tracing::info!(worker = %key.0, rank = key.1, "started a rank");
            }
            Instruction::Stop { worker_id } => {
                // The group's ranks leave supervision before they are killed,
                // so their exits are not reported. An unspawned reservation of
                // the group is closed.
                let stopped: Vec<RankKey> = ranks
                    .keys()
                    .filter(|(worker, _)| *worker == worker_id)
                    .cloned()
                    .collect();
                let mut group: HashMap<RankKey, Rank> = stopped
                    .into_iter()
                    .filter_map(|key| ranks.remove(&key).map(|rank| (key, rank)))
                    .collect();
                terminate(&mut group);
                rendezvous.remove(&worker_id);
                tracing::info!(worker = %worker_id, "head asked this host to stop a worker's ranks");
            }
            Instruction::Terminate => {
                tracing::info!("head asked this host to stop");
                return Ok(());
            }
        }
        report_exits(writer, ranks)?;
    }
}

/// Starts one rank of its group's `generation` from the launch the head
/// derived for it.
///
/// `store_listener` is the group's reserved collective store socket when this
/// rank serves the store. The rank inherits it, and the launch descriptor
/// records the file descriptor number it inherits it at, which only this
/// process knows. This process keeps no copy once the rank is started.
fn start_rank(
    args: &Args,
    generation: u64,
    mut launch: RankLaunch,
    store_listener: Option<TcpListener>,
) -> anyhow::Result<Rank> {
    let directory = tempfile::Builder::new()
        .prefix("uniserve-worker-launch")
        .tempdir()
        .context("creating the launch descriptor directory")?;
    let path = directory.path().join("launch.json");

    let mut command = Command::new(&launch.python);
    command
        .arg("-m")
        .arg("uniserve_worker.main")
        .arg("--worker-id")
        .arg(&launch.worker_id)
        .arg("--rank")
        .arg(launch.rank.to_string())
        .arg("--world-size")
        .arg(launch.world_size.to_string())
        .arg("--launch-descriptor")
        .arg(&path);
    for (name, value) in &launch.environment {
        command.env(name, value);
    }
    die_with_launcher(&mut command);
    if let Some(listener) = store_listener {
        let fd = uniserve_core::launch::inherit_listener(&mut command, listener);
        launch
            .descriptor
            .as_object_mut()
            .context("the launch descriptor is not an object")?
            .insert(
                uniserve_core::launch::RENDEZVOUS_LISTEN_FD.to_owned(),
                fd.into(),
            );
    }
    std::fs::write(&path, serde_json::to_vec_pretty(&launch.descriptor)?)
        .context("writing the launch descriptor")?;

    let child = command
        .spawn()
        .with_context(|| format!("spawning rank {} with {}", launch.rank, launch.python))?;
    tracing::debug!(rank = launch.rank, host = %args.host_identity, "spawned");
    Ok(Rank {
        child,
        generation,
        _descriptor: directory,
    })
}

/// Makes the rank `command` starts die with this launcher.
///
/// The kernel sends the rank SIGKILL when the thread that spawned it exits,
/// which covers a launcher killed by a signal or aborted before `terminate`
/// runs. Every rank is spawned from the main thread, whose exit is the
/// process's. A rank whose launcher exited between fork and the request is
/// already orphaned, so it fails to start instead.
#[cfg(target_os = "linux")]
fn die_with_launcher(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    let launcher = std::process::id() as libc::pid_t;
    // SAFETY: the closure runs in the forked child between fork and exec and
    // calls only `prctl` and `getppid`, which are async-signal-safe, and
    // builds its errors without allocating.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() != launcher {
                return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
            }
            Ok(())
        });
    }
}

/// Elsewhere only an orderly end of supervision terminates the ranks.
#[cfg(not(target_os = "linux"))]
fn die_with_launcher(_command: &mut Command) {}

/// Reports every rank that has exited since the last report and stops
/// supervising it. A rank whose exit status cannot be read is reported as
/// exited.
fn report_exits(writer: &mut TcpStream, ranks: &mut HashMap<RankKey, Rank>) -> anyhow::Result<()> {
    let mut exited = Vec::new();
    for (key, owned) in ranks.iter_mut() {
        let status = match owned.child.try_wait() {
            Ok(Some(status)) => status.to_string(),
            Ok(None) => continue,
            Err(error) => format!("exit status unreadable: {error}"),
        };
        exited.push((key.clone(), owned.generation, status));
    }
    for ((worker_id, rank), generation, status) in exited {
        ranks.remove(&(worker_id.clone(), rank));
        let report = serde_json::to_string(&Report::Exit(RankExit {
            worker_id: worker_id.clone(),
            generation,
            rank,
            status: status.clone(),
        }))?;
        writer
            .write_all(format!("{report}\n").as_bytes())
            .context("reporting a rank exit to the head")?;
        writer.flush().context("flushing a rank exit report")?;
        tracing::warn!(worker = %worker_id, generation, rank, status = %status, "a rank exited");
    }
    Ok(())
}

/// Kills every rank in `ranks`, then reaps each one, leaving `ranks` empty.
fn terminate(ranks: &mut HashMap<RankKey, Rank>) {
    for ((worker_id, rank), owned) in ranks.iter_mut() {
        if let Err(error) = owned.child.kill() {
            tracing::warn!(worker = %worker_id, rank, %error, "a rank could not be stopped");
        }
    }
    for (_, mut owned) in ranks.drain() {
        let _ = owned.child.wait();
    }
}
