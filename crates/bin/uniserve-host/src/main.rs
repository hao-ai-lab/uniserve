//! The per-host launcher.
//!
//! Each host of a deployment runs one of these, started by the cluster; exactly
//! one host runs the head. The engine does not start processes on other hosts,
//! so a launcher connects out to the head, presents its host identity, receives
//! the launch descriptors of the ranks placed on its host, of every worker
//! group the deployment places there, spawns them, reports their exits by
//! worker and rank, stops one group's ranks on instruction ahead of that
//! group's relaunch, and terminates them all when the head's connection
//! closes. For a group whose first rank it runs, it also holds the bound
//! socket that rank serves the group's collective store on, from the head's
//! reservation until the rank inherits it at spawn.
//!
//! It does nothing else. The head derives every launch value once, so a
//! launcher's command line is the head's address and its own host identity.
//!
//! The connection carries newline-delimited JSON. The head's side is the
//! engine's `LauncherRegistry`, whose instruction and report types these must
//! stay field-compatible with.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::process::{Child, Command};

use anyhow::Context;
use clap::Parser;
use serde::{Deserialize, Serialize};

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

/// What this launcher tells the head when it connects.
#[derive(Serialize)]
struct Presentation<'a> {
    host: &'a str,
}

/// One instruction the head sends a launcher.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Instruction {
    /// Start one rank from the descriptor the head derived for it.
    Spawn(Spawn),
    /// Bind and hold a collective store socket for one group, which that
    /// group's first rank inherits when it is spawned here.
    Reserve { worker_id: String },
    /// Stop every rank of one worker group this launcher owns.
    Stop { worker_id: String },
    /// Stop every rank this launcher owns and exit.
    Terminate,
}

/// Everything needed to start one rank.
#[derive(Deserialize)]
struct Spawn {
    /// Rank within its worker group's process world, not host-relative.
    /// Exits are reported by worker group and this rank.
    rank: u32,
    /// Worker group the rank belongs to.
    worker_id: String,
    /// Ranks in the worker group's process world.
    world_size: u32,
    /// Interpreter that runs the worker module.
    python: String,
    /// The head's complete launch descriptor for this rank.
    descriptor: serde_json::Value,
    /// Environment variables the head set on its own spawn command for this
    /// rank. They are applied over the environment this launcher inherited; a
    /// variable the head's command removes arrives with an empty value and is
    /// set empty here rather than removed.
    environment: HashMap<String, String>,
}

/// What this launcher tells the head when a rank exits.
#[derive(Serialize)]
struct Exit<'a> {
    worker_id: &'a str,
    rank: u32,
    /// Exit status text, or the reason the status could not be read.
    status: &'a str,
}

/// A rank's identity on this host: its worker group and its rank within it.
type RankKey = (String, u32);

/// One rank this launcher owns.
struct Rank {
    child: Child,
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
        host: &args.host_identity,
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
                let response = serde_json::json!({"worker_id": worker_id, "port": port});
                rendezvous.insert(worker_id, listener);
                writeln!(writer, "{response}")?;
            }
            Instruction::Spawn(spawn) => {
                // The group's first rank serves its store on the reserved
                // socket; the others connect to it and bind nothing.
                let store_listener = if spawn.rank == 0 {
                    rendezvous.remove(&spawn.worker_id)
                } else {
                    None
                };
                let key = (spawn.worker_id.clone(), spawn.rank);
                let started = start_rank(args, spawn, store_listener).with_context(|| {
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

/// Starts one rank from the descriptor the head derived for it.
///
/// `store_listener` is the group's reserved collective store socket when this
/// rank serves the store. The rank inherits it, and the launch descriptor
/// records the file descriptor number it inherits it at, which only this
/// process knows. This process keeps no copy once the rank is started.
fn start_rank(
    args: &Args,
    mut spawn: Spawn,
    store_listener: Option<TcpListener>,
) -> anyhow::Result<Rank> {
    let directory = tempfile::Builder::new()
        .prefix("uniserve-worker-launch")
        .tempdir()
        .context("creating the launch descriptor directory")?;
    let path = directory.path().join("launch.json");

    let mut command = Command::new(&spawn.python);
    command
        .arg("-m")
        .arg("uniserve_worker.main")
        .arg("--worker-id")
        .arg(&spawn.worker_id)
        .arg("--rank")
        .arg(spawn.rank.to_string())
        .arg("--world-size")
        .arg(spawn.world_size.to_string())
        .arg("--launch-descriptor")
        .arg(&path);
    for (name, value) in &spawn.environment {
        command.env(name, value);
    }
    if let Some(listener) = store_listener {
        let fd = uniserve_core::launch::inherit_listener(&mut command, listener);
        spawn
            .descriptor
            .as_object_mut()
            .context("the launch descriptor is not an object")?
            .insert(
                uniserve_core::launch::RENDEZVOUS_LISTEN_FD.to_owned(),
                fd.into(),
            );
    }
    std::fs::write(&path, serde_json::to_vec_pretty(&spawn.descriptor)?)
        .context("writing the launch descriptor")?;

    let child = command
        .spawn()
        .with_context(|| format!("spawning rank {} with {}", spawn.rank, spawn.python))?;
    tracing::debug!(rank = spawn.rank, host = %args.host_identity, "spawned");
    Ok(Rank {
        child,
        _descriptor: directory,
    })
}

/// Reports every rank that has exited since the last report and stops
/// supervising it. A rank whose exit status cannot be read is reported as
/// exited.
fn report_exits(writer: &mut TcpStream, ranks: &mut HashMap<RankKey, Rank>) -> anyhow::Result<()> {
    let mut exited = Vec::new();
    for (key, owned) in ranks.iter_mut() {
        match owned.child.try_wait() {
            Ok(Some(status)) => exited.push((key.clone(), status.to_string())),
            Ok(None) => {}
            Err(error) => exited.push((key.clone(), format!("exit status unreadable: {error}"))),
        }
    }
    for ((worker_id, rank), status) in exited {
        ranks.remove(&(worker_id.clone(), rank));
        let report = serde_json::to_string(&Exit {
            worker_id: &worker_id,
            rank,
            status: &status,
        })?;
        writer
            .write_all(format!("{report}\n").as_bytes())
            .context("reporting a rank exit to the head")?;
        writer.flush().context("flushing a rank exit report")?;
        tracing::warn!(worker = %worker_id, rank, status = %status, "a rank exited");
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
