//! The per-host launcher.
//!
//! Each host of a deployment runs one of these, started by the cluster; exactly
//! one host runs the head. The engine does not start processes on other hosts,
//! so a launcher connects out to the head, presents its host identity, receives
//! the launch descriptors of the ranks placed on its host, of every worker
//! group the deployment places there, spawns them, reports their exits by
//! worker and rank, stops one group's ranks on instruction ahead of that
//! group's relaunch, and terminates them all when the head's connection
//! closes.
//!
//! It does nothing else. The head derives every launch value once, so a
//! launcher's command line is the head's address and its own host identity.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
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
    /// A port of this host that the group's first rank may bind its
    /// collective rendezvous on, when that rank is placed here. The head
    /// cannot reserve a port on another machine, and the store must be bound
    /// where its owning rank runs.
    rendezvous_port: u16,
}

/// One instruction the head sends a launcher.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Instruction {
    /// Start one rank from the descriptor the head derived for it.
    Spawn(Spawn),
    /// Stop every rank of one worker group this launcher owns.
    Stop { worker_id: String },
    /// Stop every rank this launcher owns and exit.
    Terminate,
}

/// Everything needed to start one rank.
#[derive(Deserialize)]
struct Spawn {
    /// Global rank identity, which is also how exits are reported.
    rank: u32,
    /// Worker group the rank belongs to.
    worker_id: String,
    /// Ranks in the whole process world.
    world_size: u32,
    /// Interpreter that runs the worker module.
    python: String,
    /// The head's complete launch descriptor for this rank.
    descriptor: serde_json::Value,
    /// Environment the rank's numerical libraries read.
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
    /// Retains the descriptor file until the rank has read it.
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

    // The head places ranks by host identity, so the launcher says which host
    // it is before it can be given anything to run, and names a free port of
    // this host for a rendezvous the head may place here.
    let rendezvous_port = std::net::TcpListener::bind((std::net::Ipv4Addr::UNSPECIFIED, 0))
        .context("reserving a rendezvous port on this host")?
        .local_addr()
        .context("reading the reserved rendezvous port")?
        .port();
    let presentation = serde_json::to_string(&Presentation {
        host: &args.host_identity,
        rendezvous_port,
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

    // A closed head connection terminates this host's ranks, whether it closed
    // because the instance stopped or because the head was lost.
    terminate(&mut ranks);
    outcome
}

/// Follows the head's instructions until it closes the connection.
fn supervise(
    reader: &mut BufReader<TcpStream>,
    writer: &mut TcpStream,
    args: &Args,
    ranks: &mut HashMap<RankKey, Rank>,
) -> anyhow::Result<()> {
    let mut line = String::new();
    loop {
        line.clear();
        let read = reader
            .read_line(&mut line)
            .context("reading an instruction from the head")?;
        if read == 0 {
            tracing::info!("head connection closed; terminating this host's ranks");
            return Ok(());
        }
        let instruction: Instruction = serde_json::from_str(line.trim())
            .with_context(|| format!("decoding an instruction from the head: {}", line.trim()))?;
        match instruction {
            Instruction::Spawn(spawn) => {
                let key = (spawn.worker_id.clone(), spawn.rank);
                let started = start_rank(args, spawn).with_context(|| {
                    format!("starting rank {} of worker {} on this host", key.1, key.0)
                })?;
                ranks.insert(key.clone(), started);
                tracing::info!(worker = %key.0, rank = key.1, "started a rank");
            }
            Instruction::Stop { worker_id } => {
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
fn start_rank(args: &Args, spawn: Spawn) -> anyhow::Result<Rank> {
    let directory = tempfile::Builder::new()
        .prefix("uniserve-worker-launch")
        .tempdir()
        .context("creating the launch descriptor directory")?;
    let path = directory.path().join("launch.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&spawn.descriptor)?)
        .context("writing the launch descriptor")?;

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
    let child = command
        .spawn()
        .with_context(|| format!("spawning rank {} with {}", spawn.rank, spawn.python))?;
    tracing::debug!(rank = spawn.rank, host = %args.host_identity, "spawned");
    Ok(Rank {
        child,
        _descriptor: directory,
    })
}

/// Reports every rank that has exited since the last report.
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

/// Stops every rank in `ranks`.
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
