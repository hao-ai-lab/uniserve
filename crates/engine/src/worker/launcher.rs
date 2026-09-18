//! The head's registry of per-host launchers.
//!
//! The engine starts no process on another host. Each host of an instance runs
//! one `uniserve-host` launcher that connects here and presents its host
//! identity; the head then sends that launcher the launch descriptors of the
//! ranks the placement put on its host, and the launcher spawns them.
//!
//! A launcher's connection is its liveness signal in both directions. Closing
//! it terminates that host's ranks, which is how an instance stops a host it
//! can no longer reach without adding a heartbeat.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// What a launcher says when it connects.
#[derive(Deserialize)]
struct Presentation {
    /// The host identity this launcher owns, as the placement names it.
    host: String,
}

/// What a launcher says when one of its ranks exits.
#[derive(Deserialize)]
pub(crate) struct RankExit {
    /// Global rank identity of the process that exited.
    pub rank: u32,
    /// Exit status text the launcher read.
    pub status: String,
}

/// One instruction the head sends a launcher.
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Instruction<'a> {
    /// Start one rank from the descriptor the head derived for it.
    Spawn(RemoteLaunch<'a>),
    /// Stop every rank the launcher owns.
    Terminate,
}

/// Everything a launcher needs to start one rank.
#[derive(Serialize)]
pub(crate) struct RemoteLaunch<'a> {
    pub rank: u32,
    pub worker_id: &'a str,
    pub world_size: u32,
    pub python: &'a str,
    pub descriptor: &'a serde_json::Value,
    pub environment: HashMap<String, String>,
}

/// One connected launcher.
struct Launcher {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
}

/// The head's launcher registration address and the launchers that presented.
pub(crate) struct LauncherRegistry {
    listener: TcpListener,
    address: String,
    /// Connected launchers by the host identity each presented.
    hosts: HashMap<String, Launcher>,
}

impl LauncherRegistry {
    /// Binds the address every host's launcher connects to.
    ///
    /// A launcher may run on another machine, so this binds every interface
    /// rather than loopback; a single-host instance still reaches it locally.
    pub(crate) fn bind() -> anyhow::Result<Self> {
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .context("binding the launcher registration address")?;
        let address = listener
            .local_addr()
            .context("reading the launcher registration address")?
            .to_string();
        Ok(Self {
            listener,
            address,
            hosts: HashMap::new(),
        })
    }

    /// Returns the address a launcher's command line names.
    pub(crate) fn address(&self) -> &str {
        &self.address
    }

    /// Waits until every named host has presented a launcher.
    ///
    /// A host that never presents fails the launch by name rather than leaving
    /// the instance waiting for ranks that were never started.
    pub(crate) fn await_hosts(
        &mut self,
        hosts: &[String],
        timeout: Duration,
    ) -> anyhow::Result<()> {
        let deadline = Instant::now() + timeout;
        self.listener
            .set_nonblocking(true)
            .context("polling for launcher connections")?;
        while hosts.iter().any(|host| !self.hosts.contains_key(host)) {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    let host = Self::present(stream)?;
                    tracing::info!(host = %host.0, "a launcher presented its host");
                    self.hosts.insert(host.0, host.1);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    anyhow::ensure!(
                        Instant::now() < deadline,
                        "no launcher presented host {} before its deadline",
                        hosts
                            .iter()
                            .find(|host| !self.hosts.contains_key(*host))
                            .map(String::as_str)
                            .unwrap_or("<unknown>")
                    );
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(error) => {
                    return Err(error).context("accepting a launcher connection");
                }
            }
        }
        Ok(())
    }

    /// Reads one launcher's presentation.
    fn present(stream: TcpStream) -> anyhow::Result<(String, Launcher)> {
        stream
            .set_nonblocking(false)
            .context("reading a launcher presentation")?;
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .context("bounding a launcher presentation")?;
        let mut reader = BufReader::new(
            stream
                .try_clone()
                .context("splitting a launcher connection for reading")?,
        );
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .context("reading a launcher presentation")?;
        let presentation: Presentation = serde_json::from_str(line.trim())
            .with_context(|| format!("decoding a launcher presentation: {}", line.trim()))?;
        stream
            .set_read_timeout(None)
            .context("clearing a launcher's read deadline")?;
        Ok((presentation.host, Launcher { stream, reader }))
    }

    /// Sends one rank's launch to the launcher that owns its host.
    pub(crate) fn spawn_remote(
        &mut self,
        host: &str,
        launch: RemoteLaunch<'_>,
    ) -> anyhow::Result<()> {
        let launcher = self
            .hosts
            .get_mut(host)
            .with_context(|| format!("no launcher presented host {host}"))?;
        let line = serde_json::to_string(&Instruction::Spawn(launch))
            .context("encoding a remote launch")?;
        launcher
            .stream
            .write_all(format!("{line}\n").as_bytes())
            .with_context(|| format!("sending a launch to host {host}"))?;
        launcher
            .stream
            .flush()
            .with_context(|| format!("flushing a launch to host {host}"))
    }

    /// Takes every rank exit its launchers have reported.
    pub(crate) fn drain_exits(&mut self) -> Vec<(String, RankExit)> {
        let mut exits = Vec::new();
        for (host, launcher) in self.hosts.iter_mut() {
            if launcher.stream.set_nonblocking(true).is_err() {
                continue;
            }
            let mut line = String::new();
            while launcher
                .reader
                .read_line(&mut line)
                .is_ok_and(|read| read > 0)
            {
                if let Ok(exit) = serde_json::from_str::<RankExit>(line.trim()) {
                    exits.push((host.clone(), exit));
                }
                line.clear();
            }
            let _ = launcher.stream.set_nonblocking(false);
        }
        exits
    }
}

impl Drop for LauncherRegistry {
    /// Stops every host's ranks when the instance stops.
    ///
    /// A launcher terminates its ranks when this connection closes, so the
    /// instruction is a courtesy that lets a reachable host stop cleanly; the
    /// close does the work either way.
    fn drop(&mut self) {
        for (host, launcher) in self.hosts.iter_mut() {
            let Ok(line) = serde_json::to_string(&Instruction::<'_>::Terminate) else {
                continue;
            };
            if launcher
                .stream
                .write_all(format!("{line}\n").as_bytes())
                .is_err()
            {
                tracing::debug!(host = %host, "a launcher was already gone at shutdown");
            }
        }
    }
}

/// One host's launcher, addressed while a rank's launch is being derived.
///
/// The head builds a rank's descriptor and command the same way whether the
/// rank runs here or elsewhere; this delivers that launch to the host that
/// owns it, taking the environment from the command rather than deriving it a
/// second time.
pub(crate) struct RemoteHost<'a> {
    pub registry: &'a mut LauncherRegistry,
    pub host: &'a str,
}

impl RemoteHost<'_> {
    /// Sends one rank's launch to this host's launcher.
    pub(crate) fn deliver(
        &mut self,
        rank: u32,
        world_size: u32,
        python: &std::path::Path,
        worker_id: &str,
        descriptor: &serde_json::Value,
        command: &std::process::Command,
    ) -> anyhow::Result<()> {
        let environment = command
            .get_envs()
            .filter_map(|(name, value)| {
                Some((
                    name.to_str()?.to_owned(),
                    value
                        .and_then(|value| value.to_str())
                        .unwrap_or("")
                        .to_owned(),
                ))
            })
            .collect();
        let python = python
            .to_str()
            .context("the interpreter path is not valid text")?;
        self.registry.spawn_remote(
            self.host,
            RemoteLaunch {
                rank,
                worker_id,
                world_size,
                python,
                descriptor,
                environment,
            },
        )
    }
}
