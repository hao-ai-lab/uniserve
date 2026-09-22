//! The head's registry of per-host launchers.
//!
//! The engine starts no process on another host. Each host of a deployment
//! runs one `uniserve-host` launcher that connects here and presents its host
//! identity; the head then sends that launcher the launch descriptors of the
//! ranks the placement put on its host, of every worker group, and the
//! launcher spawns them. The registry is bound once per deployment and shared
//! by its groups, so a host presents one launcher however many workers place
//! ranks on it; a rank is named to its launcher by worker and rank.
//!
//! A launcher's connection is its liveness signal in both directions. Closing
//! it terminates that host's ranks, which is how an instance stops a host it
//! can no longer reach without adding a heartbeat.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// What a launcher says when it connects.
#[derive(Deserialize)]
struct Presentation {
    /// The host identity this launcher owns, as the placement names it.
    host: String,
    /// A free port of that host for a rendezvous bound by a rank placed there.
    rendezvous_port: u16,
}

/// What a launcher says when one of its ranks exits.
#[derive(Deserialize)]
pub(crate) struct RankExit {
    /// Worker group of the process that exited.
    pub worker_id: String,
    /// Rank of the process that exited within its group.
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
    /// Stop every rank of one worker group the launcher owns, before the
    /// group is relaunched.
    Stop { worker_id: &'a str },
    /// Stop every rank the launcher owns.
    Terminate,
}

/// The registry of one deployment, shared by every group that places a rank
/// on another host.
pub(crate) type Launchers = Arc<Mutex<LauncherRegistry>>;

/// Locks the deployment's registry for one operation.
pub(crate) fn lock(
    registry: &Launchers,
) -> anyhow::Result<std::sync::MutexGuard<'_, LauncherRegistry>> {
    registry
        .lock()
        .map_err(|_| anyhow::anyhow!("the launcher registry is poisoned"))
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
    /// The port the launcher reserved on its host for a rendezvous.
    rendezvous_port: u16,
}

/// The head's launcher registration address and the launchers that presented.
pub(crate) struct LauncherRegistry {
    listener: TcpListener,
    address: String,
    /// Connected launchers by the host identity each presented.
    hosts: HashMap<String, Launcher>,
    /// Exits read from the launchers and not yet taken, by worker group: a
    /// launcher reports every group's ranks on one connection, and each
    /// group takes its own.
    exits: HashMap<String, Vec<(String, RankExit)>>,
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
            exits: HashMap::new(),
        })
    }

    /// Binds a registry and waits for every remote host of a deployment.
    ///
    /// `host` is this process's own identity; a launcher is awaited for each
    /// other host any of the placements names. A deployment placing every
    /// rank on this host needs no registry.
    pub(crate) fn for_hosts(
        host: &str,
        placements: impl IntoIterator<Item = impl AsRef<[crate::WorkerRank]>>,
        timeout: Duration,
    ) -> anyhow::Result<Option<Launchers>> {
        let mut remote_hosts: Vec<String> = placements
            .into_iter()
            .flat_map(|ranks| {
                ranks
                    .as_ref()
                    .iter()
                    .filter(|rank| rank.node != host)
                    .map(|rank| rank.node.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        remote_hosts.sort();
        remote_hosts.dedup();
        if remote_hosts.is_empty() {
            return Ok(None);
        }
        let mut registry = Self::bind()?;
        tracing::info!(
            address = registry.address(),
            hosts = ?remote_hosts,
            "awaiting a launcher for each host this instance does not run on"
        );
        registry.await_hosts(&remote_hosts, timeout)?;
        Ok(Some(Arc::new(Mutex::new(registry))))
    }

    /// Returns the address a launcher's command line names.
    pub(crate) fn address(&self) -> &str {
        &self.address
    }

    /// Returns a host address of this machine that a launcher reached it at.
    ///
    /// The head cannot name itself: its placement identity need not resolve on
    /// another host, and a bound wildcard address names no interface. But a
    /// launcher has already connected to this listener, so the connection's
    /// local address is a head address reachable from a launcher host. The
    /// rendezvous and registration addresses handed to remote ranks must use
    /// that address.
    pub(crate) fn reachable_host(&self) -> anyhow::Result<std::net::IpAddr> {
        let launcher = self
            .hosts
            .values()
            .next()
            .context("no launcher has presented a reachable head address")?;
        Ok(launcher
            .stream
            .local_addr()
            .context("reading the head address a launcher reached")?
            .ip())
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
        Ok((
            presentation.host,
            Launcher {
                stream,
                reader,
                rendezvous_port: presentation.rendezvous_port,
            },
        ))
    }

    /// Returns the rendezvous address for a group whose first rank runs on
    /// `host`, which is another machine.
    ///
    /// The first rank binds the collective store, so the store's address has
    /// to be one of that rank's own host: the address its launcher connected
    /// from, at the port the launcher reserved there.
    pub(crate) fn rendezvous_on(&self, host: &str) -> anyhow::Result<String> {
        let launcher = self
            .hosts
            .get(host)
            .with_context(|| format!("no launcher presented host {host}"))?;
        let address = launcher
            .stream
            .peer_addr()
            .with_context(|| format!("reading the address host {host} connected from"))?;
        Ok(format!(
            "tcp://{}:{}",
            address.ip(),
            launcher.rendezvous_port
        ))
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

    /// Takes every exit of one worker group's ranks its launchers have
    /// reported, keeping the other groups' exits for their own owners.
    pub(crate) fn drain_exits(&mut self, worker_id: &str) -> Vec<(String, RankExit)> {
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
                    self.exits
                        .entry(exit.worker_id.clone())
                        .or_default()
                        .push((host.clone(), exit));
                }
                line.clear();
            }
            let _ = launcher.stream.set_nonblocking(false);
        }
        self.exits.remove(worker_id).unwrap_or_default()
    }

    /// Stops one worker group's ranks on every launcher, ahead of relaunching
    /// the group; the launchers stay connected for the other groups.
    pub(crate) fn stop_worker(&mut self, worker_id: &str) -> anyhow::Result<()> {
        let line = serde_json::to_string(&Instruction::Stop { worker_id })
            .context("encoding a worker stop")?;
        for (host, launcher) in self.hosts.iter_mut() {
            launcher
                .stream
                .write_all(format!("{line}\n").as_bytes())
                .with_context(|| format!("sending a worker stop to host {host}"))?;
            launcher
                .stream
                .flush()
                .with_context(|| format!("flushing a worker stop to host {host}"))?;
        }
        self.exits.remove(worker_id);
        Ok(())
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
    pub registry: &'a Mutex<LauncherRegistry>,
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
        let mut registry = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("the launcher registry is poisoned"))?;
        registry.spawn_remote(
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

#[cfg(test)]
mod tests {
    use super::LauncherRegistry;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    /// One launcher serves every worker group placed on its host: the exits
    /// it reports are routed to the group they belong to, and a stop names
    /// the group whose ranks it ends.
    #[test]
    fn exits_are_routed_to_their_worker_and_a_stop_names_its_worker() {
        let mut registry = LauncherRegistry::bind().expect("a registry binds");
        let address = registry.address().to_owned();
        let launcher = std::thread::spawn(move || {
            let mut stream = TcpStream::connect(address).expect("the launcher connects");
            stream
                .write_all(b"{\"host\":\"b\",\"rendezvous_port\":1}\n")
                .expect("the launcher presents");
            stream
                .write_all(
                    b"{\"worker_id\":\"model\",\"rank\":5,\"status\":\"exit status: 1\"}\n\
                      {\"worker_id\":\"host\",\"rank\":1,\"status\":\"exit status: 2\"}\n",
                )
                .expect("the launcher reports two exits");
            let mut line = String::new();
            BufReader::new(stream)
                .read_line(&mut line)
                .expect("the launcher reads the stop");
            line
        });
        registry
            .await_hosts(&["b".to_owned()], Duration::from_secs(5))
            .expect("host b presents");

        // Both exits arrive on one connection; each group takes its own and
        // the other's is kept for its owner.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let mut model_exits = Vec::new();
        while model_exits.is_empty() && std::time::Instant::now() < deadline {
            model_exits = registry.drain_exits("model");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(model_exits.len(), 1);
        assert_eq!((model_exits[0].0.as_str(), model_exits[0].1.rank), ("b", 5));
        let host_exits = registry.drain_exits("host");
        assert_eq!(host_exits.len(), 1);
        assert_eq!(
            (host_exits[0].1.worker_id.as_str(), host_exits[0].1.rank),
            ("host", 1)
        );
        assert!(registry.drain_exits("model").is_empty());

        registry.stop_worker("host").expect("the stop is sent");
        let stop = launcher.join().expect("the launcher thread ends");
        assert_eq!(stop.trim(), "{\"stop\":{\"worker_id\":\"host\"}}");
    }
}
