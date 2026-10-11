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
//!
//! The connection carries the messages of `uniserve_core::launch`, one JSON
//! message per line. The launcher sends its `Presentation` once, then a
//! `Report`: a `RankExit` for each rank it still supervises that exits, which
//! excludes ranks a `Stop` ended, and a `Reservation` answering each
//! `Reserve`. The head sends `Instruction`s.
//!
//! Each worker group's launches carry a generation, which every `Spawn` names
//! and the launcher echoes in that rank's `RankExit`. A `Stop` ends the
//! group's generation before the group is relaunched under the next one. A
//! launcher may report an exit of a stopped rank before it reads the `Stop`,
//! and the head may read that report only after the relaunch, so an exit is
//! attributed to a group only when it names the group's current generation.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use uniserve_core::launch::{Instruction, Presentation, RankExit, RankLaunch, Report};

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

/// One connected launcher.
///
/// `reader` wraps a clone of `stream`'s socket. Reports are read only through
/// `reader`, so its buffer and `pending` stay coherent, and instructions are
/// written only through `stream`. Blocking mode and read timeouts belong to
/// the socket, so setting them on `stream` also governs reads through
/// `reader`.
struct Launcher {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
    /// Partial report retained across nonblocking reads.
    pending: Vec<u8>,
}

/// Rank exits read from the launchers and not yet taken, and the generation
/// each worker group is launched under.
///
/// A launcher reports every group's ranks on one connection, and each group
/// takes its own. Only exits of a group's current generation are kept, so a
/// report of a rank a stop ended is never attributed to its relaunch.
#[derive(Default)]
struct ExitReports {
    /// Each group's current generation; a group absent here has never been
    /// stopped and is in generation zero.
    generations: HashMap<String, u64>,
    /// Exits of each group's current generation, with the host that
    /// reported each, by worker group.
    reported: HashMap<String, Vec<(String, RankExit)>>,
}

impl ExitReports {
    /// Returns the generation `worker_id`'s launches currently carry.
    fn generation(&self, worker_id: &str) -> u64 {
        self.generations.get(worker_id).copied().unwrap_or(0)
    }

    /// Keeps one exit a launcher on `host` reported for its group, or
    /// discards it when it names a generation a stop has ended.
    fn file(&mut self, host: &str, exit: RankExit) {
        if exit.generation != self.generation(&exit.worker_id) {
            tracing::debug!(
                host,
                worker = %exit.worker_id,
                rank = exit.rank,
                generation = exit.generation,
                "discarding the exit of a stopped rank"
            );
            return;
        }
        self.reported
            .entry(exit.worker_id.clone())
            .or_default()
            .push((host.to_owned(), exit));
    }

    /// Takes every kept exit of `worker_id`.
    fn take(&mut self, worker_id: &str) -> Vec<(String, RankExit)> {
        self.reported.remove(worker_id).unwrap_or_default()
    }

    /// Ends `worker_id`'s current generation: its kept exits are discarded,
    /// as is every exit of that generation read later.
    fn end_generation(&mut self, worker_id: &str) {
        *self.generations.entry(worker_id.to_owned()).or_default() += 1;
        self.reported.remove(worker_id);
    }
}

/// The head's launcher registration address and the launchers that presented.
pub(crate) struct LauncherRegistry {
    listener: TcpListener,
    address: String,
    /// Connected launchers by the host identity each presented.
    hosts: HashMap<String, Launcher>,
    /// Exits read from the launchers, kept for the group each belongs to.
    exits: ExitReports,
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
            exits: ExitReports::default(),
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
    /// the instance waiting for ranks that were never started. Only this
    /// function accepts connections, so a launcher that connects after the
    /// last awaited host presented is never read. A presentation that cannot
    /// be read or decoded fails the wait.
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
                pending: Vec::new(),
            },
        ))
    }

    /// Returns the rendezvous for a group whose first rank runs on `host`,
    /// which is another machine.
    ///
    /// The first rank serves the collective store, so the store's address has
    /// to be one of that rank's own host: the address its launcher connected
    /// from, at the port the launcher reserved there. The launcher keeps that
    /// port bound and hands its socket to the first rank it spawns for the
    /// group, so the rendezvous returned here carries no listener.
    pub(crate) fn rendezvous_on(
        &mut self,
        host: &str,
        worker_id: &str,
    ) -> anyhow::Result<super::registration::Rendezvous> {
        let launcher = self
            .hosts
            .get_mut(host)
            .with_context(|| format!("no launcher presented host {host}"))?;
        let address = launcher.stream.peer_addr()?.ip();

        writeln!(
            launcher.stream,
            "{}",
            serde_json::to_string(&Instruction::Reserve {
                worker_id: worker_id.to_owned()
            })?
        )?;
        launcher
            .stream
            .set_read_timeout(Some(Duration::from_secs(10)))?;
        // The launcher may report rank exits before it answers; those are
        // kept for their groups and the read continues until the reply. The
        // closure lets the read deadline be cleared on every exit path.
        let reservation = (|| {
            loop {
                let read = launcher.reader.read_until(b'\n', &mut launcher.pending)?;
                anyhow::ensure!(read > 0, "launcher disconnected while reserving rendezvous");
                let report: Report = serde_json::from_slice(&launcher.pending)
                    .context("decoding a launcher report")?;
                launcher.pending.clear();
                match report {
                    Report::Reservation { port } => {
                        return Ok(super::registration::Rendezvous {
                            address: SocketAddr::new(address, port).to_string(),
                            listener: None,
                        });
                    }
                    Report::Exit(exit) => self.exits.file(host, exit),
                }
            }
        })();
        launcher.stream.set_read_timeout(None)?;
        reservation
    }

    /// Sends one rank's launch to the launcher that owns its host, under its
    /// group's current generation.
    pub(crate) fn spawn_remote(&mut self, host: &str, launch: RankLaunch) -> anyhow::Result<()> {
        let generation = self.exits.generation(&launch.worker_id);
        let launcher = self
            .hosts
            .get_mut(host)
            .with_context(|| format!("no launcher presented host {host}"))?;
        let line = serde_json::to_string(&Instruction::Spawn { generation, launch })
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

    /// Takes every exit of one worker group's current generation its
    /// launchers have reported, keeping the other groups' exits for their own
    /// owners.
    ///
    /// The drain is best effort and never blocks: it reads whatever each
    /// connection holds, keeps a partial line in `pending` for the next call,
    /// discards lines that do not decode as a rank exit report and exits of a
    /// generation a stop has ended, and moves on from a launcher whose socket
    /// cannot be switched to nonblocking mode or at its first read error or
    /// end of stream. A closed launcher connection is therefore not reported
    /// here.
    pub(crate) fn drain_exits(&mut self, worker_id: &str) -> Vec<(String, RankExit)> {
        for (host, launcher) in self.hosts.iter_mut() {
            if launcher.stream.set_nonblocking(true).is_err() {
                continue;
            }
            while launcher
                .reader
                .read_until(b'\n', &mut launcher.pending)
                .is_ok_and(|read| read > 0)
            {
                if let Ok(Report::Exit(exit)) = serde_json::from_slice(&launcher.pending) {
                    self.exits.file(host, exit);
                }
                launcher.pending.clear();
            }
            let _ = launcher.stream.set_nonblocking(false);
        }
        self.exits.take(worker_id)
    }

    /// Stops one worker group's ranks on every launcher, ahead of relaunching
    /// the group; the launchers stay connected for the other groups.
    ///
    /// The group's generation ends first, even if a stop cannot be sent: its
    /// exits already collected are discarded, and so is any exit of the
    /// stopped ranks read later, including one a launcher wrote before it
    /// read the stop. The launcher drops the stopped ranks from its
    /// supervision before terminating them, so a rank still running at the
    /// stop is not reported at all.
    pub(crate) fn stop_worker(&mut self, worker_id: &str) -> anyhow::Result<()> {
        self.exits.end_generation(worker_id);
        let line = serde_json::to_string(&Instruction::Stop {
            worker_id: worker_id.to_owned(),
        })
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
            let Ok(line) = serde_json::to_string(&Instruction::Terminate) else {
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
            .context("the interpreter path is not valid text")?
            .to_owned();
        let mut registry = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("the launcher registry is poisoned"))?;
        registry.spawn_remote(
            self.host,
            RankLaunch {
                rank,
                worker_id: worker_id.to_owned(),
                world_size,
                python,
                descriptor: descriptor.clone(),
                environment,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::LauncherRegistry;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;
    use std::time::{Duration, Instant};
    use uniserve_core::launch::{Instruction, Presentation, RankExit, RankLaunch, Report};

    /// Writes one message to the other end as a line of JSON.
    fn send(stream: &mut TcpStream, message: &impl serde::Serialize) {
        let line = serde_json::to_string(message).expect("the message encodes");
        writeln!(stream, "{line}").expect("the message is sent");
    }

    /// Reads one instruction the head sent.
    fn receive(reader: &mut BufReader<TcpStream>) -> Instruction {
        let mut line = String::new();
        reader.read_line(&mut line).expect("an instruction arrives");
        serde_json::from_str(&line).expect("the instruction decodes")
    }

    /// Connects a launcher for host `b` to `address` and presents it.
    fn present(address: String) -> (TcpStream, BufReader<TcpStream>) {
        let mut stream = TcpStream::connect(address).expect("the launcher connects");
        send(
            &mut stream,
            &Presentation {
                host: "b".to_owned(),
            },
        );
        let reader = BufReader::new(stream.try_clone().expect("the connection splits"));
        (stream, reader)
    }

    /// Reports the exit of `rank` of `worker_id` under `generation`.
    fn report_exit(
        stream: &mut TcpStream,
        worker_id: &str,
        generation: u64,
        rank: u32,
        status: &str,
    ) {
        send(
            stream,
            &Report::Exit(RankExit {
                worker_id: worker_id.to_owned(),
                generation,
                rank,
                status: status.to_owned(),
            }),
        );
    }

    /// Returns the statuses of the first nonempty drain of `worker_id`'s
    /// exits within five seconds, or none.
    fn await_exits(registry: &mut LauncherRegistry, worker_id: &str) -> Vec<String> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let exits = registry.drain_exits(worker_id);
            if !exits.is_empty() || Instant::now() >= deadline {
                return exits.into_iter().map(|(_, exit)| exit.status).collect();
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// One launcher serves every worker group placed on its host: the exits
    /// it reports are routed to the group they belong to, and a stop names
    /// the group whose ranks it ends.
    #[test]
    fn exits_are_routed_to_their_worker_and_a_stop_names_its_worker() {
        let mut registry = LauncherRegistry::bind().expect("a registry binds");
        let address = registry.address().to_owned();
        let launcher = std::thread::spawn(move || {
            let (mut stream, mut reader) = present(address);
            report_exit(&mut stream, "model", 0, 5, "exit status: 1");
            report_exit(&mut stream, "host", 0, 1, "exit status: 2");
            receive(&mut reader)
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
        assert_eq!(
            stop,
            Instruction::Stop {
                worker_id: "host".to_owned()
            }
        );
    }

    /// A reservation resolves to the port the launcher reports on the
    /// address it connected from, and an exit the launcher reports before
    /// its reply is kept for that exit's group.
    #[test]
    fn a_reservation_answers_with_the_launcher_port_after_earlier_exits() {
        let mut registry = LauncherRegistry::bind().expect("a registry binds");
        let address = registry.address().to_owned();
        let launcher = std::thread::spawn(move || {
            let (mut stream, mut reader) = present(address);
            let reserve = receive(&mut reader);
            report_exit(&mut stream, "host", 0, 1, "exit status: 2");
            send(&mut stream, &Report::Reservation { port: 4321 });
            reserve
        });
        registry
            .await_hosts(&["b".to_owned()], Duration::from_secs(5))
            .expect("host b presents");

        let rendezvous = registry
            .rendezvous_on("b", "model")
            .expect("the launcher reserves a port");
        assert_eq!(rendezvous.address, "127.0.0.1:4321");
        assert!(rendezvous.listener.is_none());
        assert_eq!(
            launcher.join().expect("the launcher thread ends"),
            Instruction::Reserve {
                worker_id: "model".to_owned()
            }
        );
        assert_eq!(await_exits(&mut registry, "host"), ["exit status: 2"]);
    }

    /// A stop ends a group's generation: an exit its launcher reported for
    /// a stopped rank before reading the stop, and which the head has not
    /// read when it relaunches the group, is not attributed to the relaunch,
    /// while an exit of a relaunched rank is.
    #[test]
    fn an_exit_reported_before_a_stop_is_not_attributed_to_the_relaunch() {
        let mut registry = LauncherRegistry::bind().expect("a registry binds");
        let address = registry.address().to_owned();
        let (reported, stale_exit_written) = std::sync::mpsc::channel();
        let launcher = std::thread::spawn(move || {
            let (mut stream, mut reader) = present(address);

            // A launcher reports an exit under the generation its rank's
            // spawn carried.
            let mut report = |spawn: &Instruction, status: &str| {
                let Instruction::Spawn { generation, launch } = spawn else {
                    panic!("expected a spawn, received {spawn:?}");
                };
                report_exit(
                    &mut stream,
                    &launch.worker_id,
                    *generation,
                    launch.rank,
                    status,
                );
            };

            // The failed rank's exit is written before the stop is read.
            let failed = receive(&mut reader);
            report(&failed, "signal: 9 (SIGKILL)");
            reported.send(()).expect("the head awaits the report");
            let stop = receive(&mut reader);
            let relaunched = receive(&mut reader);
            report(&relaunched, "exit status: 1");
            stop
        });
        registry
            .await_hosts(&["b".to_owned()], Duration::from_secs(5))
            .expect("host b presents");

        let launch = || RankLaunch {
            rank: 5,
            worker_id: "model".to_owned(),
            world_size: 8,
            python: "python3".to_owned(),
            descriptor: serde_json::json!({}),
            environment: HashMap::new(),
        };
        registry
            .spawn_remote("b", launch())
            .expect("the rank is launched");
        stale_exit_written
            .recv()
            .expect("the launcher reports the failed rank");
        registry.stop_worker("model").expect("the stop is sent");
        registry
            .spawn_remote("b", launch())
            .expect("the rank is relaunched");

        assert_eq!(await_exits(&mut registry, "model"), ["exit status: 1"]);
        let stop = launcher.join().expect("the launcher thread ends");
        assert_eq!(
            stop,
            Instruction::Stop {
                worker_id: "model".to_owned()
            }
        );
    }
}
