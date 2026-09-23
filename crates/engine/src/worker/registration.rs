//! Rank registration: every rank reports the endpoint its channel is bound to.
//!
//! The head binds one registration address per worker group and passes it in
//! each rank's launch descriptor. A rank creates its own channel endpoint,
//! connects to that address and reports it, and the engine binds the rank's
//! channel from the report. Naming the endpoint on the rank rather than on the
//! head is what lets a rank on another host present a socket endpoint the head
//! could not have chosen in advance.

use std::io::{BufRead, BufReader};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::time::{Duration, Instant};

use anyhow::Context;
use serde::Deserialize;

/// Deadline for the whole group's reports, measured from the first spawn.
///
/// A rank reports as soon as its channel endpoint exists, which is before it
/// loads any weights, so this only has to cover interpreter and library import.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(300);
/// Interval between accept attempts while no rank has connected.
const ACCEPT_INTERVAL: Duration = Duration::from_millis(20);
/// Deadline for one accepted connection to deliver its report line.
const REPORT_READ_TIMEOUT: Duration = Duration::from_secs(30);
/// The channel mechanism a rank on the head's host offers.
use uniserve_worker_ipc::{SHARED_STORAGE_CHANNEL, SOCKET_CHANNEL};

/// One rank's report of the channel endpoint the engine connects to.
#[derive(Debug, Deserialize)]
pub(crate) struct RankReport {
    /// Worker identity the rank was launched under.
    pub worker_id: String,
    /// Global rank within that worker.
    pub rank: u32,
    /// Channel mechanism the endpoint names.
    pub transport: String,
    /// Endpoint the engine binds this rank's channel to.
    pub endpoint: String,
}

/// A worker group's collective rendezvous.
pub(crate) struct Rendezvous {
    /// The store address every rank connects to, as `host:port`.
    pub address: String,
    /// The bound, listening socket the group's first rank serves the store
    /// on, held by the process that spawns that rank until it hands the
    /// socket over. None when the first rank runs on another host, whose
    /// launcher holds that host's reservation.
    pub listener: Option<TcpListener>,
}

/// Reserves the address a worker group's ranks rendezvous at.
///
/// The collective store is a TCP store so ranks on different hosts can reach
/// it; an instance on one host reserves a loopback address and keeps the
/// semantics it had over a shared directory, and one spanning hosts reserves
/// an address those hosts can route to. The head binds the socket and the
/// group's first rank, which it spawns, inherits it and serves the store on
/// it. The port therefore stays bound from this reservation until that rank
/// exits: nothing else on the host can take it while the rank starts.
pub(crate) fn reserve_rendezvous(head: Option<IpAddr>) -> anyhow::Result<Rendezvous> {
    let (bind, advertise) = interface(head);
    let listener = TcpListener::bind(SocketAddr::from((bind, 0)))
        .context("reserving the collective rendezvous address")?;
    let port = listener
        .local_addr()
        .context("reading the collective rendezvous address")?
        .port();
    Ok(Rendezvous {
        address: format!("{advertise}:{port}"),
        listener: Some(listener),
    })
}

/// Returns the interface to bind and the host a descriptor names.
///
/// `head` is an address of this host that the other hosts route to, and none
/// when every rank runs here. A loopback address reaches only the host that
/// binds it, so an instance spanning hosts binds every interface and names
/// itself by that routable address; a bound wildcard would name no interface
/// and a placement identity need not resolve elsewhere.
fn interface(head: Option<IpAddr>) -> (Ipv4Addr, String) {
    match head {
        Some(address) => (Ipv4Addr::UNSPECIFIED, address.to_string()),
        None => (Ipv4Addr::LOCALHOST, Ipv4Addr::LOCALHOST.to_string()),
    }
}

/// The head's registration address for one worker group's ranks.
pub(crate) struct RankRegistry {
    listener: TcpListener,
    address: String,
}

impl RankRegistry {
    /// Binds the address every rank of one group reports its endpoint to.
    ///
    /// `head` is an address of this host that the other hosts route to when
    /// some rank runs elsewhere, and none when every rank runs here: a rank on
    /// another host cannot reach a loopback address, and one on this host does
    /// not need a routable one.
    pub(crate) fn bind(head: Option<IpAddr>) -> anyhow::Result<Self> {
        let (bind, advertise) = interface(head);
        let listener = TcpListener::bind(SocketAddr::from((bind, 0)))
            .context("binding the rank registration address")?;
        listener
            .set_nonblocking(true)
            .context("making the rank registration address pollable")?;
        let port = listener
            .local_addr()
            .context("reading the rank registration address")?
            .port();
        Ok(Self {
            listener,
            address: format!("{advertise}:{port}"),
        })
    }

    /// Returns the address ranks connect to, as it travels in the descriptor.
    pub(crate) fn address(&self) -> &str {
        &self.address
    }

    /// Collects one report per rank, ordered by the rank each names.
    ///
    /// `alive` is consulted whenever no report is waiting, so a rank that dies
    /// before reporting fails the launch by name instead of consuming the
    /// deadline, while a report already queued is still read.
    pub(crate) fn collect(
        &self,
        worker_id: &str,
        expected: usize,
        mut alive: impl FnMut() -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<RankReport>> {
        let deadline = Instant::now() + REGISTRATION_TIMEOUT;
        let mut reports: Vec<Option<RankReport>> = (0..expected).map(|_| None).collect();
        let mut received = 0;
        while received < expected {
            let stream = match self.listener.accept() {
                Ok((stream, _)) => stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    // Reports already queued are read before a rank's exit is
                    // noticed, so a rank that registers and then dies is a
                    // failure of what it does next, not of registration.
                    alive()?;
                    if Instant::now() >= deadline {
                        let silent = reports
                            .iter()
                            .enumerate()
                            .filter(|(_, report)| report.is_none())
                            .map(|(rank, _)| rank.to_string())
                            .collect::<Vec<_>>()
                            .join(", ");
                        anyhow::bail!(
                            "ranks {silent} reported no endpoint within {}s",
                            REGISTRATION_TIMEOUT.as_secs()
                        );
                    }
                    std::thread::sleep(ACCEPT_INTERVAL);
                    continue;
                }
                Err(error) => return Err(error).context("accepting a rank registration"),
            };
            // A report is one JSON line; the connection carries nothing else
            // and closes once the head has it.
            stream
                .set_nonblocking(false)
                .context("reading a rank registration")?;
            stream
                .set_read_timeout(Some(REPORT_READ_TIMEOUT))
                .context("reading a rank registration")?;
            let mut line = String::new();
            BufReader::new(stream)
                .read_line(&mut line)
                .context("reading a rank registration")?;
            let report: RankReport = serde_json::from_str(line.trim())
                .with_context(|| format!("parsing the rank registration {line:?}"))?;
            anyhow::ensure!(
                report.worker_id == worker_id,
                "rank registration names worker {} instead of {worker_id}",
                report.worker_id
            );
            // A rank on the head's host offers shared storage and a rank
            // elsewhere offers a socket. Any other mechanism is refused by
            // name rather than bound with the wrong transport.
            anyhow::ensure!(
                matches!(
                    report.transport.as_str(),
                    SHARED_STORAGE_CHANNEL | SOCKET_CHANNEL
                ),
                "rank {} offers the unsupported channel transport {}",
                report.rank,
                report.transport
            );
            let slot = reports
                .get_mut(report.rank as usize)
                .with_context(|| format!("rank {} is outside this worker", report.rank))?;
            anyhow::ensure!(
                slot.is_none(),
                "rank {} reported its endpoint more than once",
                report.rank
            );
            *slot = Some(report);
            received += 1;
        }
        Ok(reports
            .into_iter()
            .map(|report| report.expect("every rank slot received a report"))
            .collect())
    }
}
