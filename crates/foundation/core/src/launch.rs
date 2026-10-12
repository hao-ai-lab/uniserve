//! The messages between the head and a host's launcher, and handing a bound
//! socket to a rank process at its launch.
//!
//! The head starts no process on another host: each host runs a
//! `uniserve-host` launcher that connects to the head's launcher registry.
//! The connection carries one JSON message per line. The launcher sends its
//! [`Presentation`] once, then a [`Report`] for each rank exit and each
//! reservation; the head sends [`Instruction`]s. Both binaries encode and
//! decode the messages defined here.
//!
//! A worker group's first rank serves the group's collective store. The
//! process that spawns that rank binds the store's listening socket before the
//! rank exists and passes it across the rank's exec, so the port is never
//! released between the reservation and the store serving on it.

use std::collections::HashMap;
use std::net::TcpListener;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::process::CommandExt;
use std::process::Command;

use serde::{Deserialize, Serialize};

/// Launch descriptor key naming the descriptor number of the listening socket
/// the group's first rank serves its collective store on.
pub const RENDEZVOUS_LISTEN_FD: &str = "rendezvous_listen_fd";

/// What a launcher tells the head when it connects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Presentation {
    /// The host identity this launcher owns, as the placement names it.
    pub host: String,
}

/// One instruction the head sends a launcher.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Instruction {
    /// Start one rank as a member of its worker group's `generation`.
    ///
    /// The head launches a group under a new generation each time it
    /// relaunches the group after a `Stop`, and the rank's exit report carries
    /// the generation back, so an exit the launcher reported before it read
    /// the stop is not attributed to the relaunch.
    Spawn { generation: u64, launch: RankLaunch },
    /// Bind and hold a collective store socket for one worker group, which
    /// the launcher hands to that group's first rank when it spawns it, and
    /// answer with a `Report::Reservation`.
    Reserve { worker_id: String },
    /// Stop every rank of one worker group the launcher owns, before the
    /// group is relaunched under its next generation. The launcher reports no
    /// exit of the ranks it stops.
    Stop { worker_id: String },
    /// Stop every rank the launcher owns and exit.
    Terminate,
}

/// Everything a launcher needs to start one rank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankLaunch {
    /// Rank within its worker group's process world, not host-relative.
    pub rank: u32,
    /// Worker group the rank belongs to.
    pub worker_id: String,
    /// Ranks in the worker group's process world.
    pub world_size: u32,
    /// Interpreter that runs the worker module.
    pub python: String,
    /// The launch descriptor the rank reads, as the head derived it for a
    /// local spawn. The launcher adds the inherited store socket's number
    /// (`RENDEZVOUS_LISTEN_FD`) to the first rank's copy, which only it knows.
    pub descriptor: serde_json::Value,
    /// The variables the head's spawn command for this rank sets, applied
    /// over the environment the launcher inherited. A variable the command
    /// removes is sent with an empty value and set empty rather than removed.
    pub environment: HashMap<String, String>,
}

/// One message a launcher sends the head after its presentation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Report {
    /// A rank the launcher still supervised exited.
    Exit(RankExit),
    /// The answer to the `Instruction::Reserve` the head sent last: the
    /// launcher holds the group's store socket, bound at `port` on every
    /// interface of its host.
    Reservation { port: u16 },
}

/// What a launcher reports when one of its ranks exits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankExit {
    /// Worker group of the process that exited.
    pub worker_id: String,
    /// The group generation the rank's `Instruction::Spawn` carried.
    pub generation: u64,
    /// Rank of the process that exited within its group.
    pub rank: u32,
    /// Exit status text, or the reason the status could not be read.
    pub status: String,
}

/// Makes the process `command` starts inherit `listener` at the descriptor
/// number returned, which the caller names in that process's launch
/// descriptor.
///
/// The command owns `listener` from here on and closes it when the command is
/// dropped, so the socket stays open until the spawn has duplicated it and the
/// spawning process holds no listening copy afterwards. Close-on-exec is
/// cleared only in the forked child: the descriptor keeps the flag in the
/// spawning process, so no other process it starts meanwhile inherits the
/// socket. A failure to clear the flag fails the spawn.
pub fn inherit_listener(command: &mut Command, listener: TcpListener) -> RawFd {
    let fd = listener.as_raw_fd();

    // SAFETY: the closure runs in the forked child between fork and exec, and
    // only calls `fcntl`, which is async-signal-safe. `listener` is moved into
    // the closure, so the descriptor is open in the parent at fork time.
    unsafe {
        command.pre_exec(move || {
            // FD_CLOEXEC is the only descriptor flag, so clearing every flag
            // leaves the socket open across exec at the same number.
            if libc::fcntl(listener.as_raw_fd(), libc::F_SETFD, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    fd
}
