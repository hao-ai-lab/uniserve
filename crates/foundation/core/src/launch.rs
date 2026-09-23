//! Handing a bound socket to a rank process at its launch.
//!
//! A worker group's first rank serves the group's collective store. The
//! process that spawns that rank binds the store's listening socket before the
//! rank exists and passes it across the rank's exec, so the port is never
//! released between the reservation and the store serving on it.

use std::net::TcpListener;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::process::CommandExt;
use std::process::Command;

/// Launch descriptor key naming the descriptor number of the listening socket
/// the group's first rank serves its collective store on.
pub const RENDEZVOUS_LISTEN_FD: &str = "rendezvous_listen_fd";

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
