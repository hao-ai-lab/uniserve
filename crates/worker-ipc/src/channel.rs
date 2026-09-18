//! The engine's end of one rank's channel, over either transport.
//!
//! A rank on the head's host offers a shared-memory service; a rank elsewhere
//! offers a socket. The engine addresses both the same way: it sends a request
//! under an identity and takes the response that carries it back. Which
//! mechanism a rank offers is data the rank reports at registration, so the
//! engine holds whichever one the report named.

use std::sync::Arc;
use std::time::Duration;

use crate::iceoryx::{ClientEndpoint, Frame, IpcError, IpcResult, Pending, WakeEvents, WakeSender};
use crate::request::WorkerRequest;
use crate::socket::SocketClient;

/// The mechanism a rank names for its channel at registration.
pub const SHARED_MEMORY_CHANNEL: &str = "iceoryx2";
/// The mechanism a rank off the head's host names for its channel.
pub const SOCKET_CHANNEL: &str = "tcp";

/// A wake this engine can fire, whichever transport raised the need for it.
///
/// A command and a worker death are raised by threads inside the engine, so
/// neither travels between processes. A shared-memory channel already owns a
/// notifier for them; a socket channel raises them on a local descriptor the
/// engine polls beside the socket.
#[derive(Clone)]
pub enum Wake {
    /// Fires the shared-memory channel's notifier.
    Shared(WakeSender),
    /// Fires a local descriptor this process polls.
    Local(Arc<dyn Fn() + Send + Sync>),
}

impl Wake {
    /// Fires one wake.
    pub fn wake(&self) {
        match self {
            Self::Shared(sender) => sender.wake(),
            Self::Local(fire) => fire(),
        }
    }
}

/// One request the engine has sent and not yet matched to a response.
pub enum Outstanding {
    /// A retained shared-memory request handle.
    Shared(Pending),
    /// The identity a socket response will carry.
    Socket(u64),
}

/// The engine's end of one rank's channel.
pub enum RankChannel {
    /// A rank on the head's host, reached through shared memory.
    Shared(Box<ClientEndpoint>),
    /// A rank elsewhere, reached over a stream socket.
    Socket(Box<SocketClient>),
}

impl RankChannel {
    /// Binds the channel a rank named, refusing a mechanism the head cannot bind.
    pub fn connect(
        transport: &str,
        endpoint: &str,
        max_payload: usize,
        depth: usize,
        timeout: Duration,
    ) -> IpcResult<Self> {
        match transport {
            SHARED_MEMORY_CHANNEL => Ok(Self::Shared(Box::new(ClientEndpoint::connect(
                endpoint,
                max_payload,
                depth,
            )?))),
            SOCKET_CHANNEL => Ok(Self::Socket(Box::new(SocketClient::connect(
                endpoint,
                max_payload,
                timeout,
            )?))),
            other => Err(IpcError::Transport(format!(
                "rank offers the unsupported channel transport {other}"
            ))),
        }
    }

    /// Returns the wake a worker-death watcher fires.
    pub fn death_wake(&self) -> Wake {
        match self {
            Self::Shared(client) => Wake::Shared(client.death_wake()),
            Self::Socket(client) => Wake::Local(client.death_wake()),
        }
    }

    /// Sends one request without waiting for channel capacity.
    pub fn send_request_attempt(&mut self, request: &WorkerRequest) -> IpcResult<Outstanding> {
        match self {
            Self::Shared(client) => client
                .send_request_attempt(request)
                .map(Outstanding::Shared),
            Self::Socket(client) => client.send_request(request).map(Outstanding::Socket),
        }
    }

    /// Takes the response to one outstanding request, if it has arrived.
    pub fn try_recv_response(&mut self, pending: &Outstanding) -> IpcResult<Option<Frame>> {
        match (self, pending) {
            (Self::Shared(client), Outstanding::Shared(handle)) => client.try_recv_response(handle),
            (Self::Socket(client), Outstanding::Socket(identity)) => {
                client.try_recv_response(*identity)
            }
            _ => Err(IpcError::Transport(
                "an outstanding request was matched against another channel".into(),
            )),
        }
    }

    /// Waits for one outstanding response until the deadline passes.
    pub fn recv_response_timeout(
        &mut self,
        pending: &Outstanding,
        timeout: Duration,
    ) -> IpcResult<Option<Frame>> {
        match (self, pending) {
            (Self::Shared(client), Outstanding::Shared(handle)) => {
                client.recv_response_timeout(handle, timeout)
            }
            (Self::Socket(client), Outstanding::Socket(identity)) => {
                client.recv_response_timeout(*identity, timeout)
            }
            _ => Err(IpcError::Transport(
                "an outstanding request was matched against another channel".into(),
            )),
        }
    }

    /// Reports whether the rank's end of the channel is connected.
    ///
    /// A shared-memory request is loaned before a server exists to take it, so
    /// the engine asks the handle. A socket request could not have been sent
    /// without a connection, so there is nothing further to ask.
    pub fn is_connected(&self, pending: &Outstanding) -> bool {
        match pending {
            Outstanding::Shared(handle) => handle.number_of_server_connections() > 0,
            Outstanding::Socket(_) => true,
        }
    }

    /// Waits until a wake source fires or the deadline passes.
    pub fn wait_wake(&mut self, timeout: Duration) -> IpcResult<WakeEvents> {
        match self {
            Self::Shared(client) => client.wait_wake(timeout),
            Self::Socket(client) => client.wait_wake(timeout),
        }
    }

    /// Reports the wakes already observed without waiting.
    pub fn drain_wakes(&mut self) -> IpcResult<WakeEvents> {
        match self {
            Self::Shared(client) => client.drain_wakes(),
            Self::Socket(client) => client.drain_wakes(),
        }
    }

    /// Returns the descriptors the engine polls for this rank's progress.
    ///
    /// A shared-memory channel multiplexes every wake onto one listener. A
    /// socket carries only the result wake, so the local descriptor its
    /// command and death wakes fire on is polled beside it.
    pub fn progress_fds(&self) -> Vec<i32> {
        match self {
            Self::Shared(client) => vec![client.wake_file_descriptor()],
            Self::Socket(client) => vec![client.wake_file_descriptor(), client.local_wake_fd()],
        }
    }
}
