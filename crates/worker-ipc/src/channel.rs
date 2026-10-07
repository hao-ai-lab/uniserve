//! Both ends of one rank's channel, over either transport.
//!
//! A rank on the head's host offers a shared-storage service ([`crate::iceoryx`]);
//! a rank elsewhere offers a socket ([`crate::socket`]). The engine addresses
//! both the same way: it sends a request under an identity and takes the
//! response that carries it back.
//!
//! The head decides the mechanism from the placement and passes it to the rank
//! in its launch descriptor (`channel_transport`); the rank binds a
//! [`RankServer`] for it, names the endpoint, and reports both at registration.
//! The engine's registration refuses any mechanism other than
//! [`SHARED_STORAGE_CHANNEL`] and [`SOCKET_CHANNEL`], and `PendingRank::adopt`
//! then connects a [`RankChannel`] to whatever the report named. On the rank
//! side the PyO3 `Server` in `worker-ipc-py` owns the [`RankServer`].
//!
//! The two mechanisms differ in ways the enums do not hide: a shared-storage
//! request can be sent before any server exists (see
//! [`RankChannel::is_connected`]), queue-full and inflight-limit failures
//! surface as different errors, and only the shared-storage server checks a
//! response's identity against the requests it received.

use std::sync::Arc;
use std::time::Duration;

use crate::iceoryx::{
    ClientEndpoint, Frame, IpcError, IpcResult, Pending, ServerEndpoint, WakeEvents, WakeSender,
};
use crate::request::WorkerRequest;
use crate::socket::{SocketClient, SocketServer};

/// The mechanism name of a channel over iceoryx2 shared storage, served by a
/// rank on the head's host.
///
/// The name travels as a string in the launch descriptor's `channel_transport`
/// and in the registration report's `transport`.
pub const SHARED_STORAGE_CHANNEL: &str = "iceoryx2";
/// The mechanism name of a channel over a TCP stream, served by a rank off the
/// head's host.
pub const SOCKET_CHANNEL: &str = "tcp";

/// One rank's report of the channel endpoint the engine connects to.
///
/// The rank sends one JSON line after binding its endpoint and before loading
/// model weights. The head uses it to connect the rank channel.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct RankReport {
    /// Worker identity the rank was launched under.
    pub worker_id: String,
    /// Global rank within that worker.
    pub rank: u32,
    /// Channel mechanism the endpoint names.
    pub transport: String,
    /// Endpoint the engine binds this rank's channel to.
    pub endpoint: String,
}

/// A wake raised inside one process for that process's own service loop,
/// whichever transport its channel uses.
///
/// Two sources use it: the engine's `DeathWatcher` fires a rank channel's
/// [`RankChannel::death_wake`], and a rank's completion callbacks fire
/// [`RankServer::completion_wake`]. Neither travels between processes. A
/// shared-storage channel fires one of its own iceoryx2 notifiers; a socket
/// channel writes to a local descriptor its owner polls beside the socket.
/// On both, wakes fired before the owner drains them coalesce into one
/// notification: a shared-storage channel's senders share one pending bit,
/// and a socket channel's drain reads every byte its senders wrote.
#[derive(Clone)]
pub enum Wake {
    /// Fires the shared-storage channel's notifier.
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
    /// A retained shared-storage request handle.
    Shared(Pending),
    /// The identity a socket response will carry.
    ///
    /// Outstanding socket requests need distinct identities: the socket client
    /// keys received responses by identity, so a later response under the
    /// same identity replaces an untaken one.
    Socket(u64),
}

/// The engine's end of one rank's channel.
pub enum RankChannel {
    /// A rank on the head's host, reached through shared storage.
    Shared(Box<ClientEndpoint>),
    /// A rank elsewhere, reached over a stream socket.
    Socket(Box<SocketClient>),
}

impl RankChannel {
    /// Binds the channel a rank named, refusing a mechanism the head cannot bind.
    ///
    /// `max_payload` bounds every frame's payload in bytes. `depth` is the
    /// number of outstanding request handles for shared storage and the number
    /// of queued outgoing frames for a socket. `timeout` bounds only the TCP
    /// connect; a shared-storage connect does not wait for the rank's server.
    pub fn connect(
        transport: &str,
        endpoint: &str,
        max_payload: usize,
        depth: usize,
        timeout: Duration,
    ) -> IpcResult<Self> {
        match transport {
            SHARED_STORAGE_CHANNEL => Ok(Self::Shared(Box::new(ClientEndpoint::connect(
                endpoint,
                max_payload,
                depth,
            )?))),
            SOCKET_CHANNEL => Ok(Self::Socket(Box::new(SocketClient::connect(
                endpoint,
                max_payload,
                depth,
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
    ///
    /// A full socket send queue fails with [`IpcError::WouldBlock`]. An
    /// exhausted shared-storage request limit fails with
    /// [`IpcError::Transport`] instead, so `RankProcess` compares its own
    /// outstanding count with the depth before submitting a batch. A
    /// shared-storage success does not mean a server took the request; check
    /// [`Self::is_connected`] with the returned handle.
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
    ///
    /// Returns `Ok(None)` at the deadline; the socket form also returns it once
    /// the channel has failed while other responses still wait. Wakes observed
    /// while waiting, including a worker-death wake, are consumed without being
    /// reported, so a caller that tracks rank death must not rely on this
    /// method to surface it.
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
    /// A shared-storage request is loaned before a server exists to take it, so
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
    /// A shared-storage channel multiplexes its result and death wakes onto
    /// one listener. A socket carries only results, so the local descriptor
    /// its death wake fires on is polled beside it; the socket's own entry
    /// also requests writability while queued bytes remain unsent. After any
    /// descriptor polls ready, call [`Self::drain_wakes`].
    pub fn progress_fds(&self) -> Vec<libc::pollfd> {
        match self {
            Self::Shared(client) => vec![libc::pollfd {
                fd: client.wake_file_descriptor(),
                events: libc::POLLIN,
                revents: 0,
            }],
            Self::Socket(client) => client.progress_fds(),
        }
    }
}

/// A rank's end of its channel, over either transport.
///
/// A rank on the head's host serves a shared-storage endpoint; a rank elsewhere
/// serves a socket. The head states which mechanism the placement calls for and
/// the rank names the endpoint, so the rank creates whichever one it was told
/// to offer and reports the name it chose.
pub enum RankServer {
    /// A shared-storage endpoint on the head's host.
    Shared(Box<ServerEndpoint>),
    /// A socket endpoint reachable from another host.
    Socket(Box<SocketServer>),
}

impl RankServer {
    /// Creates the endpoint this rank will report.
    ///
    /// `name` is the service name for a shared-storage endpoint and the
    /// interface to bind for a socket, which binds a system-chosen port. It is
    /// the one argument whose meaning differs between the mechanisms.
    /// `max_inflight` bounds outstanding requests for shared storage and
    /// queued response frames for a socket. A socket bind does not wait for
    /// the engine to connect.
    pub fn bind(
        transport: &str,
        name: &str,
        max_payload: usize,
        max_inflight: usize,
    ) -> IpcResult<Self> {
        match transport {
            SHARED_STORAGE_CHANNEL => Ok(Self::Shared(Box::new(ServerEndpoint::bind(
                name,
                max_payload,
                max_inflight,
            )?))),
            SOCKET_CHANNEL => Ok(Self::Socket(Box::new(SocketServer::bind(
                name,
                max_payload,
                max_inflight,
            )?))),
            other => Err(IpcError::Transport(format!(
                "a rank cannot serve the channel transport {other}"
            ))),
        }
    }

    /// Returns the endpoint the rank reports for the engine to bind.
    ///
    /// A shared-storage endpoint returns `service` unchanged, so the caller
    /// passes the name it bound. A socket endpoint returns its bound address,
    /// whose host is the bind interface and may be a wildcard;
    /// `uniserve_worker.bootstrap.launch.register_endpoint` keeps its port and
    /// replaces the host with an address the head can route to.
    pub fn endpoint(&self, service: &str) -> String {
        match self {
            Self::Shared(_) => service.to_string(),
            Self::Socket(server) => server.address().to_string(),
        }
    }

    /// Returns the mechanism the rank reports alongside its endpoint.
    pub fn transport(&self) -> &'static str {
        match self {
            Self::Shared(_) => SHARED_STORAGE_CHANNEL,
            Self::Socket(_) => SOCKET_CHANNEL,
        }
    }

    /// Takes one request, if the engine has sent one.
    pub fn try_recv(&mut self) -> IpcResult<Option<Frame>> {
        match self {
            Self::Shared(server) => server.try_recv(),
            Self::Socket(server) => {
                // The engine connects once, after the rank has reported; the
                // first receive is where the rank notices it has arrived.
                server.accept_if_pending()?;
                server.try_recv()
            }
        }
    }

    /// Waits until a request is readable, a completion fires, or time passes.
    ///
    /// Returns `Ok(())` on a wake and on the timeout alike without saying
    /// which occurred, and may consume a pending completion wake, so the
    /// caller re-checks every progress source afterwards.
    pub fn wait_incoming(&mut self, timeout: Duration) -> IpcResult<()> {
        match self {
            Self::Shared(server) => server.wait_incoming(timeout),
            Self::Socket(server) => {
                server.accept_if_pending()?;
                server.wait_incoming(timeout)
            }
        }
    }

    /// Returns the sender this rank's completion callbacks fire.
    pub fn completion_wake(&self) -> Wake {
        match self {
            Self::Shared(server) => Wake::Shared(server.completion_wake()),
            Self::Socket(server) => Wake::Local(server.completion_wake()),
        }
    }

    /// Waits for one request, however long it takes.
    ///
    /// Returns only with a frame or an error; there is no overall deadline.
    /// Both forms park in bounded steps between receive attempts.
    pub fn recv(&mut self) -> IpcResult<Frame> {
        match self {
            Self::Shared(server) => server.recv(),
            Self::Socket(server) => loop {
                server.accept_if_pending()?;
                if let Some(frame) = server.try_recv()? {
                    return Ok(frame);
                }
                server.wait_incoming(Duration::from_millis(50))?;
            },
        }
    }

    /// Answers one request, encoding the response for the channel.
    pub fn respond(&mut self, response: &crate::request::WorkerResponse) -> IpcResult<()> {
        match self {
            Self::Shared(server) => server.respond(response),
            Self::Socket(server) => {
                let header = crate::iceoryx::header_for_response(response);
                let payload = crate::codec::encode_response(response)?;
                server.respond_raw(header, &payload)
            }
        }
    }

    /// Answers one request under the identity it carried.
    ///
    /// `header.len` is stamped from `payload`. The shared-storage server fails
    /// when `header.message_id` matches no request it has received and not yet
    /// answered; the socket server sends whatever identity it is given, so the
    /// caller owns that correlation there.
    pub fn respond_raw(&mut self, header: crate::iceoryx::Header, payload: &[u8]) -> IpcResult<()> {
        match self {
            Self::Shared(server) => server.respond_raw(header, payload),
            Self::Socket(server) => server.respond_raw(header, payload),
        }
    }
}
