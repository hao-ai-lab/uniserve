//! Listener abstraction for TCP, Unix-domain, and inherited sockets.
//!
//! [`Listener`] presents a single accept interface to Axum over either a TCP or
//! a Unix-domain listener, obtained according to `HttpListenerMode`.
//! `crate::http::serve` binds it, logs its inherent `local_addr` string, and
//! wraps it with `tap_io` to enable `TCP_NODELAY` on TCP connections only.

use std::io::Result;
use std::net::TcpListener as StdTcpListener;
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
use std::os::unix::net::UnixListener as StdUnixListener;

use socket2::Socket;
use tokio::net::{TcpListener, TcpStream, UnixListener, UnixStream};
use tokio_util::either::Either;

use crate::HttpListenerMode;

/// Runtime listener type used by the OpenAI-compatible HTTP server, which is
/// either a TCP listener or a Unix-domain listener.
#[derive(Debug)]
pub(crate) enum Listener {
    Tcp(TcpListener),
    Unix(UnixListener),
}

impl Listener {
    /// Binds or adopts the listener described by the frontend configuration.
    ///
    /// For inherited sockets, the concrete listener kind is detected from the
    /// socket family of the supplied file descriptor. `BindUnix` fails when a
    /// file already exists at the socket path.
    pub(crate) async fn bind(mode: &HttpListenerMode) -> Result<Self> {
        match mode {
            HttpListenerMode::BindTcp { host, port } => {
                Ok(Self::Tcp(TcpListener::bind((host.as_str(), *port)).await?))
            }
            HttpListenerMode::BindUnix { path } => Ok(Self::Unix(UnixListener::bind(path)?)),
            HttpListenerMode::InheritedFd { fd } => Self::from_inherited_fd(*fd),
        }
    }

    /// Returns a log-friendly local address string for either TCP or Unix
    /// sockets.
    pub(crate) fn local_addr(&self) -> Result<String> {
        match self {
            Self::Tcp(listener) => Ok(listener.local_addr()?.to_string()),
            Self::Unix(listener) => Ok(match listener.local_addr()?.as_pathname() {
                Some(path) => format!("unix:{}", path.display()),
                None => "unix:<unnamed>".to_string(),
            }),
        }
    }

    /// Adopts an inherited stream socket and prepares it for asynchronous accepts.
    ///
    /// The descriptor must be a bound stream socket; it need not be listening
    /// yet. Returns `EBADF` for a negative descriptor, and the underlying error
    /// when the descriptor is not open or when `listen`, `set_nonblocking`,
    /// the address lookup, or Tokio registration fails. Once the descriptor is
    /// adopted, any later failure closes it.
    fn from_inherited_fd(fd: i32) -> Result<Self> {
        // Validate the raw integer before taking ownership of it. `OwnedFd` assumes
        // the fd is open and will `close(2)` it on drop, so handing it a negative or
        // already-closed descriptor is undefined behavior and could close an unrelated
        // fd. A non-negative value plus a successful `fcntl(F_GETFD)` probe confirms the
        // descriptor is open before we adopt it.
        if fd < 0 {
            return Err(std::io::Error::from_raw_os_error(libc::EBADF));
        }
        if unsafe { libc::fcntl(fd, libc::F_GETFD) } < 0 {
            return Err(std::io::Error::last_os_error());
        }

        // SAFETY: `fd` was just validated as a non-negative, open descriptor, and we
        // take ownership of it exactly once to create a single listener.
        let owned_fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let socket = Socket::from(owned_fd);

        // The supplier may hand over a socket that is only bound, so this side
        // puts it into the listening state. Tokio's `from_std` requires the
        // socket to be non-blocking already.
        socket.listen(libc::SOMAXCONN)?;
        socket.set_nonblocking(true)?;

        // Any non-Unix family is treated as TCP. `into_raw_fd` releases the
        // `Socket`'s ownership, so the std listener becomes the only owner.
        if socket.local_addr()?.is_unix() {
            let std_listener = unsafe { StdUnixListener::from_raw_fd(socket.into_raw_fd()) };
            Ok(Self::Unix(UnixListener::from_std(std_listener)?))
        } else {
            let std_listener = unsafe { StdTcpListener::from_raw_fd(socket.into_raw_fd()) };
            Ok(Self::Tcp(TcpListener::from_std(std_listener)?))
        }
    }
}

/// Allows the unified listener to plug directly into `axum::serve(...)`.
impl axum::serve::Listener for Listener {
    type Addr = Either<std::net::SocketAddr, tokio::net::unix::SocketAddr>;
    type Io = Either<TcpStream, UnixStream>;

    /// Accepts the next incoming connection.
    ///
    /// Each arm resolves to Axum's `Listener::accept` for the Tokio listener
    /// rather than Tokio's fallible inherent `accept`. Axum retries accept
    /// errors internally (logging and pausing on errors other than
    /// per-connection ones), so this future resolves only with a connection.
    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self {
            Self::Tcp(listener) => {
                let (io, addr) = listener.accept().await;
                (Either::Left(io), Either::Left(addr))
            }
            Self::Unix(listener) => {
                let (io, addr) = listener.accept().await;
                (Either::Right(io), Either::Right(addr))
            }
        }
    }

    /// Returns the listener socket address.
    fn local_addr(&self) -> Result<Self::Addr> {
        match self {
            Self::Tcp(listener) => listener.local_addr().map(Either::Left),
            Self::Unix(listener) => listener.local_addr().map(Either::Right),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, SocketAddrV4};
    use std::os::fd::IntoRawFd;

    use socket2::{Domain, SockAddr, Socket, Type};
    use uuid::Uuid;

    use super::Listener;
    use crate::HttpListenerMode;

    // The inherited sockets in the family-detection tests are bound but not
    // yet listening; `Listener::bind` must start listening on them and detect
    // their family.
    #[tokio::test(flavor = "current_thread")]
    async fn inherited_fd_detects_tcp_listener_without_uds_hint() {
        let socket = Socket::new(Domain::IPV4, Type::STREAM, None).unwrap();
        socket
            .bind(&SockAddr::from(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)))
            .unwrap();
        let fd = socket.into_raw_fd();

        let listener = Listener::bind(&HttpListenerMode::InheritedFd { fd })
            .await
            .unwrap();

        assert!(matches!(listener, Listener::Tcp(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn inherited_fd_detects_unix_listener_from_fd() {
        let path = std::env::temp_dir().join(format!("uniserve-{}.sock", Uuid::new_v4()));
        let socket = Socket::new(Domain::UNIX, Type::STREAM, None).unwrap();
        socket.bind(&SockAddr::unix(&path).unwrap()).unwrap();
        let fd = socket.into_raw_fd();

        let listener = Listener::bind(&HttpListenerMode::InheritedFd { fd })
            .await
            .unwrap();

        assert!(matches!(listener, Listener::Unix(_)));
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn inherited_fd_rejects_negative_fd() {
        let err = Listener::bind(&HttpListenerMode::InheritedFd { fd: -1 })
            .await
            .expect_err("a negative fd must be rejected, not adopted");
        assert_eq!(err.raw_os_error(), Some(libc::EBADF));
    }
}
