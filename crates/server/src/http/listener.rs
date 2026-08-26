//! Unified HTTP listener wrapper for the Rust frontend.
//!
//! This module hides the difference between TCP and Unix-domain listeners so
//! the rest of the server can bind or inherit one socket and pass it to
//! `axum::serve(...)` through a single type.

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
    /// Bind or adopt the listener described by the frontend configuration.
    ///
    /// For inherited sockets, the concrete listener kind is detected from the
    /// socket family of the supplied file descriptor.
    pub(crate) async fn bind(mode: &HttpListenerMode) -> Result<Self> {
        match mode {
            HttpListenerMode::BindTcp { host, port } => {
                Ok(Self::Tcp(TcpListener::bind((host.as_str(), *port)).await?))
            }
            HttpListenerMode::BindUnix { path } => Ok(Self::Unix(UnixListener::bind(path)?)),
            HttpListenerMode::InheritedFd { fd } => Self::from_inherited_fd(*fd),
        }
    }

    /// Return a log-friendly local address string for either TCP or Unix
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

        // The Python supervisor pre-binds the socket to reserve the endpoint early, but
        // Rust is responsible for transitioning inherited stream sockets into
        // the listening state before accepting connections.
        socket.listen(libc::SOMAXCONN)?;
        socket.set_nonblocking(true)?;

        if socket.local_addr()?.is_unix() {
            let std_listener = unsafe { StdUnixListener::from_raw_fd(socket.into_raw_fd()) };
            Ok(Self::Unix(UnixListener::from_std(std_listener)?))
        } else {
            let std_listener = unsafe { StdTcpListener::from_raw_fd(socket.into_raw_fd()) };
            Ok(Self::Tcp(TcpListener::from_std(std_listener)?))
        }
    }
}

/// Allow the unified listener to plug directly into `axum::serve(...)`.
impl axum::serve::Listener for Listener {
    type Addr = Either<std::net::SocketAddr, tokio::net::unix::SocketAddr>;
    type Io = Either<TcpStream, UnixStream>;

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
