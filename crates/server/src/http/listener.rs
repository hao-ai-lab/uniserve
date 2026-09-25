//! Listener abstraction for TCP, Unix-domain, and inherited sockets.
//!
//! [`Listener`] presents a single accept interface to Axum over either a TCP or
//! a Unix-domain listener, obtained according to `HttpListenerMode`.
//! `crate::http::serve` binds it, logs its inherent `local_addr` string, and
//! wraps it with `tap_io` to enable `TCP_NODELAY` on TCP connections only.
//!
//! A Unix-domain socket bound at a path leaves its file behind when the socket
//! closes. The listener therefore owns the file of a socket it binds: binding
//! first removes a stale socket file nobody listens on, and dropping the
//! listener (at shutdown) removes the file again.

use std::io::{ErrorKind, Result};
use std::net::TcpListener as StdTcpListener;
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
use std::os::unix::fs::{FileTypeExt as _, MetadataExt as _};
use std::os::unix::net::UnixListener as StdUnixListener;
use std::path::{Path, PathBuf};

use socket2::Socket;
use tokio::net::{TcpListener, TcpStream, UnixListener, UnixStream};
use tokio_util::either::Either;

use crate::HttpListenerMode;

/// Runtime listener type used by the OpenAI-compatible HTTP server, which is
/// either a TCP listener or a Unix-domain listener.
#[derive(Debug)]
pub(crate) enum Listener {
    Tcp(TcpListener),
    Unix {
        listener: UnixListener,
        /// The socket's file when this process bound it at a path; `None` for
        /// an inherited socket, whose file belongs to its supplier.
        _socket_file: Option<SocketFile>,
    },
}

/// The filesystem entry of a Unix-domain socket this process bound, removed
/// when dropped.
///
/// The entry is identified by device and inode, so removal is skipped when the
/// path no longer names the bound socket, for example after the file was
/// deleted and another server bound the same path.
#[derive(Debug)]
pub(crate) struct SocketFile {
    path: PathBuf,
    device: u64,
    inode: u64,
}

impl Drop for SocketFile {
    fn drop(&mut self) {
        let bound = std::fs::symlink_metadata(&self.path)
            .is_ok_and(|metadata| metadata.dev() == self.device && metadata.ino() == self.inode);
        if bound {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

impl Listener {
    /// Binds or adopts the listener described by the frontend configuration.
    ///
    /// For inherited sockets, the concrete listener kind is detected from the
    /// socket family of the supplied file descriptor. `BindUnix` first removes
    /// a stale socket file at its path (see `remove_stale_socket`); it fails
    /// with `EADDRINUSE` when the path holds a live server's socket or a file
    /// that is not a socket. The bound socket's file is removed when the
    /// listener is dropped.
    pub(crate) async fn bind(mode: &HttpListenerMode) -> Result<Self> {
        match mode {
            HttpListenerMode::BindTcp { host, port } => {
                Ok(Self::Tcp(TcpListener::bind((host.as_str(), *port)).await?))
            }
            HttpListenerMode::BindUnix { path } => Self::bind_unix(Path::new(path)).await,
            HttpListenerMode::InheritedFd { fd } => Self::from_inherited_fd(*fd),
        }
    }

    /// Binds a Unix-domain listener at `path` and takes ownership of its file.
    async fn bind_unix(path: &Path) -> Result<Self> {
        remove_stale_socket(path).await?;
        let listener = UnixListener::bind(path)?;
        let metadata = std::fs::symlink_metadata(path)?;
        Ok(Self::Unix {
            listener,
            _socket_file: Some(SocketFile {
                path: path.to_owned(),
                device: metadata.dev(),
                inode: metadata.ino(),
            }),
        })
    }

    /// Returns a log-friendly local address string for either TCP or Unix
    /// sockets.
    pub(crate) fn local_addr(&self) -> Result<String> {
        match self {
            Self::Tcp(listener) => Ok(listener.local_addr()?.to_string()),
            Self::Unix { listener, .. } => Ok(match listener.local_addr()?.as_pathname() {
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
            Ok(Self::Unix {
                listener: UnixListener::from_std(std_listener)?,
                _socket_file: None,
            })
        } else {
            let std_listener = unsafe { StdTcpListener::from_raw_fd(socket.into_raw_fd()) };
            Ok(Self::Tcp(TcpListener::from_std(std_listener)?))
        }
    }
}

/// Removes the socket file at `path` when no server listens on it.
///
/// A socket whose connection attempt is refused was left behind by a server
/// that exited without removing it, such as a crashed or killed one. Anything
/// else at `path` (a live server's socket, a file that is not a socket, or a
/// socket that cannot be probed) is left in place for `bind` to refuse.
async fn remove_stale_socket(path: &Path) -> Result<()> {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return Ok(());
    };
    if !metadata.file_type().is_socket() {
        return Ok(());
    }
    let refused = UnixStream::connect(path)
        .await
        .is_err_and(|error| error.kind() == ErrorKind::ConnectionRefused);
    if refused {
        match std::fs::remove_file(path) {
            Err(error) if error.kind() != ErrorKind::NotFound => return Err(error),
            _ => {}
        }
    }
    Ok(())
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
            Self::Unix { listener, .. } => {
                let (io, addr) = listener.accept().await;
                (Either::Right(io), Either::Right(addr))
            }
        }
    }

    /// Returns the listener socket address.
    fn local_addr(&self) -> Result<Self::Addr> {
        match self {
            Self::Tcp(listener) => listener.local_addr().map(Either::Left),
            Self::Unix { listener, .. } => listener.local_addr().map(Either::Right),
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

        assert!(matches!(listener, Listener::Unix { .. }));
        let _ = std::fs::remove_file(path);
    }

    /// A socket file left by a server that exited without unlinking it (a
    /// crash, or a kill) does not prevent a restart on the same path.
    #[tokio::test(flavor = "current_thread")]
    async fn a_stale_socket_file_is_replaced() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("uniserve.sock");
        drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
        assert!(path.exists());

        let listener = Listener::bind(&HttpListenerMode::BindUnix {
            path: path.to_str().unwrap().to_owned(),
        })
        .await
        .unwrap();

        assert!(matches!(listener, Listener::Unix { .. }));
        std::os::unix::net::UnixStream::connect(&path).unwrap();
    }

    /// Dropping a bound Unix listener, as a clean shutdown does, removes its
    /// socket file.
    #[tokio::test(flavor = "current_thread")]
    async fn dropping_a_unix_listener_removes_its_socket_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("uniserve.sock");
        let mode = HttpListenerMode::BindUnix {
            path: path.to_str().unwrap().to_owned(),
        };

        let listener = Listener::bind(&mode).await.unwrap();
        assert!(path.exists());
        drop(listener);

        assert!(!path.exists());
    }

    /// Only a socket nobody listens on is replaced: a live server's socket and
    /// a file that is not a socket are left in place and the bind fails.
    #[tokio::test(flavor = "current_thread")]
    async fn a_live_socket_or_another_file_is_not_replaced() {
        let directory = tempfile::tempdir().unwrap();

        let live = directory.path().join("live.sock");
        let server = std::os::unix::net::UnixListener::bind(&live).unwrap();
        let error = Listener::bind(&HttpListenerMode::BindUnix {
            path: live.to_str().unwrap().to_owned(),
        })
        .await
        .expect_err("a live server's socket must not be taken over");
        assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
        std::os::unix::net::UnixStream::connect(&live).unwrap();
        drop(server);

        let file = directory.path().join("data.sock");
        std::fs::write(&file, b"not a socket").unwrap();
        let error = Listener::bind(&HttpListenerMode::BindUnix {
            path: file.to_str().unwrap().to_owned(),
        })
        .await
        .expect_err("a regular file must not be replaced");
        assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
        assert_eq!(std::fs::read(&file).unwrap(), b"not a socket");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn inherited_fd_rejects_negative_fd() {
        let err = Listener::bind(&HttpListenerMode::InheritedFd { fd: -1 })
            .await
            .expect_err("a negative fd must be rejected, not adopted");
        assert_eq!(err.raw_os_error(), Some(libc::EBADF));
    }
}
