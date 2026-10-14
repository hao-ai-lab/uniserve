//! Grants a process's shareable descriptors to consumer processes on its host.
//!
//! A device whose driver exports POSIX file descriptors rather than fabric
//! handles publishes an allocation as a descriptor, which names an open file
//! of the producing process and means nothing in another one. A consumer
//! receives it through `SCM_RIGHTS` on a Unix-domain `SOCK_SEQPACKET` socket:
//! it sends one [`REQUEST_BYTES`]-byte publication id and receives one record,
//! [`GRANTED`] carrying the descriptor or [`REFUSED`] without one.
//!
//! [`DescriptorGrants`] serves those requests on a native thread that reads
//! no interpreter state. A producer's own device work can wait on the very
//! consumers that ask it for descriptors (a collective they have not reached
//! yet), and its Python threads may all be blocked meanwhile, for instance a
//! main thread that holds the GIL in a CUDA launch waiting for queue space; a
//! grant that needed the interpreter would then deadlock the deployment.

use std::collections::HashMap;
use std::ffi::c_void;
use std::io;
use std::mem;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

/// Bytes of one request: a publication id, 32 ASCII characters.
pub const REQUEST_BYTES: usize = 32;
/// Reply byte of a request whose publication is registered; the record
/// carries its descriptor.
pub const GRANTED: u8 = 1;
/// Reply byte of a request for an unregistered or withdrawn publication.
pub const REFUSED: u8 = 0;
/// Bound on receiving one request and sending its reply.
const CONNECTION_TIMEOUT: libc::timeval = libc::timeval {
    tv_sec: 30,
    tv_usec: 0,
};
/// Pending connections the listener holds; a batch of products is read by
/// every rank of a deployment at once.
const BACKLOG: i32 = 64;

type PublicationId = [u8; REQUEST_BYTES];

/// Serves one address space's publication descriptors to local consumers.
///
/// Descriptors are registered as the producer publishes and withdrawn as its
/// publications retire. A registered descriptor stays owned by whoever opened
/// it; the table only lends it out. Requests are served one connection at a
/// time on the server thread, and every method may be called from any thread.
pub struct DescriptorGrants {
    table: Arc<Mutex<HashMap<PublicationId, RawFd>>>,
    address: PathBuf,
    // Readable once `close` asks the server thread, which owns the listener,
    // to stop.
    stop: OwnedFd,
    server: Mutex<Option<JoinHandle<()>>>,
}

impl DescriptorGrants {
    /// Binds the grant socket at `address` and starts serving it.
    ///
    /// The parent directory is created when missing. Binding fails when a
    /// socket already exists at `address`.
    pub fn bind(address: &Path) -> io::Result<Self> {
        if let Some(parent) = address.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let listener = listen(address)?;
        // SAFETY: `eventfd` has no memory preconditions; a negative result is
        // checked before the descriptor is owned.
        let stop = match owned(unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) })
        {
            Ok(stop) => stop,
            Err(error) => {
                let _ = std::fs::remove_file(address);
                return Err(error);
            }
        };
        let table = Arc::new(Mutex::new(HashMap::new()));

        let server = {
            let table = Arc::clone(&table);
            let stop = stop.as_raw_fd();
            std::thread::Builder::new()
                .name("uniserve-descriptor-grants".to_owned())
                .spawn(move || serve(&listener, stop, &table))
        };
        let server = match server {
            Ok(server) => server,
            Err(error) => {
                let _ = std::fs::remove_file(address);
                return Err(error);
            }
        };
        Ok(Self {
            table,
            address: address.to_path_buf(),
            stop,
            server: Mutex::new(Some(server)),
        })
    }

    /// Makes one publication's descriptor grantable.
    ///
    /// # Errors
    ///
    /// `InvalidInput` when `publication_id` is not [`REQUEST_BYTES`] bytes.
    pub fn register(&self, publication_id: &str, descriptor: RawFd) -> io::Result<()> {
        lock(&self.table).insert(publication_key(publication_id)?, descriptor);
        Ok(())
    }

    /// Withdraws a publication; later requests for it are refused.
    ///
    /// On return no grant of its descriptor is still being sent, so the
    /// owner may close the descriptor.
    pub fn release(&self, publication_id: &str) {
        if let Ok(key) = publication_key(publication_id) {
            lock(&self.table).remove(&key);
        }
    }

    /// Stops serving, joins the server thread and removes the socket.
    ///
    /// Registered descriptors are forgotten, not closed. Closing again does
    /// nothing.
    pub fn close(&self) {
        let Some(server) = lock(&self.server).take() else {
            return;
        };
        lock(&self.table).clear();
        let signal = 1_u64.to_ne_bytes();
        // SAFETY: `stop` is an open eventfd owned by `self`; an eventfd
        // accepts one 8-byte counter increment.
        let _ = unsafe { libc::write(self.stop.as_raw_fd(), signal.as_ptr().cast(), signal.len()) };
        let _ = server.join();
        let _ = std::fs::remove_file(&self.address);
    }
}

impl Drop for DescriptorGrants {
    fn drop(&mut self) {
        self.close();
    }
}

/// Accepts and answers connections until `stop` becomes readable.
///
/// `stop` stays open until this returns: `close` joins the thread before the
/// descriptor drops.
fn serve(listener: &OwnedFd, stop: RawFd, table: &Mutex<HashMap<PublicationId, RawFd>>) {
    loop {
        let mut descriptors = [
            libc::pollfd {
                fd: listener.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: stop,
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        // SAFETY: `descriptors` holds two initialized entries for the call.
        let ready = unsafe { libc::poll(descriptors.as_mut_ptr(), 2, -1) };
        if ready < 0 {
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return;
        }
        if descriptors[1].revents != 0 || descriptors[0].revents & libc::POLLIN == 0 {
            return;
        }

        // The listener is non-blocking, so a connection withdrawn between the
        // poll and the accept costs one more poll.
        // SAFETY: null address arguments ask `accept4` for no peer address.
        let connection = unsafe {
            libc::accept4(
                listener.as_raw_fd(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                libc::SOCK_CLOEXEC,
            )
        };
        let Ok(connection) = owned(connection) else {
            continue;
        };
        // A consumer that disconnects or times out costs only its own
        // request; the listener keeps serving.
        let _ = reply(&connection, table);
    }
}

/// Answers one request with its publication's descriptor or a refusal.
fn reply(connection: &OwnedFd, table: &Mutex<HashMap<PublicationId, RawFd>>) -> io::Result<()> {
    let timeout = CONNECTION_TIMEOUT;
    for option in [libc::SO_RCVTIMEO, libc::SO_SNDTIMEO] {
        // SAFETY: the option value is a `timeval` of the size passed.
        check(unsafe {
            libc::setsockopt(
                connection.as_raw_fd(),
                libc::SOL_SOCKET,
                option,
                (&raw const timeout).cast(),
                mem::size_of::<libc::timeval>() as libc::socklen_t,
            )
        })?;
    }

    let mut request: PublicationId = [0; REQUEST_BYTES];
    // SAFETY: `request` is writable for its full length.
    let received = unsafe {
        libc::recv(
            connection.as_raw_fd(),
            request.as_mut_ptr().cast(),
            REQUEST_BYTES,
            0,
        )
    };
    check(received.try_into().unwrap_or(-1))?;

    // The table stays locked until the descriptor is in flight: once
    // `sendmsg` returns the kernel holds its own reference, so a `release`
    // that returned has no grant of the descriptor still being sent.
    let table = lock(table);
    let descriptor = usize::try_from(received)
        .is_ok_and(|length| length == REQUEST_BYTES)
        .then(|| table.get(&request).copied())
        .flatten();
    match descriptor {
        Some(descriptor) => send_descriptor(connection, descriptor),
        None => send_byte(connection, REFUSED),
    }
}

/// Sends [`GRANTED`] with `descriptor` attached as `SCM_RIGHTS`.
fn send_descriptor(connection: &OwnedFd, descriptor: RawFd) -> io::Result<()> {
    let payload = [GRANTED];
    let mut buffer = libc::iovec {
        iov_base: payload.as_ptr().cast_mut().cast::<c_void>(),
        iov_len: payload.len(),
    };
    // A control buffer of `u64` words satisfies `cmsghdr` alignment.
    // SAFETY: `CMSG_SPACE` only computes a size.
    let space = unsafe { libc::CMSG_SPACE(mem::size_of::<RawFd>() as u32) } as usize;
    let mut control = vec![0_u64; space.div_ceil(mem::size_of::<u64>())];
    // SAFETY: an all-zero `msghdr` is a valid empty message, filled below.
    let mut message: libc::msghdr = unsafe { mem::zeroed() };
    message.msg_iov = &raw mut buffer;
    message.msg_iovlen = 1;
    message.msg_control = control.as_mut_ptr().cast();
    message.msg_controllen = space as _;
    // SAFETY: the control buffer holds one complete header for one
    // descriptor, so `CMSG_FIRSTHDR` and `CMSG_DATA` stay inside it.
    unsafe {
        let header = libc::CMSG_FIRSTHDR(&raw const message);
        (*header).cmsg_level = libc::SOL_SOCKET;
        (*header).cmsg_type = libc::SCM_RIGHTS;
        (*header).cmsg_len = libc::CMSG_LEN(mem::size_of::<RawFd>() as u32) as _;
        libc::CMSG_DATA(header)
            .cast::<RawFd>()
            .write_unaligned(descriptor);
    }
    // SAFETY: `message` refers to `buffer`, `payload` and `control`, which
    // outlive the call.
    let sent = unsafe {
        libc::sendmsg(
            connection.as_raw_fd(),
            &raw const message,
            libc::MSG_NOSIGNAL,
        )
    };
    check(sent.try_into().unwrap_or(-1))
}

fn send_byte(connection: &OwnedFd, byte: u8) -> io::Result<()> {
    // SAFETY: the one-byte buffer is readable for the call.
    let sent = unsafe {
        libc::send(
            connection.as_raw_fd(),
            (&raw const byte).cast(),
            1,
            libc::MSG_NOSIGNAL,
        )
    };
    check(sent.try_into().unwrap_or(-1))
}

/// Creates a non-blocking `SOCK_SEQPACKET` listener bound at `address`.
fn listen(address: &Path) -> io::Result<OwnedFd> {
    // SAFETY: `socket` has no memory preconditions.
    let listener = owned(unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
            0,
        )
    })?;

    // SAFETY: an all-zero `sockaddr_un` is valid and filled below.
    let mut name: libc::sockaddr_un = unsafe { mem::zeroed() };
    name.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let path = address.as_os_str().as_bytes();
    if path.len() >= name.sun_path.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "grant socket path {} exceeds the socket name length",
                address.display()
            ),
        ));
    }
    for (slot, byte) in name.sun_path.iter_mut().zip(path) {
        *slot = *byte as libc::c_char;
    }
    // SAFETY: `name` is a valid `sockaddr_un` of the size passed.
    check(unsafe {
        libc::bind(
            listener.as_raw_fd(),
            (&raw const name).cast(),
            mem::size_of::<libc::sockaddr_un>() as libc::socklen_t,
        )
    })?;
    // SAFETY: `listener` is a bound socket.
    check(unsafe { libc::listen(listener.as_raw_fd(), BACKLOG) })?;
    Ok(listener)
}

fn publication_key(publication_id: &str) -> io::Result<PublicationId> {
    publication_id.as_bytes().try_into().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("a publication id has {REQUEST_BYTES} bytes, got {publication_id:?}"),
        )
    })
}

fn owned(descriptor: RawFd) -> io::Result<OwnedFd> {
    check(descriptor)?;
    // SAFETY: a non-negative descriptor was just returned to this process and
    // is owned by nothing else.
    Ok(unsafe { OwnedFd::from_raw_fd(descriptor) })
}

fn check(result: i32) -> io::Result<()> {
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // The guarded values hold no invariant a panicking holder could break.
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
