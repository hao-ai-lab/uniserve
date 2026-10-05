//! Local CUDA allocation descriptors passed through Unix sockets.

use std::collections::HashMap;
use std::io::{self, IoSlice, IoSliceMut};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, FromRawFd, OwnedFd, RawFd};
use std::os::linux::net::SocketAddrExt;
use std::os::unix::net::{SocketAddr, UnixDatagram, UnixStream};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rustix::event::{PollFd, PollFlags, poll};
use rustix::net::{
    RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags, SendAncillaryBuffer,
    SendAncillaryMessage, SendFlags, SocketAddrUnix, bind, recvmsg, sendmsg_addr,
};

const REQUEST_BYTES: usize = 32;
const TIMEOUT: Duration = Duration::from_secs(30);

type Descriptors = Arc<Mutex<Option<HashMap<String, Arc<OwnedFd>>>>>;

/// Grants registered allocation descriptors to consumers in this host's
/// network namespace. Abstract socket names require no shared directory.
///
/// Registration duplicates the descriptor. Revocation removes that owned
/// reference, while an in-flight reply retains its own borrow until sendmsg
/// has passed a reference to the receiver. No Python callbacks are involved.
pub struct DescriptorGrants {
    descriptors: Descriptors,
    worker: Mutex<Option<(UnixStream, JoinHandle<io::Result<()>>)>>,
}

impl DescriptorGrants {
    pub fn new(endpoint: &str) -> io::Result<Self> {
        let socket = UnixDatagram::bind_addr(&address(endpoint)?)?;
        socket.set_nonblocking(true)?;
        let (shutdown, stop) = UnixStream::pair()?;
        let descriptors = Arc::new(Mutex::new(Some(HashMap::new())));
        let registered = Arc::clone(&descriptors);
        let worker = thread::Builder::new()
            .name("worker-descriptor-grants".into())
            .spawn(move || serve(socket, stop, registered))?;

        Ok(Self {
            descriptors,
            worker: Mutex::new(Some((shutdown, worker))),
        })
    }

    /// Retain an allocation until revocation. The caller keeps ownership of fd.
    pub fn register(&self, publication: &str, fd: RawFd) -> io::Result<()> {
        request(publication)?;

        // SAFETY: fcntl validates the caller's descriptor and returns a new
        // descriptor owned by this registration, with close-on-exec set.
        let duplicate = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
        if duplicate < 0 {
            return Err(io::Error::last_os_error());
        }

        // SAFETY: fcntl returned this open descriptor with sole ownership.
        let descriptor = unsafe { OwnedFd::from_raw_fd(duplicate) };
        let mut registered = self.descriptors.lock().unwrap_or_else(|e| e.into_inner());
        let table = registered.as_mut().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotConnected, "descriptor grants closed")
        })?;
        table.insert(publication.to_owned(), Arc::new(descriptor));
        Ok(())
    }

    /// Refuse later requests; already received descriptors remain usable.
    pub fn release(&self, publication: &str) {
        let mut registered = self.descriptors.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(table) = registered.as_mut() {
            table.remove(publication);
        }
    }

    /// Revoke registrations and join the service thread, including when idle.
    pub fn close(&self) -> io::Result<()> {
        let mut worker = self.worker.lock().unwrap_or_else(|e| e.into_inner());
        let Some((shutdown, handle)) = worker.take() else {
            return Ok(());
        };

        *self.descriptors.lock().unwrap_or_else(|e| e.into_inner()) = None;

        // Closing the stream wakes poll with POLLHUP. No peer request or
        // timeout is needed to stop an otherwise idle descriptor service.
        drop(shutdown);
        handle
            .join()
            .map_err(|_| io::Error::other("descriptor service thread panicked"))?
    }
}

impl Drop for DescriptorGrants {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Receive an owned descriptor. The caller closes it after importing the
/// allocation; CUDA's import retains its own allocation reference.
pub fn fetch_descriptor(endpoint: &str, publication: &str) -> io::Result<OwnedFd> {
    let request = request(publication)?;
    let socket = UnixDatagram::unbound()?;
    socket.set_read_timeout(Some(TIMEOUT))?;
    socket.set_write_timeout(Some(TIMEOUT))?;

    // Linux autobind assigns the reply socket a unique abstract address when
    // bind receives only sa_family_t. Its lifetime follows this socket.
    bind(&socket, &SocketAddrUnix::new_unnamed())?;
    socket.connect_addr(&address(endpoint)?)?;
    socket.send(request)?;
    receive(&socket)
}

fn address(endpoint: &str) -> io::Result<SocketAddr> {
    SocketAddr::from_abstract_name(format!("uniserve-grants-{endpoint}"))
}

fn request(publication: &str) -> io::Result<&[u8]> {
    if publication.len() != REQUEST_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "descriptor requests require a 32-byte publication name",
        ));
    }
    Ok(publication.as_bytes())
}

fn serve(socket: UnixDatagram, stop: UnixStream, descriptors: Descriptors) -> io::Result<()> {
    let mut ready = [
        PollFd::new(&stop, PollFlags::IN),
        PollFd::new(&socket, PollFlags::IN),
    ];

    loop {
        if let Err(error) = poll(&mut ready, None) {
            if error == rustix::io::Errno::INTR {
                continue;
            }
            return Err(error.into());
        }
        if !ready[0].revents().is_empty() {
            return Ok(());
        }

        // The extra byte distinguishes oversized requests from a full name.
        let mut name = [0; REQUEST_BYTES + 1];
        let (count, peer) = match socket.recv_from(&mut name) {
            Ok(received) => received,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                continue;
            }
            Err(error) => return Err(error),
        };
        let descriptor = if count == REQUEST_BYTES {
            let registered = descriptors.lock().unwrap_or_else(|e| e.into_inner());
            std::str::from_utf8(&name[..count])
                .ok()
                .and_then(|name| registered.as_ref()?.get(name).cloned())
        } else {
            None
        };

        // A disconnected or unreadable consumer affects only its own grant.
        // Nonblocking replies cannot hold up independent consumers or close.
        let _ = reply(&socket, &peer, descriptor.as_deref());
    }
}

fn reply(socket: &UnixDatagram, peer: &SocketAddr, descriptor: Option<&OwnedFd>) -> io::Result<()> {
    let Some(name) = peer.as_abstract_name() else {
        return Ok(());
    };
    let address = SocketAddrUnix::new_abstract_name(name)?;
    let granted = [u8::from(descriptor.is_some())];
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
    let mut control = SendAncillaryBuffer::new(&mut space);
    let descriptors;

    if let Some(descriptor) = descriptor {
        descriptors = [descriptor.as_fd()];
        control.push(SendAncillaryMessage::ScmRights(&descriptors));
    }

    sendmsg_addr(
        socket,
        &address,
        &[IoSlice::new(&granted)],
        &mut control,
        SendFlags::NOSIGNAL,
    )?;
    Ok(())
}

fn receive(socket: &UnixDatagram) -> io::Result<OwnedFd> {
    let mut granted = [0_u8];
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
    let mut control = RecvAncillaryBuffer::new(&mut space);
    let message = recvmsg(
        socket,
        &mut [IoSliceMut::new(&mut granted)],
        &mut control,
        RecvFlags::CMSG_CLOEXEC,
    )?;

    // The ancillary owner closes every unconsumed descriptor on error, even
    // when a reply carries more than the single descriptor requested.
    if message.bytes == 1
        && granted[0] == 1
        && !message
            .flags
            .intersects(ReturnFlags::TRUNC | ReturnFlags::CTRUNC)
    {
        let mut received = control
            .drain()
            .filter_map(|message| match message {
                RecvAncillaryMessage::ScmRights(descriptors) => Some(descriptors),
                _ => None,
            })
            .flatten();
        if let (Some(descriptor), None) = (received.next(), received.next()) {
            return Ok(descriptor);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "allocation has no live descriptor grant",
    ))
}
