//! Media bytes handed between processes through POSIX shared storage.
//!
//! Generated artifacts travel from a worker to the engine: a model worker
//! publishes finished media bytes through [`MediaSource`] and reports its
//! name as an `ArtifactHandle::PosixShm` in the call result. The engine claims
//! it with [`SharedMedia::open`] when it receives the batch result
//! (`WorkerResult::receive`), and the resulting `Arc<SharedMedia>` reaches the
//! server through the engine's output events. Opening unlinks the name, so a
//! later open of the same name fails and the kernel frees the storage when the
//! last reference to it goes away.
//!
//! Condition media travel the other way: the server publishes each fetched
//! condition as a [`MediaSource`], which keeps the name until it drops, and
//! the worker's media reader opens it by name.

use std::ffi::CString;
use std::fs::File;
use std::io::{self, Write};
use std::os::fd::{AsRawFd, FromRawFd};

/// Read-only mapping of one claimed shared-memory media publication.
///
/// The mapping covers exactly the published byte count and is unmapped on
/// drop. Share it with `Arc` rather than reopening: the name no longer exists
/// once it has been claimed.
#[derive(Debug)]
pub struct SharedMedia {
    address: *mut libc::c_void,
    bytes: usize,
}

// The pointer refers to a `PROT_READ` mapping that no process writes after
// publication (the `open` contract) and that stays mapped until this value
// drops, so moving or sharing it across threads cannot race.
unsafe impl Send for SharedMedia {}
unsafe impl Sync for SharedMedia {}

impl SharedMedia {
    /// Claims an immutable POSIX shared-storage object and maps its published extent.
    ///
    /// `name` is the object name without its leading `/`, and `num_bytes` is
    /// the published length; the object may be longer, and only the first
    /// `num_bytes` bytes are mapped.
    ///
    /// Opening transfers ownership: the name is unlinked immediately, and bytes
    /// remain readable until this mapping is dropped. Once the name has been
    /// unlinked, a later failure (inspection, a short object, or the mapping)
    /// still leaves the storage released. Failures before that point (a zero
    /// or oversized length, an empty name or one containing `/` or NUL, a
    /// failed open, or a failed unlink) leave any named object in place.
    ///
    /// # Errors
    ///
    /// Returns a message describing the first failed step.
    ///
    /// # Safety
    ///
    /// The publisher must relinquish all writes and resizing before this call,
    /// and no process may modify the object while the mapping remains live.
    pub unsafe fn open(name: &str, num_bytes: u64) -> Result<Self, String> {
        let bytes = usize::try_from(num_bytes)
            .map_err(|_| "generated media is too large for this host".to_string())?;

        // Reject a malformed locator before touching any object: the name must
        // be one non-empty component (the leading `/` is added below), and a
        // publication is never empty.
        if bytes == 0 || name.is_empty() || name.contains('/') {
            return Err("generated media has an invalid shared-storage locator".to_string());
        }
        let name = CString::new(format!("/{}", name))
            .map_err(|_| "generated media has an invalid shared-storage name".to_string())?;

        // SAFETY: name is a valid NUL-terminated POSIX shm name.
        let descriptor = unsafe { libc::shm_open(name.as_ptr(), libc::O_RDONLY, 0) };
        if descriptor < 0 {
            return Err(format!(
                "failed to open generated media shared storage: {}",
                std::io::Error::last_os_error()
            ));
        }

        // Unlink the name as soon as the object is open so no other consumer can
        // claim it and the storage is freed with its last reference even if a
        // later step fails. The descriptor, and then the mapping, keeps the
        // bytes alive.
        // SAFETY: name identifies the object opened above.
        if unsafe { libc::shm_unlink(name.as_ptr()) } != 0 {
            let error = std::io::Error::last_os_error();
            // SAFETY: descriptor is open.
            unsafe { libc::close(descriptor) };
            return Err(format!(
                "failed to claim generated media shared storage: {error}"
            ));
        }

        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: descriptor is open and stat points to writable storage.
        let stat_result = unsafe { libc::fstat(descriptor, stat.as_mut_ptr()) };
        if stat_result != 0 {
            let error = std::io::Error::last_os_error();
            // SAFETY: descriptor is open.
            unsafe { libc::close(descriptor) };
            return Err(format!(
                "failed to inspect generated media shared storage: {error}"
            ));
        }
        // SAFETY: fstat initialized stat on success.
        let extent = unsafe { stat.assume_init() }.st_size;

        // A publication shorter than its locator is incomplete; reject it
        // rather than map past the object's end.
        if extent < 0
            || u64::try_from(extent)
                .ok()
                .is_none_or(|value| value < num_bytes)
        {
            // SAFETY: descriptor is open.
            unsafe { libc::close(descriptor) };
            return Err("generated media shared storage is shorter than its locator".to_string());
        }

        // SAFETY: descriptor names a readable shared-storage object of at least `bytes` bytes.
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ,
                libc::MAP_SHARED,
                descriptor,
                0,
            )
        };
        // SAFETY: this scope exclusively owns the valid descriptor; the mapping retains its
        // kernel object independently after the descriptor is closed.
        unsafe { libc::close(descriptor) };
        if address == libc::MAP_FAILED {
            return Err(format!(
                "failed to map generated media shared storage: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(Self { address, bytes })
    }

    /// Number of published bytes owned by this mapping.
    pub fn len(&self) -> usize {
        self.bytes
    }

    /// Whether the published extent is empty; successful mappings are nonempty.
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Borrows the immutable published extent for validation or response output.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: the mapping remains live for this borrow and is immutable after publication.
        unsafe { std::slice::from_raw_parts(self.address.cast(), self.bytes) }
    }
}

impl AsRef<[u8]> for SharedMedia {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl Drop for SharedMedia {
    /// Unmaps the published extent. The name was unlinked at open, so the
    /// kernel frees the storage once no process maps it or holds it open.
    fn drop(&mut self) {
        // SAFETY: address is the live mapping created in `open` with exactly this extent.
        unsafe { libc::munmap(self.address, self.bytes) };
    }
}

/// Media bytes this process published, under a fresh name, as a POSIX
/// shared-memory object for other processes on this host to read.
///
/// The server publishes each fetched condition this way; the worker's media
/// reader opens the object by name, reads it and never unlinks it. The
/// publisher keeps the name: dropping the value unlinks it, and the kernel
/// frees the storage once no reader still maps it. The engine holds a
/// request's publications until the request retires, after its last call.
#[derive(Debug)]
pub struct MediaSource {
    name: String,
    bytes: u64,
}

impl MediaSource {
    /// Copies `bytes` into a new shared-memory object.
    ///
    /// The object is created exclusively, readable and writable by its owner
    /// only, and physically reserved to exactly `bytes.len()` bytes.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when `bytes` is empty or storage allocation or
    /// writing fails; no object is left behind.
    pub fn publish(bytes: &[u8]) -> io::Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};

        // Names are unique per process; the process id separates processes.
        static NEXT: AtomicU64 = AtomicU64::new(0);

        if bytes.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "published media must not be empty",
            ));
        }
        let length = libc::off_t::try_from(bytes.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "published media is too large")
        })?;
        let name = format!(
            "uniserve-media-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let path = CString::new(format!("/{name}"))?;

        // SAFETY: path is a valid NUL-terminated POSIX shm name.
        let descriptor = unsafe {
            libc::shm_open(
                path.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_RDWR | libc::O_CLOEXEC,
                0o600,
            )
        };
        if descriptor < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: shm_open returned an open descriptor owned by this call.
        let mut file = unsafe { File::from_raw_fd(descriptor) };

        // From here the object exists; constructing the owner first unlinks
        // it on every failure below.
        let source = Self {
            name,
            bytes: bytes.len() as u64,
        };

        // Reserve physical pages before publication. ftruncate alone can
        // leave a sparse tmpfs object whose later mapped writes raise SIGBUS.
        // posix_fallocate returns its error number directly, without errno.
        // SAFETY: file owns a writable descriptor and length is positive.
        let error = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, length) };
        if error != 0 {
            return Err(io::Error::from_raw_os_error(error));
        }
        file.write_all(bytes)?;
        Ok(source)
    }

    /// The object's name without its leading `/` and its byte count.
    pub fn locator(&self) -> crate::MediaLocator {
        crate::MediaLocator {
            name: self.name.clone(),
            bytes: self.bytes,
        }
    }

    /// Hand the named object to its receiving process. The receiver must
    /// claim it with `SharedMedia::open`, which unlinks it on receipt.
    /// Dropping this publisher afterwards leaves the name available.
    pub fn into_locator(mut self) -> crate::MediaLocator {
        crate::MediaLocator {
            name: std::mem::take(&mut self.name),
            bytes: self.bytes,
        }
    }
}

impl PartialEq for MediaSource {
    /// Two publications are equal when they name the same object.
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for MediaSource {}

impl Drop for MediaSource {
    /// Unlinks the name; readers that still map the object keep its bytes.
    fn drop(&mut self) {
        if self.name.is_empty() {
            return;
        }

        if let Ok(path) = CString::new(format!("/{}", self.name)) {
            // SAFETY: path is a valid NUL-terminated POSIX shm name.
            unsafe { libc::shm_unlink(path.as_ptr()) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A publication is readable by name while it lives and unlinked once
    /// it drops.
    #[test]
    fn a_published_source_is_readable_until_dropped() {
        let source = MediaSource::publish(b"condition bytes").unwrap();
        let locator = source.locator();
        assert_eq!(locator.bytes, 15);
        let path = CString::new(format!("/{}", locator.name)).unwrap();

        // SAFETY: path is a valid NUL-terminated POSIX shm name.
        let descriptor = unsafe { libc::shm_open(path.as_ptr(), libc::O_RDONLY, 0) };
        assert!(descriptor >= 0);
        let mut read = vec![0_u8; 15];
        // SAFETY: descriptor is open and `read` holds 15 writable bytes.
        let count = unsafe { libc::read(descriptor, read.as_mut_ptr().cast(), 15) };
        // SAFETY: descriptor is open.
        unsafe { libc::close(descriptor) };
        assert_eq!(count, 15);
        assert_eq!(read, b"condition bytes");

        drop(source);
        // SAFETY: path is a valid NUL-terminated POSIX shm name.
        let reopened = unsafe { libc::shm_open(path.as_ptr(), libc::O_RDONLY, 0) };
        assert!(reopened < 0, "the name outlived its publisher");
    }

    #[test]
    fn empty_media_is_refused() {
        assert!(MediaSource::publish(b"").is_err());
    }
}
