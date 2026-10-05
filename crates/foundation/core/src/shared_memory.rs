//! Owned, physically backed POSIX shared-memory allocations.

use std::ffi::CString;
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd};
use std::sync::atomic::{AtomicU64, Ordering};

/// Owns a shared-memory name. Open descriptors and mappings retain the kernel
/// allocation independently; dropping this owner prevents subsequent opens.
#[derive(Debug)]
pub struct SharedMemory {
    name: String,
    size: usize,
}

impl SharedMemory {
    /// Open an existing segment without taking responsibility for unlinking.
    pub fn open(name: &str, writable: bool) -> io::Result<File> {
        let name = CString::new(format!("/{}", name.trim_start_matches('/')))?;
        let access = if writable {
            libc::O_RDWR
        } else {
            libc::O_RDONLY
        };

        // SAFETY: name is NUL-terminated; a successful call returns an owned fd.
        let descriptor = unsafe { libc::shm_open(name.as_ptr(), access | libc::O_CLOEXEC, 0) };
        if descriptor < 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(unsafe { File::from_raw_fd(descriptor) })
    }

    /// Allocate physical storage and return its writable descriptor. The
    /// caller may close the descriptor as soon as writing or mapping ends.
    pub fn create(size: usize) -> io::Result<(Self, File)> {
        static NEXT: AtomicU64 = AtomicU64::new(0);

        if size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "shared memory must not be empty",
            ));
        }

        let length = libc::off_t::try_from(size).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "shared memory is too large")
        })?;
        let name = format!(
            "uniserve-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let path = CString::new(format!("/{name}"))?;

        // SAFETY: path is a NUL-terminated name; the returned fd is owned here.
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
        // SAFETY: shm_open returned one new descriptor owned by this call.
        let file = unsafe { File::from_raw_fd(descriptor) };
        let storage = Self { name, size };

        // Reserve tmpfs pages before exposing a writable mapping. A sparse
        // allocation can otherwise raise SIGBUS during a later mapped write.
        // posix_fallocate returns its error number directly, without errno.
        let error = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, length) };
        if error != 0 {
            return Err(io::Error::from_raw_os_error(error));
        }

        Ok((storage, file))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Transfer unlink responsibility to a receiving process.
    pub fn into_name(mut self) -> String {
        std::mem::take(&mut self.name)
    }
}

impl Drop for SharedMemory {
    fn drop(&mut self) {
        if !self.name.is_empty()
            && let Ok(path) = CString::new(format!("/{}", self.name))
        {
            // SAFETY: this owner created the NUL-terminated POSIX name.
            unsafe { libc::shm_unlink(path.as_ptr()) };
        }
    }
}
