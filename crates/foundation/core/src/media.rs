//! Immutable ownership of generated POSIX shared-storage artifacts.
//!
//! A model worker publishes finished media bytes as a POSIX shared-memory
//! object (`publish_media_bytes` in `uniserve_worker.media.storage`) and
//! reports its name as an `ArtifactHandle::PosixShm` in the call result. The
//! engine claims it with [`SharedMedia::open`] when it receives the batch
//! result (`WorkerResult::receive`), and the resulting `Arc<SharedMedia>`
//! reaches the server through the engine's output events.
//!
//! Opening unlinks the name, so a later open of the same name fails and the
//! kernel frees the storage when the last reference to it goes away.

use std::ffi::CString;

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
