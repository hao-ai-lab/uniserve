//! Immutable ownership of generated POSIX shared-memory artifacts.

use axum::body::Bytes;
use std::ffi::CString;
use uniserve_core::ArtifactEvent;

pub(crate) struct SharedMedia {
    address: *mut libc::c_void,
    pub(crate) bytes: usize,
}

// The mapping is immutable after publication and remains valid until the final Arc drops.
unsafe impl Send for SharedMedia {}
unsafe impl Sync for SharedMedia {}

impl SharedMedia {
    /// Claims and maps a generated shared-memory artifact for response streaming.
    pub(crate) fn open(artifact: &ArtifactEvent) -> Result<Self, String> {
        let bytes = usize::try_from(artifact.bytes)
            .map_err(|_| "generated media is too large for this host".to_string())?;
        if bytes == 0
            || artifact.artifact.posix_shm_name().is_empty()
            || artifact.artifact.posix_shm_name().contains('/')
        {
            return Err("generated media has an invalid shared-memory locator".to_string());
        }
        let name = CString::new(format!("/{}", artifact.artifact.posix_shm_name()))
            .map_err(|_| "generated media has an invalid shared-memory name".to_string())?;
        // SAFETY: name is a valid NUL-terminated POSIX shm name.
        let descriptor = unsafe { libc::shm_open(name.as_ptr(), libc::O_RDONLY, 0) };
        if descriptor < 0 {
            return Err(format!(
                "failed to open generated media shared memory: {}",
                std::io::Error::last_os_error()
            ));
        }
        // The response is the sole consumer. Claim the object as soon as it is open; the
        // descriptor keeps the bytes alive across inspection and mapping failures.
        // SAFETY: name identifies the object opened above.
        if unsafe { libc::shm_unlink(name.as_ptr()) } != 0 {
            let error = std::io::Error::last_os_error();
            // SAFETY: descriptor is open.
            unsafe { libc::close(descriptor) };
            return Err(format!(
                "failed to claim generated media shared memory: {error}"
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
                "failed to inspect generated media shared memory: {error}"
            ));
        }
        // SAFETY: fstat initialized stat on success.
        let extent = unsafe { stat.assume_init() }.st_size;
        if extent < 0
            || u64::try_from(extent)
                .ok()
                .is_none_or(|value| value < artifact.bytes)
        {
            // SAFETY: descriptor is open.
            unsafe { libc::close(descriptor) };
            return Err("generated media shared memory is shorter than its locator".to_string());
        }
        // SAFETY: descriptor names a readable shared-memory object of at least `bytes` bytes.
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
                "failed to map generated media shared memory: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(Self { address, bytes })
    }

    /// Wraps bytes as an HTTP response-body frame.
    pub(crate) fn chunk(&self, offset: usize, count: usize) -> Bytes {
        // SAFETY: caller bounds offset/count to the mapping extent and the mapping is immutable.
        let value =
            unsafe { std::slice::from_raw_parts((self.address as *const u8).add(offset), count) };
        Bytes::copy_from_slice(value)
    }
}

impl Drop for SharedMedia {
    /// Releases resources owned by this value.
    fn drop(&mut self) {
        // SAFETY: address is the live mapping created in `open` with exactly this extent.
        unsafe { libc::munmap(self.address, self.bytes) };
    }
}
