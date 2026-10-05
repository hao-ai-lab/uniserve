//! Shared host tensor storage and stream-ordered readiness.

use std::io;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use uniserve_core::SharedMemory;

use crate::cuda::{self, DeviceGuard, Event, Stream};
use crate::{Error, Result};

pub const SHM_HEADER_BYTES: usize = 512;
const ACK_OFFSET: usize = 64;
const ACK_SLOTS: usize = 64;
const PENDING: u32 = 0;
const READY: u32 = 1;
const FAILED: u32 = 2;
const CLAIMED: u32 = 1;
const ACKNOWLEDGED: u32 = 2;

/// A mapping can outlive its producer while a borrowed tensor view or a CUDA
/// host function still uses it. It owns no CUDA resources: its final release
/// is safe on a CUDA callback thread.
pub struct SharedMapping {
    address: usize,
    size: usize,
}

impl SharedMapping {
    pub fn open(name: &str, size: usize) -> io::Result<Self> {
        let file = SharedMemory::open(name, true)?;
        if file.metadata()?.len() < size as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "shared buffer is shorter than its locator",
            ));
        }

        Self::new(&file, size)
    }

    fn new(file: &std::fs::File, size: usize) -> io::Result<Self> {
        // SAFETY: storage is physically backed for the full writable extent.
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            address: address as usize,
            size,
        })
    }

    pub fn address(&self) -> usize {
        self.address
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn word(&self, offset: usize) -> &AtomicU32 {
        // SAFETY: only aligned header offsets within this mapping are used.
        // Both processes access these words atomically; payload writes start
        // at SHM_HEADER_BYTES and cannot overlap them.
        unsafe { &*((self.address + offset) as *const AtomicU32) }
    }
}

impl Drop for SharedMapping {
    fn drop(&mut self) {
        // SAFETY: this is the last owner of this live mapping.
        unsafe { libc::munmap(self.address as *mut _, self.size()) };
    }
}

/// Owns a producer's segment and optional CUDA host registration. The caller
/// keeps the device stream alive through close and retains this buffer until
/// all granted readers have acknowledged it.
pub struct SharedBuffer {
    // Some tensor consumers retain only the exporter, not a Py_buffer lease.
    // Keep this reference even after close has unlinked the allocation.
    mapping: Arc<SharedMapping>,
    storage: Option<SharedMemory>,
    consumers: Vec<usize>,
    device: Option<(i32, usize)>,
    bytes: usize,
    submitted: bool,
    completion: Option<Event>,
}

impl SharedBuffer {
    pub fn new(
        bytes: usize,
        consumers: Vec<usize>,
        device: Option<(i32, usize)>,
    ) -> std::result::Result<Self, String> {
        if consumers.iter().any(|slot| *slot >= ACK_SLOTS) {
            return Err("shared buffer acknowledgment slot is out of range".into());
        }

        let size = SHM_HEADER_BYTES
            .checked_add(bytes.max(1))
            .ok_or("shared buffer is too large")?;
        let (storage, file) = SharedMemory::create(size).map_err(|error| error.to_string())?;
        let mapping = Arc::new(SharedMapping::new(&file, size).map_err(|error| error.to_string())?);

        // Fresh POSIX storage is zero-filled: readiness is PENDING and all
        // reader slots are unclaimed before any locator can escape.
        if let Some((device, _)) = device {
            let _device = DeviceGuard::new(device)?;
            // SAFETY: the mapping owns this full range through unregister.
            unsafe { cuda::register_host(mapping.address(), mapping.size())? };
        }

        Ok(Self {
            mapping,
            storage: Some(storage),
            consumers,
            device,
            bytes,
            submitted: false,
            completion: None,
        })
    }

    pub fn mapping(&self) -> std::result::Result<Arc<SharedMapping>, String> {
        self.name()?;
        Ok(Arc::clone(&self.mapping))
    }

    pub fn name(&self) -> std::result::Result<&str, String> {
        self.storage
            .as_ref()
            .map(SharedMemory::name)
            .ok_or_else(|| "shared buffer is closed".into())
    }

    pub fn nbytes(&self) -> usize {
        self.bytes
    }

    pub fn is_cuda(&self) -> bool {
        self.device.is_some()
    }

    /// Announce bytes after their last write, either immediately for a CPU
    /// copy or in the producer's CUDA stream. The callback touches only host
    /// memory; it neither enters Python nor calls CUDA.
    /// The caller submits exactly one copy sequence per buffer and must not
    /// modify its payload after announcing readiness.
    pub fn mark_ready(&mut self) -> std::result::Result<(), String> {
        let mapping = self.mapping()?;

        if let Some((device, stream)) = self.device {
            let _device = DeviceGuard::new(device)?;
            cuda::launch(stream, move || {
                mapping.word(0).store(READY, Ordering::Release)
            })?;

            let completion = Event::new(device, false, false);
            completion.record(stream)?;
            self.completion = Some(completion);
            Ok(())
        } else {
            mapping.word(0).store(READY, Ordering::Release);
            Ok(())
        }
    }

    /// Declare possible DMA before the numerical backend submits its first
    /// copy. A failed submission may still have enqueued earlier spans.
    pub fn begin_copy(&mut self) {
        self.submitted = true;
    }

    pub fn settled(&self) -> bool {
        let mapping = &self.mapping;
        self.storage.is_none()
            || (mapping.word(0).load(Ordering::Acquire) != PENDING
                && self.consumers.iter().all(|slot| {
                    mapping
                        .word(ACK_OFFSET + slot * size_of::<u32>())
                        .load(Ordering::Acquire)
                        != CLAIMED
                }))
    }

    /// Shutdown and failed submission must finish DMA even if readiness was
    /// never scheduled. Normal retirement observes READY and does not wait.
    pub fn synchronize(&self) -> std::result::Result<(), String> {
        if self.submitted
            && let Some((device, stream)) = self.device
            && self.mapping.word(0).load(Ordering::Acquire) == PENDING
        {
            let _device = DeviceGuard::new(device)?;
            if let Some(completion) = &self.completion {
                completion.wait()?;
            } else {
                // Submission failed before its completion event was recorded.
                Stream::borrowed(stream).wait()?;
            }
        }

        Ok(())
    }

    pub fn close(&mut self) -> std::result::Result<(), String> {
        if self.storage.is_none() {
            return Ok(());
        }

        self.synchronize()?;
        if let Some((device, _)) = self.device {
            let _device = DeviceGuard::new(device)?;
            // SAFETY: producer DMA has stopped and the mapping is still live.
            unsafe { cuda::unregister_host(self.mapping.address())? };
        }

        self.device = None;
        self.storage.take();
        Ok(())
    }
}

impl Drop for SharedBuffer {
    #[expect(
        clippy::mem_forget,
        reason = "unknown device completion must retain its registered mapping"
    )]
    fn drop(&mut self) {
        if self.close().is_err() {
            std::mem::forget(Arc::clone(&self.mapping));
            std::mem::forget(self.storage.take());
        }
    }
}

/// A claimed payload range on the reader's host. Its mapping stays alive
/// through borrowed tensor views; release acknowledges the end of reading.
pub struct SharedRead {
    mapping: Arc<SharedMapping>,
    slot: Option<usize>,
    offset: usize,
    bytes: usize,
}

impl SharedRead {
    /// Map and claim one byte range. The caller must wait for readiness
    /// before accessing the payload and end all reads before releasing it.
    pub fn open(name: &str, offset: usize, bytes: usize, slot: usize) -> Result<Self> {
        if slot >= ACK_SLOTS {
            return Err(Error::Invalid(
                "shared buffer acknowledgment slot is out of range".into(),
            ));
        }
        let offset = SHM_HEADER_BYTES
            .checked_add(offset)
            .ok_or_else(|| Error::Invalid("shared read offset is too large".into()))?;
        let size = offset
            .checked_add(bytes)
            .ok_or_else(|| Error::Invalid("shared read is too large".into()))?;
        let mapping = SharedMapping::open(name, size).map_err(|error| match error.kind() {
            io::ErrorKind::NotFound => {
                Error::Invalid("shared buffer is no longer available".into())
            }
            io::ErrorKind::InvalidInput => Error::Invalid(error.to_string()),
            _ => Error::Transport(error.to_string()),
        })?;
        mapping
            .word(ACK_OFFSET + slot * size_of::<u32>())
            .store(CLAIMED, Ordering::Release);

        Ok(Self {
            mapping: Arc::new(mapping),
            slot: Some(slot),
            offset,
            bytes,
        })
    }

    /// Poll shared host memory without interpreter calls. `check` observes
    /// the consuming transfer's native cancellation state between polls.
    /// Failure or timeout leaves the claim held until release or destruction.
    pub fn wait(&self, timeout: Duration, mut check: impl FnMut() -> Result<()>) -> Result<()> {
        let started = Instant::now();
        let mut pause = Duration::ZERO;

        loop {
            check()?;
            match self.mapping.word(0).load(Ordering::Acquire) {
                READY => return Ok(()),
                FAILED => return Err(Error::Resource("export producer failed before readiness")),
                _ => {}
            }
            if started.elapsed() >= timeout {
                return Err(Error::Resource("export endpoint was lost before readiness"));
            }

            if !pause.is_zero() {
                std::thread::sleep(pause);
            }
            pause = (pause + Duration::from_micros(50)).min(Duration::from_millis(1));
        }
    }

    pub fn mapping(&self) -> Result<Arc<SharedMapping>> {
        if self.slot.is_none() {
            return Err(Error::State("shared read is released"));
        }
        Ok(Arc::clone(&self.mapping))
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn nbytes(&self) -> usize {
        self.bytes
    }

    /// Limit a padded media row to its actual frame bytes before encoding.
    pub fn truncate(&mut self, bytes: usize) -> Result<()> {
        if bytes > self.bytes {
            return Err(Error::Invalid(
                "shared read exceeds its borrowed range".into(),
            ));
        }
        self.bytes = bytes;
        Ok(())
    }

    pub fn release(&mut self) {
        if let Some(slot) = self.slot.take() {
            self.mapping
                .word(ACK_OFFSET + slot * size_of::<u32>())
                .store(ACKNOWLEDGED, Ordering::Release);
        }
    }
}

impl Drop for SharedRead {
    fn drop(&mut self) {
        self.release();
    }
}
