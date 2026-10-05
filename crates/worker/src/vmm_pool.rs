//! Bounded export ranges and asynchronous observation of remote readers.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use crate::cuda::{self, CopyBatch, DeviceGuard, Event, PinnedBuffer, Stream};
use crate::{DescriptorGrants, Error, Result};

pub const ACK_SLOTS: usize = 64;
pub const HEADER_BYTES: usize = ACK_SLOTS * size_of::<u32>();
pub const CHUNK_ALIGNMENT: usize = 512;
const CLAIMED: u32 = 1;

/// An immutable byte range. The payload follows the remote-reader header.
pub struct PoolChunk {
    pub offset: usize,
    pub nbytes: usize,
    pub payload_bytes: usize,
}

impl PoolChunk {
    pub fn payload_offset(&self) -> usize {
        self.offset + HEADER_BYTES
    }
}

struct Allocation {
    chunk: Arc<PoolChunk>,
    // None denotes a live export; Some selects readers after revocation.
    readers: Option<u64>,
    grant: Option<(Arc<DescriptorGrants>, String)>,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if let Some((grants, export)) = &self.grant {
            grants.release(export);
        }
    }
}

/// Own chunk placement and retirement within one device's exported mapping.
///
/// The numerical backend supplies the mapping and retains its allocation in
/// `backing`. Reader words are copied on a private stream. Polling observes a
/// completed snapshot and never waits for device work on the host.
pub struct VmmPool<B> {
    backing: Option<B>,
    device: i32,
    address: usize,
    capacity: usize,
    watermark: usize,
    allocations: BTreeMap<usize, Allocation>,
    initializers: HashSet<usize>,
    stream: Stream,
    completion: Event,
    observed: Vec<usize>,
    readback: Vec<PinnedBuffer>,
    readback_slot: usize,
    failed: bool,
}

impl<B> VmmPool<B> {
    /// The mapping must cover `capacity` writable bytes on `device` and remain
    /// valid as long as `backing` is held. Construction performs no device copy.
    ///
    /// # Safety
    /// `address` must identify that mapping. All producer streams passed to
    /// reserve remain valid through close, which drains header initialization.
    pub unsafe fn new(backing: B, device: i32, address: usize, capacity: usize) -> Result<Self> {
        let _device = DeviceGuard::new(device).map_err(Error::Cuda)?;
        Ok(Self {
            backing: Some(backing),
            device,
            address,
            capacity,
            watermark: 0,
            allocations: BTreeMap::new(),
            initializers: HashSet::new(),
            stream: Stream::new().map_err(Error::Cuda)?,
            completion: Event::new(device, false, false),
            observed: Vec::new(),
            readback: Vec::new(),
            readback_slot: 0,
            failed: false,
        })
    }

    pub fn backing(&self) -> Result<&B> {
        self.backing
            .as_ref()
            .ok_or(Error::State("VMM pool is closed"))
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Reserve a payload and clear its reader words on the producer stream.
    /// None reports capacity exhaustion without reserving any bytes.
    /// Keep the borrowed stream alive until the pool closes.
    pub fn reserve(&mut self, bytes: usize, stream: usize) -> Result<Option<Arc<PoolChunk>>> {
        self.backing()?;
        if self.failed {
            return Err(Error::Resource("VMM pool device completion is unknown"));
        }
        if bytes == 0 {
            return Err(Error::Invalid(
                "a pool chunk needs a positive length".into(),
            ));
        }
        let span = bytes
            .checked_add(HEADER_BYTES)
            .and_then(|value| value.checked_add(CHUNK_ALIGNMENT - 1))
            .map(|value| value / CHUNK_ALIGNMENT * CHUNK_ALIGNMENT)
            .ok_or_else(|| Error::Invalid("VMM payload is too large".into()))?;
        let Some(offset) = self.free_offset(span) else {
            return Ok(None);
        };
        let chunk = Arc::new(PoolChunk {
            offset,
            nbytes: span,
            payload_bytes: bytes,
        });
        self.allocations.insert(
            offset,
            Allocation {
                chunk: Arc::clone(&chunk),
                readers: None,
                grant: None,
            },
        );

        let _device = DeviceGuard::new(self.device).map_err(Error::Cuda)?;
        self.initializers.insert(stream);
        // SAFETY: this reservation owns all 64 aligned header words. Later
        // payload copies and the export's readiness event use this same stream.
        if let Err(error) = unsafe { cuda::fill_words(self.address + offset, 0, ACK_SLOTS, stream) }
        {
            self.failed = true;
            return Err(Error::Cuda(error));
        }
        Ok(Some(chunk))
    }

    fn free_offset(&mut self, span: usize) -> Option<usize> {
        if span <= self.capacity - self.watermark {
            let offset = self.watermark;
            self.watermark += span;
            return Some(offset);
        }

        let mut cursor = 0;
        for (&offset, allocation) in &self.allocations {
            if span <= offset - cursor {
                return Some(cursor);
            }
            cursor = offset + allocation.chunk.nbytes;
        }
        if span <= self.capacity - cursor {
            self.watermark = self.watermark.max(cursor + span);
            Some(cursor)
        } else {
            None
        }
    }

    /// Return a range whose accesses the caller has already drained. Releasing
    /// a retired or replaced handle cannot release a different live export.
    pub fn release(&mut self, chunk: &Arc<PoolChunk>) -> Result<()> {
        let Some(allocation) = self.allocations.get(&chunk.offset) else {
            return Ok(());
        };
        if !Arc::ptr_eq(&allocation.chunk, chunk) {
            return Ok(());
        }
        if allocation.readers.is_some() {
            return Err(Error::State("VMM chunk still awaits reader retirement"));
        }
        self.remove(chunk.offset);
        Ok(())
    }

    /// Revoke a chunk after its consuming calls have resolved. Existing remote
    /// readers may still access it; no new reader may claim it after this call.
    pub fn retire(
        &mut self,
        chunk: &Arc<PoolChunk>,
        consumers: &[usize],
        producer: &Event,
        grant: Option<(Arc<DescriptorGrants>, String)>,
    ) -> Result<()> {
        let mut readers = 0;
        for &slot in consumers {
            if slot >= ACK_SLOTS {
                return Err(Error::Invalid(
                    "VMM acknowledgment slot is out of range".into(),
                ));
            }
            readers |= 1u64 << slot;
        }
        let allocation = self
            .allocations
            .get_mut(&chunk.offset)
            .filter(|allocation| Arc::ptr_eq(&allocation.chunk, chunk))
            .ok_or(Error::State("VMM chunk is no longer allocated"))?;
        if allocation.readers.is_some() {
            return Ok(());
        }

        if readers == 0 && producer.ready().map_err(Error::Cuda)? {
            if let Some((grants, export)) = grant {
                grants.release(&export);
            }
            self.remove(chunk.offset);
            return Ok(());
        }

        // Enqueue this dependency now, before the event pool can reuse the
        // producer event. A later snapshot must not observe the header before
        // its initialization or outlive the producer's final payload write.
        let _device = DeviceGuard::new(self.device).map_err(Error::Cuda)?;
        producer
            .wait_on(self.stream.handle())
            .map_err(Error::Cuda)?;
        allocation.readers = Some(readers);
        allocation.grant = grant;
        Ok(())
    }

    pub fn awaiting_acknowledgment(&self) -> bool {
        self.allocations
            .values()
            .any(|allocation| allocation.readers.is_some())
    }

    /// Consume only completed readback, then submit the next bounded snapshot.
    pub fn reap(&mut self) -> Result<()> {
        self.backing()?;
        if self.failed {
            return Err(Error::Resource("VMM pool device completion is unknown"));
        }
        let result = self.poll();
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    #[expect(
        clippy::expect_used,
        reason = "each snapshot retains its retired ranges and pinned destination"
    )]
    fn poll(&mut self) -> Result<()> {
        let _device = DeviceGuard::new(self.device).map_err(Error::Cuda)?;
        if !self.observed.is_empty() {
            if !self.completion.ready().map_err(Error::Cuda)? {
                return Ok(());
            }
            let mut index = 0;
            for offset in std::mem::take(&mut self.observed) {
                let allocation = &self.allocations[&offset];
                let readers = allocation.readers.expect("snapshot owns retired chunk");
                if readers == 0 {
                    self.remove(offset);
                    continue;
                }
                let address = self.readback[self.readback_slot].address();
                // SAFETY: the completed snapshot contains this chunk's full
                // header, and no DMA writes this pinned allocation until the
                // next snapshot is submitted below.
                let words = unsafe {
                    std::slice::from_raw_parts(
                        (address + index * HEADER_BYTES) as *const u32,
                        ACK_SLOTS,
                    )
                };
                if words
                    .iter()
                    .enumerate()
                    .all(|(slot, word)| readers & (1 << slot) == 0 || *word != CLAIMED)
                {
                    self.remove(offset);
                }
                index += 1;
            }
        }

        let observed: Vec<_> = self
            .allocations
            .iter()
            .filter_map(|(&offset, allocation)| allocation.readers.map(|_| offset))
            .collect();
        if observed.is_empty() {
            return Ok(());
        }
        let watched: Vec<_> = observed
            .iter()
            .copied()
            .filter(|offset| self.allocations[offset].readers != Some(0))
            .collect();
        let bytes = watched.len() * HEADER_BYTES;
        let destination = if bytes == 0 {
            0
        } else {
            // Freeing pinned memory can wait for unrelated device work.
            // Reuse buffers by capacity and release the cache only at close;
            // geometric growth bounds its total size by twice its largest buffer.
            self.readback_slot = match self
                .readback
                .iter()
                .position(|buffer| buffer.size() >= bytes)
            {
                Some(slot) => slot,
                None => {
                    self.readback
                        .push(PinnedBuffer::new(bytes.next_power_of_two()).map_err(Error::Cuda)?);
                    self.readback.len() - 1
                }
            };
            self.readback[self.readback_slot].address()
        };
        let copies = CopyBatch::new(watched.iter().enumerate().map(|(index, offset)| {
            (
                destination + index * HEADER_BYTES,
                self.address + offset,
                HEADER_BYTES,
            )
        }));
        // Retain the observed ranges even if a copy or fence submission fails.
        // Shutdown drains this stream before either mapping can be destroyed.
        self.observed = observed;
        copies.copy_on(self.stream.handle()).map_err(Error::Cuda)?;
        self.completion
            .record(self.stream.handle())
            .map_err(Error::Cuda)
    }

    fn remove(&mut self, offset: usize) {
        self.allocations.remove(&offset);
        if self.allocations.is_empty() {
            self.watermark = 0;
        }
    }

    /// The caller ends all local and remote uses before closing the pool.
    /// Drain owned readback and return numerical backing outside owner locks.
    pub fn close(&mut self) -> Result<Option<B>> {
        if self.backing.is_none() {
            return Ok(None);
        }
        let _device = DeviceGuard::new(self.device).map_err(Error::Cuda)?;
        for &stream in &self.initializers {
            Stream::borrowed(stream).wait().map_err(Error::Cuda)?;
        }
        self.initializers.clear();
        self.stream.wait().map_err(Error::Cuda)?;
        self.observed.clear();
        self.readback.clear();
        self.allocations.clear();
        Ok(self.backing.take())
    }
}

impl<B> Drop for VmmPool<B> {
    #[expect(
        clippy::mem_forget,
        reason = "unknown device completion must retain both DMA mappings"
    )]
    fn drop(&mut self) {
        if self.close().is_err() {
            std::mem::forget(self.backing.take());
            std::mem::forget(std::mem::take(&mut self.readback));
        }
    }
}
