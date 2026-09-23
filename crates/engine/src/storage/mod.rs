//! Scheduler-owned request slots, KV references, latent pages, and Tensor spans.
//!
//! Each pool allocates its actual backing. Worker addresses are assembled from
//! these owned allocations when constructing a batch; no region copies persist.

use std::collections::BTreeMap;

use uniserve_core::HashAlgo;
use uniserve_worker_ipc::{RequestKey, WorkerInfo};

use crate::kv::{BlockPool, BlockTable, KvCacheCoordinator};

/// Owned request row, returned only after its request epoch retires.
///
/// The row index is private so that only [`RequestPool::allocate`] can
/// create a slot.
#[derive(Debug)]
pub(crate) struct RequestSlot {
    index: u32,
}

impl RequestSlot {
    /// Returns the request-state row this slot owns.
    pub(crate) fn index(&self) -> u32 {
        self.index
    }
}

/// KV page references retain shared prefixes through ordinary ownership.
#[derive(Debug)]
pub(crate) struct KvAllocation {
    pub(crate) tables: Vec<BlockTable>,
}

/// Pages and capacity of a latent trajectory.
#[derive(Debug)]
pub(crate) struct LatentPages {
    pub(crate) pages: Vec<u32>,
    pub(crate) units: u64,
}

/// A byte span whose publication may outlive its producing request.
#[derive(Debug)]
pub(crate) struct BufferSpan {
    pub(crate) owner: RequestKey,
    pub(crate) offset: u64,
    pub(crate) bytes: u64,
}

/// Allocation failure identifying the exhausted resource class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OutOfStorage {
    #[error("request slot capacity is exhausted")]
    /// No request-state row remains available.
    RequestSlots,
    #[error("KV page capacity is exhausted")]
    /// The paged KV pool cannot satisfy the requested token capacity.
    Kv,
    #[error("latent page capacity is exhausted")]
    /// The latent page pool cannot satisfy the requested trajectory capacity.
    Latent,
    #[error("persistent buffer capacity is exhausted")]
    /// The persistent-buffer arena has no suitable free span.
    Buffer,
    #[error("the requested allocation is invalid")]
    /// The requested shape, count, or alignment is invalid.
    InvalidAllocation,
}

/// Dense pool of stable request-state row identifiers.
///
/// Only [`RequestPool::allocate`] creates a [`RequestSlot`] and only
/// [`RequestPool::free`] consumes one, so a slot in hand names a live row of
/// the pool that issued it and cannot be released twice.
pub(crate) struct RequestPool {
    free: Vec<u32>,
    capacity: usize,
}

impl RequestPool {
    /// Creates an allocator for the supplied capacity.
    pub(crate) fn new(capacity: usize) -> Self {
        let capacity = capacity.clamp(1, u32::MAX as usize);
        Self {
            free: (1..=capacity as u32).rev().collect(),
            capacity,
        }
    }

    /// Returns the allocator capacity.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns whether the collection contains no entries.
    pub(crate) fn is_empty(&self) -> bool {
        self.free.is_empty()
    }

    /// Returns the number of request rows that can still be assigned.
    pub(crate) fn available(&self) -> usize {
        self.free.len()
    }
}

/// Fixed-size page allocator for request-owned latent storage.
pub(crate) struct LatentPool {
    page_units: u32,
    free: Vec<u32>,
}

impl LatentPool {
    /// Creates an allocator with page zero reserved for inactive input.
    pub(crate) fn new(num_pages: u32, page_units: u32) -> Self {
        Self {
            page_units,
            free: (1..num_pages).rev().collect(),
        }
    }

    /// Computes the number of physical pages needed for logical latent units.
    fn pages_needed(&self, units: u64) -> Option<usize> {
        if units == 0 {
            return Some(0);
        }
        if self.page_units == 0 {
            return None;
        }
        usize::try_from(units.div_ceil(u64::from(self.page_units))).ok()
    }

    /// Extends an allocation atomically; narrowing retains its backing pages.
    fn reserve(&mut self, pages: &mut Vec<u32>, units: u64) -> bool {
        let Some(needed) = self.pages_needed(units) else {
            return false;
        };
        let additional = needed.saturating_sub(pages.len());
        // Pages come from the tail of the free stack, most recently released first.
        let Some(retained) = self.free.len().checked_sub(additional) else {
            return false;
        };
        pages.extend(self.free.drain(retained..).rev());
        true
    }

    /// Returns exclusively owned pages after the allocation retires.
    fn release(&mut self, pages: Vec<u32>) {
        self.free.extend(pages.into_iter().rev());
    }
}

/// KV-cache geometry and coordination derived from worker capabilities.
pub(crate) struct KVCacheManager {
    pub(crate) block_pool: BlockPool,
    pub(crate) coordinator: KvCacheCoordinator,
    pub(crate) usable_blocks: usize,
}

/// Builds scheduler KV state when a worker advertises paged cache capacity.
impl KVCacheManager {
    pub(crate) fn from_worker_info(info: &WorkerInfo) -> Option<Self> {
        info.kv_cache.as_ref().map(|kv_cache| {
            let block_pool = if kv_cache.groups.is_empty() {
                BlockPool::new(kv_cache.num_blocks, kv_cache.block_size as usize)
            } else {
                let mut offset = 0_u32;
                let group_shapes = kv_cache
                    .groups
                    .iter()
                    .map(|group| {
                        let shape = (offset, group.num_blocks);
                        offset = offset.saturating_add(group.num_blocks);
                        shape
                    })
                    .collect::<Vec<_>>();
                BlockPool::with_groups(
                    kv_cache.num_blocks as usize,
                    kv_cache.block_size as usize,
                    &group_shapes,
                )
            };
            let usable_blocks = block_pool.request_page_capacity();
            KVCacheManager {
                block_pool,
                coordinator: KvCacheCoordinator::default(),
                usable_blocks,
            }
        })
    }
}
/// Aligned, byte-addressed persistent Tensor storage.
pub(crate) struct BufferPool {
    capacity: u64,
    free: BTreeMap<u64, u64>,
}

impl BufferPool {
    /// Creates an address-space allocator with one initial free extent.
    pub(crate) fn new(capacity: u64) -> Self {
        let free = (capacity > 0)
            .then_some((0, capacity))
            .into_iter()
            .collect();
        Self { capacity, free }
    }

    /// Allocates the first aligned extent large enough for `bytes`.
    fn reserve(&mut self, bytes: u64, alignment: u32) -> Option<u64> {
        if bytes == 0 || alignment == 0 || !alignment.is_power_of_two() {
            return None;
        }
        let alignment = u64::from(alignment);
        let candidate = self.free.iter().find_map(|(&offset, &extent)| {
            let aligned = offset.checked_add(alignment - 1)? & !(alignment - 1);
            let end = aligned.checked_add(bytes)?;
            (end <= offset.checked_add(extent)?).then_some((offset, extent, aligned, end))
        })?;
        let (offset, extent, aligned, end) = candidate;
        self.free.remove(&offset);
        if aligned > offset {
            self.free.insert(offset, aligned - offset);
        }
        let range_end = offset + extent;
        if end < range_end {
            self.free.insert(end, range_end - end);
        }
        Some(aligned)
    }

    /// Returns an extent and coalesces it with immediately adjacent free ranges.
    fn release(&mut self, offset: u64, bytes: u64) {
        let mut start = offset;
        let mut end = offset.saturating_add(bytes).min(self.capacity);
        if let Some((&left, &extent)) = self.free.range(..=start).next_back()
            && left.saturating_add(extent) == start
        {
            start = left;
            self.free.remove(&left);
        }
        if let Some((&right, &extent)) = self.free.range(start..).next()
            && end == right
        {
            end = right.saturating_add(extent);
            self.free.remove(&right);
        }
        self.free.insert(start, end.saturating_sub(start));
    }
}

impl RequestPool {
    /// Reserve one stable positive row for an admitted request epoch.
    pub(crate) fn allocate(&mut self) -> Result<RequestSlot, OutOfStorage> {
        let index = self.free.pop().ok_or(OutOfStorage::RequestSlots)?;
        Ok(RequestSlot { index })
    }

    /// Return a row after all participating workers acknowledge its release.
    ///
    /// The caller passes the slot this pool issued; consuming it is what
    /// retires the row, so no liveness check remains to fail.
    pub(crate) fn free(&mut self, allocation: RequestSlot) {
        let RequestSlot { index } = allocation;
        self.free.push(index);
    }
}

impl LatentPool {
    /// Reserve the physical pages of one latent trajectory atomically.
    pub(crate) fn allocate(&mut self, units: u64) -> Result<LatentPages, OutOfStorage> {
        let mut pages = Vec::new();
        if !self.reserve(&mut pages, units) {
            return Err(OutOfStorage::Latent);
        }
        Ok(LatentPages { pages, units })
    }

    /// Grow without moving existing pages; failure leaves the allocation intact.
    pub(crate) fn grow(
        &mut self,
        allocation: &mut LatentPages,
        requested: u64,
    ) -> Result<(), OutOfStorage> {
        let LatentPages { pages, units, .. } = allocation;
        if !self.reserve(pages, requested) {
            return Err(OutOfStorage::Latent);
        }
        *units = requested;
        Ok(())
    }

    /// Return trajectory pages after their physical consumers have finished.
    pub(crate) fn free(&mut self, allocation: LatentPages) {
        let LatentPages { pages, .. } = allocation;
        self.release(pages);
    }
}

impl BufferPool {
    /// Reserve an aligned nonempty span; failure does not consume capacity.
    pub(crate) fn allocate(
        &mut self,
        owner: RequestKey,
        bytes: u64,
        alignment: u32,
    ) -> Result<BufferSpan, OutOfStorage> {
        if bytes == 0 || alignment == 0 || !alignment.is_power_of_two() {
            return Err(OutOfStorage::InvalidAllocation);
        }
        let offset = self.reserve(bytes, alignment).ok_or(OutOfStorage::Buffer)?;
        Ok(BufferSpan {
            owner,
            offset,
            bytes,
        })
    }

    /// Return the complete span after its publication and transport readers retire.
    pub(crate) fn free(&mut self, allocation: BufferSpan) {
        let BufferSpan { offset, bytes, .. } = allocation;
        self.release(offset, bytes);
    }
}

impl KVCacheManager {
    /// Acquire one page table per loaded KV group with atomic initial capacity.
    pub(crate) fn allocate(&self, tokens: u32) -> Result<KvAllocation, OutOfStorage> {
        let mut tables = (0..self.block_pool.num_groups())
            .map(|group| BlockTable::new(group, self.block_pool.block_size()))
            .collect::<Vec<_>>();
        self.coordinator
            .ensure_capacity(&self.block_pool, &mut tables, tokens as usize)
            .ok_or(OutOfStorage::Kv)?;
        Ok(KvAllocation { tables })
    }

    /// Grow all cache groups atomically while retaining shared prefix references.
    pub(crate) fn grow(
        &self,
        allocation: &mut KvAllocation,
        tokens: u32,
    ) -> Result<(), OutOfStorage> {
        let KvAllocation { tables, .. } = allocation;
        self.coordinator
            .ensure_capacity(&self.block_pool, tables, tokens as usize)
            .ok_or(OutOfStorage::Kv)?;
        Ok(())
    }

    pub(crate) fn set_prefix_cache(&mut self, enabled: bool) {
        self.coordinator.set_prefix_enabled(enabled);
    }

    pub(crate) fn set_hash_algo(&mut self, algo: HashAlgo) {
        self.coordinator.set_hash_algo(algo);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::RequestId;

    fn owner(id: u64) -> RequestKey {
        RequestKey::new(1, RequestId(id), 1)
    }

    #[test]
    fn request_rows_remain_exclusive_until_release() {
        let mut pool = RequestPool::new(2);
        let first = pool.allocate().unwrap();
        let second = pool.allocate().unwrap();
        assert_ne!(first.index, second.index);
        assert!(first.index > 0);
        assert!(matches!(pool.allocate(), Err(OutOfStorage::RequestSlots)));
        pool.free(first);
        let replacement = pool.allocate().unwrap();
        assert_ne!(replacement.index, second.index);
        pool.free(second);
        pool.free(replacement);
    }

    #[test]
    fn tensor_spans_are_aligned_exclusive_and_reclaimable() {
        let mut pool = BufferPool::new(1024);
        let first = pool.allocate(owner(1), 200, 256).unwrap();
        let second = pool.allocate(owner(2), 200, 256).unwrap();
        let BufferSpan {
            offset: start,
            bytes,
            ..
        } = &first;
        let BufferSpan {
            offset: other,
            bytes: other_bytes,
            ..
        } = &second;
        assert_eq!(start % 256, 0);
        assert_eq!(other % 256, 0);
        assert!(start + bytes <= *other || other + other_bytes <= *start);
        assert!(matches!(
            pool.allocate(owner(3), 1024, 256),
            Err(OutOfStorage::Buffer)
        ));
        pool.free(first);
        pool.free(second);
        assert!(pool.allocate(owner(3), 1024, 256).is_ok());
    }

    #[test]
    fn latent_growth_preserves_pages_and_failed_reservations() {
        let mut pool = LatentPool::new(4, 4);
        let mut first = pool.allocate(4).unwrap();
        let second = pool.allocate(4).unwrap();
        let LatentPages { pages, .. } = &first;
        let held_pages = pages.clone();
        let LatentPages {
            pages: other_pages, ..
        } = &second;
        assert!(pages.iter().all(|page| !other_pages.contains(page)));
        assert_eq!(pool.grow(&mut first, 12), Err(OutOfStorage::Latent));
        let LatentPages { pages, units, .. } = &first;
        assert_eq!(pages, &held_pages);
        assert_eq!(*units, 4);
        pool.free(second);
        pool.grow(&mut first, 12).unwrap();
        let LatentPages { pages, units, .. } = &first;
        assert!(pages.starts_with(&held_pages));
        assert_eq!(*units, 12);
        assert_eq!(pages.len(), 3);
        pool.free(first);
        assert!(pool.allocate(12).is_ok());
    }
}
