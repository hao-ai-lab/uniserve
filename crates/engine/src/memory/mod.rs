//! Scheduler-owned request slots, KV references, latent pages, and Tensor spans.
//!
//! Each pool allocates its actual backing. Worker addresses are assembled from
//! these owned allocations when constructing a batch; no region copies persist.

use std::collections::BTreeMap;

use uniserve_core::HashAlgo;
use uniserve_worker_ipc::{RequestKey, WorkerInfo};

use crate::kv::{BlockPool, BlockTable, KvCacheCoordinator};

/// Owned backing for one resource, consumed when its final reader retires.
#[derive(Debug)]
pub(crate) enum Allocation {
    RequestSlot {
        owner: RequestKey,
        index: u32,
    },
    Kv {
        owner: RequestKey,
        tables: Vec<BlockTable>,
    },
    Latent {
        owner: RequestKey,
        pages: Vec<u32>,
        units: u64,
    },
    Buffer {
        owner: RequestKey,
        offset: u64,
        bytes: u64,
    },
}

impl Allocation {
    /// Request epoch that originally acquired the backing.
    pub(crate) const fn owner(&self) -> RequestKey {
        match self {
            Self::RequestSlot { owner, .. }
            | Self::Kv { owner, .. }
            | Self::Latent { owner, .. }
            | Self::Buffer { owner, .. } => *owner,
        }
    }

    pub(crate) fn request_slot(&self) -> Option<u32> {
        match self {
            Self::RequestSlot { index, .. } => Some(*index),
            _ => None,
        }
    }

    pub(crate) fn kv_tables(&self) -> Option<&[BlockTable]> {
        match self {
            Self::Kv { tables, .. } => Some(tables),
            _ => None,
        }
    }

    pub(crate) fn kv_tables_mut(&mut self) -> Option<&mut Vec<BlockTable>> {
        match self {
            Self::Kv { tables, .. } => Some(tables),
            _ => None,
        }
    }
}

/// Allocation failure identifying the exhausted resource class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OutOfMemory {
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
pub(crate) struct RequestPool {
    free: Vec<u32>,
    live: Vec<bool>,
}

impl RequestPool {
    /// Creates an allocator for the supplied capacity.
    pub(crate) fn new(capacity: usize) -> Self {
        let capacity = capacity.clamp(1, u32::MAX as usize);
        Self {
            free: (1..=capacity as u32).rev().collect(),
            live: vec![false; capacity + 1],
        }
    }

    /// Returns the allocator capacity.
    pub(crate) fn capacity(&self) -> usize {
        self.live.len().saturating_sub(1)
    }

    /// Returns whether the collection contains no entries.
    pub(crate) fn is_empty(&self) -> bool {
        self.free.is_empty()
    }

    /// Returns the number of request rows that can still be assigned.
    pub(crate) fn available(&self) -> usize {
        self.free.len()
    }

    /// Acquires an available slot.
    fn acquire(&mut self) -> Option<u32> {
        let index = self.free.pop()?;
        self.live[index as usize] = true;
        Some(index)
    }

    /// Releases the supplied allocation.
    fn release(&mut self, index: u32) -> Result<(), &'static str> {
        let Some(live) = self.live.get_mut(index as usize) else {
            return Err("request-pool index is outside scheduler capacity");
        };
        if index == 0 || !*live {
            return Err("request-pool index is not live");
        }
        *live = false;
        self.free.push(index);
        Ok(())
    }
}

/// Fixed-size page allocator for request-owned latent storage.
pub(crate) struct LatentPool {
    page_units: u32,
    capacity: usize,
    free: Vec<u32>,
}

impl LatentPool {
    /// Creates an allocator with page zero reserved for inactive input.
    pub(crate) fn new(num_pages: u32, page_units: u32) -> Self {
        Self {
            page_units,
            capacity: num_pages.saturating_sub(1) as usize,
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
        if additional > self.free.len() {
            return false;
        }
        pages.reserve(additional);
        for _ in 0..additional {
            pages.push(self.free.pop().expect("latent free-page invariant"));
        }
        true
    }

    /// Returns exclusively owned pages after the allocation retires.
    fn release(&mut self, pages: Vec<u32>) {
        self.free.extend(pages.into_iter().rev());
    }

    /// Returns the number of allocated pages.
    pub(crate) fn used_pages(&self) -> usize {
        self.capacity - self.free.len()
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
                BlockPool::new(kv_cache.num_blocks as usize, kv_cache.block_size as usize)
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
    pub(crate) fn allocate(&mut self, owner: RequestKey) -> Result<Allocation, OutOfMemory> {
        let index = self.acquire().ok_or(OutOfMemory::RequestSlots)?;
        Ok(Allocation::RequestSlot { owner, index })
    }

    /// Return a row after all participating workers acknowledge its release.
    pub(crate) fn free(&mut self, allocation: Allocation) {
        let Allocation::RequestSlot { index, .. } = allocation else {
            panic!("request pool received another resource class");
        };
        self.release(index)
            .expect("request row remains allocated until release");
    }
}

impl LatentPool {
    /// Reserve the physical pages of one latent trajectory atomically.
    pub(crate) fn allocate(
        &mut self,
        owner: RequestKey,
        units: u64,
    ) -> Result<Allocation, OutOfMemory> {
        let mut pages = Vec::new();
        if !self.reserve(&mut pages, units) {
            return Err(OutOfMemory::Latent);
        }
        Ok(Allocation::Latent {
            owner,
            pages,
            units,
        })
    }

    /// Grow without moving existing pages; failure leaves the allocation intact.
    pub(crate) fn grow(
        &mut self,
        allocation: &mut Allocation,
        requested: u64,
    ) -> Result<(), OutOfMemory> {
        let Allocation::Latent { pages, units, .. } = allocation else {
            return Err(OutOfMemory::InvalidAllocation);
        };
        if !self.reserve(pages, requested) {
            return Err(OutOfMemory::Latent);
        }
        *units = requested;
        Ok(())
    }

    /// Return trajectory pages after their physical consumers have finished.
    pub(crate) fn free(&mut self, allocation: Allocation) {
        let Allocation::Latent { pages, .. } = allocation else {
            panic!("latent pool received another resource class");
        };
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
    ) -> Result<Allocation, OutOfMemory> {
        if bytes == 0 || alignment == 0 || !alignment.is_power_of_two() {
            return Err(OutOfMemory::InvalidAllocation);
        }
        let offset = self.reserve(bytes, alignment).ok_or(OutOfMemory::Buffer)?;
        Ok(Allocation::Buffer {
            owner,
            offset,
            bytes,
        })
    }

    /// Return the complete span after its publication and transport readers retire.
    pub(crate) fn free(&mut self, allocation: Allocation) {
        let Allocation::Buffer { offset, bytes, .. } = allocation else {
            panic!("buffer pool received another resource class");
        };
        self.release(offset, bytes);
    }
}

impl KVCacheManager {
    /// Acquire one page table per loaded KV group with atomic initial capacity.
    pub(crate) fn allocate(
        &self,
        owner: RequestKey,
        tokens: u32,
    ) -> Result<Allocation, OutOfMemory> {
        let mut tables = (0..self.block_pool.num_groups())
            .map(|group| BlockTable::new(group, self.block_pool.block_size()))
            .collect::<Vec<_>>();
        self.coordinator
            .ensure_capacity(&self.block_pool, &mut tables, tokens as usize)
            .ok_or(OutOfMemory::Kv)?;
        Ok(Allocation::Kv { owner, tables })
    }

    /// Grow all cache groups atomically while retaining shared prefix references.
    pub(crate) fn grow(&self, allocation: &mut Allocation, tokens: u32) -> Result<(), OutOfMemory> {
        let Allocation::Kv { tables, .. } = allocation else {
            return Err(OutOfMemory::InvalidAllocation);
        };
        self.coordinator
            .ensure_capacity(&self.block_pool, tables, tokens as usize)
            .ok_or(OutOfMemory::Kv)?;
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
        let first = pool.allocate(owner(1)).unwrap();
        let second = pool.allocate(owner(2)).unwrap();
        assert_ne!(first.request_slot(), second.request_slot());
        assert!(first.request_slot().unwrap() > 0);
        assert!(matches!(
            pool.allocate(owner(3)),
            Err(OutOfMemory::RequestSlots)
        ));
        pool.free(first);
        let replacement = pool.allocate(owner(3)).unwrap();
        assert_ne!(replacement.request_slot(), second.request_slot());
        assert_eq!(replacement.owner(), owner(3));
        pool.free(second);
        pool.free(replacement);
    }

    #[test]
    fn tensor_spans_are_aligned_exclusive_and_reclaimable() {
        let mut pool = BufferPool::new(1024);
        let first = pool.allocate(owner(1), 200, 256).unwrap();
        let second = pool.allocate(owner(2), 200, 256).unwrap();
        let Allocation::Buffer {
            offset: start,
            bytes,
            ..
        } = &first
        else {
            unreachable!()
        };
        let Allocation::Buffer {
            offset: other,
            bytes: other_bytes,
            ..
        } = &second
        else {
            unreachable!()
        };
        assert_eq!(start % 256, 0);
        assert_eq!(other % 256, 0);
        assert!(start + bytes <= *other || other + other_bytes <= *start);
        assert!(matches!(
            pool.allocate(owner(3), 1024, 256),
            Err(OutOfMemory::Buffer)
        ));
        pool.free(first);
        pool.free(second);
        assert!(pool.allocate(owner(3), 1024, 256).is_ok());
    }

    #[test]
    fn latent_growth_preserves_pages_and_failed_reservations() {
        let mut pool = LatentPool::new(4, 4);
        let mut first = pool.allocate(owner(1), 4).unwrap();
        let second = pool.allocate(owner(1), 4).unwrap();
        let Allocation::Latent { pages, .. } = &first else {
            unreachable!()
        };
        let held_pages = pages.clone();
        let Allocation::Latent {
            pages: other_pages, ..
        } = &second
        else {
            unreachable!()
        };
        assert!(pages.iter().all(|page| !other_pages.contains(page)));
        assert_eq!(pool.grow(&mut first, 12), Err(OutOfMemory::Latent));
        let Allocation::Latent { pages, units, .. } = &first else {
            unreachable!()
        };
        assert_eq!(pages, &held_pages);
        assert_eq!(*units, 4);
        assert_eq!(pool.used_pages(), 2);
        pool.free(second);
        pool.grow(&mut first, 12).unwrap();
        let Allocation::Latent { pages, units, .. } = &first else {
            unreachable!()
        };
        assert!(pages.starts_with(&held_pages));
        assert_eq!(*units, 12);
        assert_eq!(pages.len(), 3);
        pool.free(first);
        assert_eq!(pool.used_pages(), 0);
        assert!(pool.allocate(owner(2), 12).is_ok());
    }
}
