//! Scheduler-owned logical device allocations and worker region descriptors.
//!
//! The manager reserves request slots, paged KV blocks, latent regions, and
//! aligned buffers. Allocation identities remain host-side; workers receive
//! immutable regions derived from them.

use std::collections::{BTreeMap, HashMap, HashSet};

use uniserve_core::{BlockId, HashAlgo};
use uniserve_worker_ipc::{BufferId, RequestKey, WorkerInfo};

use crate::kv::{BlockPool, BlockTable, EncoderCacheManager, KvCacheCoordinator};

/// Opaque identity of one scheduler-owned allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocationId(u64);

/// Logical resource shape requested from the memory manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryLayout {
    /// One stable request-state row.
    RequestSlot,
    /// Paged KV capacity replicated across cache groups.
    Kv {
        /// Token capacity requested for each group.
        tokens: u32,
        /// Number of cache groups requiring block tables.
        groups: u32,
    },
    /// Paged latent trajectory capacity.
    Latent {
        /// Number of logical latent units requested.
        units: u64,
    },
    /// Byte-addressed persistent tensor storage.
    Buffer {
        /// Number of bytes requested.
        bytes: u64,
        /// Required byte alignment of the allocation start.
        alignment: u32,
    },
}

/// Immutable worker-visible region of a logical allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationRegion {
    /// Stable request-state row assigned to the allocation.
    RequestSlot {
        /// Positive row index; zero remains reserved for inactive graph input.
        index: u32,
    },
    /// Physical page tables for each requested KV cache group.
    Kv {
        /// Per-group block identifiers in logical order.
        tables: Vec<Vec<BlockId>>,
    },
    /// Physical pages backing a latent trajectory.
    Latent {
        /// Latent page identifiers in logical order.
        pages: Vec<u32>,
        /// Logical latent units covered by the region.
        units: u64,
    },
    /// Address span in the persistent-buffer arena.
    Buffer {
        /// Byte offset from the start of the arena.
        offset: u64,
        /// Reserved span length in bytes.
        bytes: u64,
    },
}

#[derive(Debug)]
enum AllocationBacking {
    RequestSlot(u32),
    Kv(Vec<BlockTable>),
    Latent {
        pages: Vec<u32>,
        units: u64,
    },
    Buffer {
        offset: u64,
        bytes: u64,
        capacity: u64,
    },
}

/// An owned logical allocation. The handle is intentionally not cloneable;
/// request retirement consumes it through [`Memory::free`].
#[derive(Debug)]
pub struct Allocation {
    id: AllocationId,
    owner: RequestKey,
    backing: AllocationBacking,
}

impl Allocation {
    /// Returns the manager-assigned allocation identity.
    pub const fn id(&self) -> AllocationId {
        self.id
    }

    /// Returns the request lineage that owns this allocation.
    pub const fn owner(&self) -> RequestKey {
        self.owner
    }

    /// Projects the currently authorized physical region from its owned storage.
    pub fn region(&self) -> AllocationRegion {
        match &self.backing {
            AllocationBacking::RequestSlot(index) => {
                AllocationRegion::RequestSlot { index: *index }
            }
            AllocationBacking::Kv(tables) => AllocationRegion::Kv {
                tables: tables.iter().map(BlockTable::page_ids).collect(),
            },
            AllocationBacking::Latent { pages, units } => AllocationRegion::Latent {
                pages: pages.clone(),
                units: *units,
            },
            AllocationBacking::Buffer { offset, bytes, .. } => AllocationRegion::Buffer {
                offset: *offset,
                bytes: *bytes,
            },
        }
    }

    /// Returns the request-slot identifier.
    pub(crate) fn request_slot(&self) -> Option<u32> {
        match self.backing {
            AllocationBacking::RequestSlot(index) => Some(index),
            _ => None,
        }
    }

    /// Returns shared access to the KV tables.
    pub(crate) fn kv_tables(&self) -> Option<&[BlockTable]> {
        match &self.backing {
            AllocationBacking::Kv(tables) => Some(tables),
            _ => None,
        }
    }

    /// Returns mutable access to the KV tables.
    pub(crate) fn kv_tables_mut(&mut self) -> Option<&mut Vec<BlockTable>> {
        match &mut self.backing {
            AllocationBacking::Kv(tables) => Some(tables),
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
    #[error("the requested allocation layout is invalid")]
    /// The requested shape, count, or alignment is invalid.
    InvalidLayout,
}

/// Current allocation counts and available capacity by memory class.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryStats {
    /// Number of allocation handles currently owned by the manager.
    pub live_allocations: usize,
    /// Request-state rows available for admission.
    pub free_request_slots: usize,
    /// Physical KV pages available across all cache groups.
    pub free_kv_pages: usize,
    /// Physical latent pages currently assigned to requests.
    pub used_latent_pages: usize,
    /// Total bytes available in persistent-buffer free spans.
    pub free_buffer_bytes: u64,
}

/// Dense pool of stable request-state row identifiers.
pub(crate) struct RequestSlotPool {
    free: Vec<u32>,
    live: Vec<bool>,
}

impl RequestSlotPool {
    /// Creates an allocator for the supplied capacity.
    fn new(capacity: usize) -> Self {
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

    /// Acquires an available slot.
    pub(crate) fn acquire(&mut self) -> Option<u32> {
        let index = self.free.pop()?;
        self.live[index as usize] = true;
        Some(index)
    }

    /// Releases the supplied allocation.
    pub(crate) fn release(&mut self, index: u32) -> Result<(), &'static str> {
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
pub(crate) struct LatentPagePool {
    page_units: u32,
    capacity: usize,
    free: Vec<u32>,
}

impl LatentPagePool {
    /// Creates an allocator with page zero reserved for inactive input.
    fn new(num_pages: u32, page_units: u32) -> Self {
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
pub(crate) struct KvMemoryState {
    pub(crate) block_pool: BlockPool,
    pub(crate) coordinator: KvCacheCoordinator,
    pub(crate) usable_blocks: usize,
}

/// Builds scheduler KV state when a worker advertises paged cache capacity.
pub(crate) fn worker_kv_state(info: &WorkerInfo) -> Option<KvMemoryState> {
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
        KvMemoryState {
            block_pool,
            coordinator: KvCacheCoordinator::default(),
            usable_blocks,
        }
    })
}

struct BufferPool {
    capacity: u64,
    free: BTreeMap<u64, u64>,
}

impl BufferPool {
    /// Creates an address-space allocator with one initial free extent.
    fn new(capacity: u64) -> Self {
        let free = (capacity > 0)
            .then_some((0, capacity))
            .into_iter()
            .collect();
        Self { capacity, free }
    }

    /// Allocates the first aligned extent large enough for `bytes`.
    fn alloc(&mut self, bytes: u64, alignment: u32) -> Option<u64> {
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
    fn free(&mut self, offset: u64, bytes: u64) {
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

    /// Extends an allocation into its adjacent free extent without relocating it.
    fn grow_in_place(&mut self, offset: u64, current: u64, requested: u64) -> bool {
        if requested <= current {
            return true;
        }
        let Some(start) = offset.checked_add(current) else {
            return false;
        };
        let Some(required) = requested.checked_sub(current) else {
            return false;
        };
        let Some(available) = self.free.get(&start).copied() else {
            return false;
        };
        if available < required {
            return false;
        }
        self.free.remove(&start);
        if available > required {
            self.free.insert(start + required, available - required);
        }
        true
    }

    /// Returns the number of unallocated bytes.
    fn free_bytes(&self) -> u64 {
        self.free.values().copied().sum()
    }
}

/// Scheduler authority for request-scoped logical allocations.
pub struct Memory {
    pub(crate) cache: Option<KvMemoryState>,
    pub(crate) encoder_cache: EncoderCacheManager,
    pub(crate) reserved_encoder_entries: usize,
    pub(crate) request_slots: RequestSlotPool,
    pub(crate) latent_pages: LatentPagePool,
    pub(crate) reserved_blocks: usize,
    buffers: BufferPool,
    encoder_buffers: HashMap<BufferId, Allocation>,
    next_allocation_id: u64,
    live: HashSet<AllocationId>,
}

impl Memory {
    /// Creates scheduler memory state from explicit logical pool capacities.
    pub(crate) fn new(
        cache: Option<KvMemoryState>,
        encoder_cache_budget: usize,
        request_pool_capacity: usize,
        num_latent_pages: u32,
        latent_page_units: u32,
    ) -> Self {
        Self {
            cache,
            encoder_cache: EncoderCacheManager::new(encoder_cache_budget),
            reserved_encoder_entries: 0,
            request_slots: RequestSlotPool::new(request_pool_capacity),
            latent_pages: LatentPagePool::new(num_latent_pages, latent_page_units),
            reserved_blocks: 0,
            buffers: BufferPool::new(0),
            encoder_buffers: HashMap::new(),
            next_allocation_id: 1,
            live: HashSet::new(),
        }
    }

    /// Constructs memory pools from worker-advertised capacities.
    pub fn from_worker_info(info: &WorkerInfo) -> Self {
        Self::with_buffer_capacity(info, info.buffer_pool_bytes, info.request_slots as usize)
    }

    /// Constructs memory pools with explicit persistent-buffer and encoder-cache limits.
    pub fn with_buffer_capacity(
        info: &WorkerInfo,
        buffer_pool_bytes: u64,
        encoder_cache_entries: usize,
    ) -> Self {
        let mut memory = Self::new(
            worker_kv_state(info),
            encoder_cache_entries,
            info.request_slots as usize,
            info.latent_pages,
            info.latent_page_units,
        );
        memory.buffers = BufferPool::new(buffer_pool_bytes);
        memory
    }

    /// Returns the allocation identifier.
    fn allocation_id(&mut self) -> AllocationId {
        let id = AllocationId(self.next_allocation_id);
        self.next_allocation_id = self.next_allocation_id.saturating_add(1);
        let inserted = self.live.insert(id);
        debug_assert!(inserted, "allocation identity reused");
        id
    }

    /// Reserves one request-owned resource and returns its immutable region.
    ///
    /// # Errors
    ///
    /// Returns the exhausted resource class or [`OutOfMemory::InvalidLayout`] for
    /// a shape that does not match the configured worker pools.
    pub fn alloc(
        &mut self,
        owner: RequestKey,
        layout: MemoryLayout,
    ) -> Result<Allocation, OutOfMemory> {
        // Each resource class records enough backing state to release or grow
        // the region without consulting the caller's requested layout.
        let backing = match layout {
            MemoryLayout::RequestSlot => {
                let index = self
                    .request_slots
                    .acquire()
                    .ok_or(OutOfMemory::RequestSlots)?;
                AllocationBacking::RequestSlot(index)
            }
            MemoryLayout::Kv { tokens, groups } => {
                let cache = self.cache.as_ref().ok_or(OutOfMemory::Kv)?;
                if groups as usize != cache.block_pool.num_groups() {
                    return Err(OutOfMemory::InvalidLayout);
                }
                let mut tables = (0..groups as usize)
                    .map(|group| BlockTable::new(group, cache.block_pool.block_size()))
                    .collect::<Vec<_>>();
                cache
                    .coordinator
                    .ensure_capacity(&cache.block_pool, &mut tables, tokens as usize)
                    .ok_or(OutOfMemory::Kv)?;
                AllocationBacking::Kv(tables)
            }
            MemoryLayout::Latent { units } => {
                let mut pages = Vec::new();
                if !self.latent_pages.reserve(&mut pages, units) {
                    return Err(OutOfMemory::Latent);
                }
                AllocationBacking::Latent { pages, units }
            }
            MemoryLayout::Buffer { bytes, alignment } => {
                if bytes == 0 || alignment == 0 || !alignment.is_power_of_two() {
                    return Err(OutOfMemory::InvalidLayout);
                }
                let offset = self
                    .buffers
                    .alloc(bytes, alignment)
                    .ok_or(OutOfMemory::Buffer)?;
                AllocationBacking::Buffer {
                    offset,
                    bytes,
                    capacity: bytes,
                }
            }
        };

        let id = self.allocation_id();
        Ok(Allocation { id, owner, backing })
    }

    /// Expands or narrows a live allocation within its existing resource class.
    ///
    /// # Errors
    ///
    /// Returns [`OutOfMemory::InvalidLayout`] for stale handles or class changes,
    /// and the relevant capacity error when expansion cannot be reserved.
    pub fn grow(
        &mut self,
        allocation: &mut Allocation,
        layout: MemoryLayout,
    ) -> Result<AllocationRegion, OutOfMemory> {
        if !self.live.contains(&allocation.id) {
            return Err(OutOfMemory::InvalidLayout);
        }

        match (&mut allocation.backing, &layout) {
            (AllocationBacking::RequestSlot(_), MemoryLayout::RequestSlot) => {}
            (AllocationBacking::Kv(tables), MemoryLayout::Kv { tokens, groups }) => {
                let cache = self.cache.as_ref().ok_or(OutOfMemory::Kv)?;
                if *groups as usize != tables.len() {
                    return Err(OutOfMemory::InvalidLayout);
                }
                cache
                    .coordinator
                    .ensure_capacity(&cache.block_pool, tables, *tokens as usize)
                    .ok_or(OutOfMemory::Kv)?;
            }
            (
                AllocationBacking::Latent {
                    pages,
                    units: held_units,
                },
                MemoryLayout::Latent { units },
            ) => {
                if !self.latent_pages.reserve(pages, *units) {
                    return Err(OutOfMemory::Latent);
                }
                *held_units = *units;
            }
            (
                AllocationBacking::Buffer {
                    offset,
                    bytes,
                    capacity,
                },
                MemoryLayout::Buffer {
                    bytes: requested,
                    alignment,
                },
            ) => {
                if *requested == 0 || *alignment == 0 || !alignment.is_power_of_two() {
                    return Err(OutOfMemory::InvalidLayout);
                }
                let aligned = offset.is_multiple_of(u64::from(*alignment));
                if *requested > *capacity || !aligned {
                    // Keep the current reservation until a suitably aligned
                    // extension or replacement has succeeded.
                    let new_offset =
                        if aligned && self.buffers.grow_in_place(*offset, *capacity, *requested) {
                            *offset
                        } else {
                            let new_offset = self
                                .buffers
                                .alloc(*requested, *alignment)
                                .ok_or(OutOfMemory::Buffer)?;
                            self.buffers.free(*offset, *capacity);
                            new_offset
                        };
                    *offset = new_offset;
                    *capacity = *requested;
                }
                *bytes = *requested;
            }
            _ => return Err(OutOfMemory::InvalidLayout),
        }
        Ok(allocation.region())
    }

    /// Releases an allocation and returns its backing capacity to the owning pool.
    pub fn free(&mut self, allocation: Allocation) {
        let live = self.live.remove(&allocation.id);
        debug_assert!(live, "stale or duplicate allocation free");
        if !live {
            return;
        }

        match allocation.backing {
            AllocationBacking::RequestSlot(index) => {
                let result = self.request_slots.release(index);
                debug_assert!(result.is_ok(), "request slot allocation was not live");
            }
            AllocationBacking::Kv(tables) => drop(tables),
            AllocationBacking::Latent { pages, .. } => self.latent_pages.release(pages),
            AllocationBacking::Buffer {
                offset, capacity, ..
            } => self.buffers.free(offset, capacity),
        }
    }

    /// Returns an instantaneous capacity snapshot.
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            live_allocations: self.live.len(),
            free_request_slots: self.request_slots.free.len(),
            free_kv_pages: self.free_blocks(),
            used_latent_pages: self.latent_pages.used_pages(),
            free_buffer_bytes: self.buffers.free_bytes(),
        }
    }

    /// Retains the encoder buffer.
    pub(crate) fn retain_encoder_buffer(
        &mut self,
        buffer: BufferId,
        allocation: Allocation,
    ) -> Result<(), Allocation> {
        if allocation.owner != buffer.owner
            || !matches!(allocation.backing, AllocationBacking::Buffer { .. })
            || self.encoder_buffers.contains_key(&buffer)
        {
            return Err(allocation);
        }
        self.encoder_buffers.insert(buffer, allocation);
        Ok(())
    }

    /// Takes the encoder buffer.
    pub(crate) fn take_encoder_buffer(&mut self, buffer: BufferId) -> Option<Allocation> {
        self.encoder_buffers.remove(&buffer)
    }

    /// Returns allocations transferred from this lineage to the shared encoder cache.
    pub(crate) fn retained_buffers(&self, request: RequestKey) -> Vec<BufferId> {
        let mut buffers: Vec<_> = self
            .encoder_buffers
            .keys()
            .copied()
            .filter(|buffer| buffer.owner == request)
            .collect();
        buffers.sort_unstable_by_key(|buffer| {
            (
                buffer.producer_op_id.0,
                buffer.output_index,
                buffer.generation,
            )
        });
        buffers
    }

    /// Returns shared access to the KV cache.
    pub(crate) fn cache(&self) -> &KvMemoryState {
        self.cache
            .as_ref()
            .expect("generation scheduling requires worker KV resources")
    }

    /// Returns the number of free KV blocks.
    pub(crate) fn free_blocks(&self) -> usize {
        self.cache
            .as_ref()
            .map_or(0, |state| state.block_pool.free_request_pages())
    }

    /// Returns the number of KV blocks available for allocation.
    pub(crate) fn usable_blocks(&self) -> usize {
        self.cache.as_ref().map_or(0, |state| state.usable_blocks)
    }

    /// Enables or disables prefix caching.
    pub(crate) fn set_prefix_cache(&mut self, enabled: bool) {
        if let Some(cache) = self.cache.as_mut() {
            cache.coordinator.set_prefix_enabled(enabled);
        }
    }

    /// Sets the prefix-cache hashing algorithm.
    pub(crate) fn set_hash_algo(&mut self, algo: HashAlgo) {
        if let Some(cache) = self.cache.as_mut() {
            cache.coordinator.set_hash_algo(algo);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::RequestId;

    fn owner(request_id: u64) -> RequestKey {
        RequestKey::new(1, RequestId(request_id), 1)
    }

    #[test]
    fn buffer_allocations_are_non_overlapping_and_free_exactly() {
        let info = WorkerInfo {
            buffer_pool_bytes: 1_024,
            ..WorkerInfo::default()
        };
        let mut memory = Memory::from_worker_info(&info);
        let first = memory
            .alloc(
                owner(1),
                MemoryLayout::Buffer {
                    bytes: 200,
                    alignment: 256,
                },
            )
            .unwrap();
        let second = memory
            .alloc(
                owner(2),
                MemoryLayout::Buffer {
                    bytes: 200,
                    alignment: 256,
                },
            )
            .unwrap();
        assert_eq!(
            first.region(),
            AllocationRegion::Buffer {
                offset: 0,
                bytes: 200
            }
        );
        assert_eq!(
            second.region(),
            AllocationRegion::Buffer {
                offset: 256,
                bytes: 200,
            }
        );
        assert_eq!(memory.stats().free_buffer_bytes, 624);

        memory.free(first);
        memory.free(second);
        assert_eq!(memory.stats().free_buffer_bytes, 1_024);
        assert_eq!(memory.stats().live_allocations, 0);
    }

    #[test]
    fn failed_buffer_growth_leaves_allocation_and_pool_unchanged() {
        let info = WorkerInfo {
            buffer_pool_bytes: 1_024,
            ..WorkerInfo::default()
        };
        let mut memory = Memory::from_worker_info(&info);
        let mut first = memory
            .alloc(
                owner(1),
                MemoryLayout::Buffer {
                    bytes: 512,
                    alignment: 256,
                },
            )
            .unwrap();
        let second = memory
            .alloc(
                owner(2),
                MemoryLayout::Buffer {
                    bytes: 512,
                    alignment: 256,
                },
            )
            .unwrap();
        let region = first.region();
        let stats = memory.stats();

        assert_eq!(
            memory.grow(
                &mut first,
                MemoryLayout::Buffer {
                    bytes: 768,
                    alignment: 256,
                },
            ),
            Err(OutOfMemory::Buffer)
        );
        assert_eq!(first.region(), region);
        assert_eq!(memory.stats(), stats);

        memory.free(second);
        assert_eq!(
            memory
                .grow(
                    &mut first,
                    MemoryLayout::Buffer {
                        bytes: 768,
                        alignment: 256,
                    },
                )
                .unwrap(),
            AllocationRegion::Buffer {
                offset: 0,
                bytes: 768,
            }
        );
        memory.free(first);
        assert_eq!(memory.stats().free_buffer_bytes, 1_024);
    }

    #[test]
    fn latent_allocations_keep_independent_pages_for_one_request() {
        let info = WorkerInfo {
            latent_pages: 4,
            latent_page_units: 4,
            ..WorkerInfo::default()
        };
        let mut memory = Memory::from_worker_info(&info);
        let mut first = memory
            .alloc(owner(1), MemoryLayout::Latent { units: 4 })
            .unwrap();
        let second = memory
            .alloc(owner(1), MemoryLayout::Latent { units: 4 })
            .unwrap();
        let AllocationRegion::Latent {
            pages: first_pages, ..
        } = first.region()
        else {
            panic!("latent allocation returned another resource class");
        };
        let AllocationRegion::Latent {
            pages: second_pages,
            ..
        } = second.region()
        else {
            panic!("latent allocation returned another resource class");
        };
        assert!(first_pages.iter().all(|page| !second_pages.contains(page)));
        let region = first.region();
        let stats = memory.stats();
        assert_eq!(
            memory.grow(&mut first, MemoryLayout::Latent { units: 12 }),
            Err(OutOfMemory::Latent),
        );
        assert_eq!(first.region(), region);
        assert_eq!(memory.stats(), stats);
        memory.free(second);
        let grown = memory
            .grow(&mut first, MemoryLayout::Latent { units: 12 })
            .unwrap();
        let AllocationRegion::Latent { pages, units } = grown else {
            panic!("latent growth returned another resource class");
        };
        assert_eq!(units, 12);
        assert_eq!(pages.len(), 3);
        assert!(pages.starts_with(&first_pages));
        memory.free(first);
        assert_eq!(memory.stats().used_latent_pages, 0);
        assert!(
            memory
                .alloc(owner(2), MemoryLayout::Latent { units: 12 })
                .is_ok()
        );
    }

    #[test]
    fn buffer_narrowing_retains_capacity_until_release() {
        let info = WorkerInfo {
            buffer_pool_bytes: 1_024,
            ..WorkerInfo::default()
        };
        let mut memory = Memory::from_worker_info(&info);
        let mut allocation = memory
            .alloc(
                owner(1),
                MemoryLayout::Buffer {
                    bytes: 512,
                    alignment: 256,
                },
            )
            .unwrap();
        let region = memory
            .grow(
                &mut allocation,
                MemoryLayout::Buffer {
                    bytes: 128,
                    alignment: 256,
                },
            )
            .unwrap();
        assert_eq!(
            region,
            AllocationRegion::Buffer {
                offset: 0,
                bytes: 128
            }
        );
        assert_eq!(memory.stats().free_buffer_bytes, 512);
        assert!(
            memory
                .grow(
                    &mut allocation,
                    MemoryLayout::Buffer {
                        bytes: 512,
                        alignment: 256
                    },
                )
                .is_ok()
        );
        memory.free(allocation);
        assert_eq!(memory.stats().free_buffer_bytes, 1_024);
    }

    #[test]
    fn buffer_resize_preserves_alignment_and_failed_reservations() {
        let info = WorkerInfo {
            buffer_pool_bytes: 1_024,
            ..WorkerInfo::default()
        };
        let mut memory = Memory::from_worker_info(&info);
        let prefix = memory
            .alloc(
                owner(1),
                MemoryLayout::Buffer {
                    bytes: 256,
                    alignment: 256,
                },
            )
            .unwrap();
        let mut allocation = memory
            .alloc(
                owner(2),
                MemoryLayout::Buffer {
                    bytes: 256,
                    alignment: 256,
                },
            )
            .unwrap();
        let suffix = memory
            .alloc(
                owner(3),
                MemoryLayout::Buffer {
                    bytes: 512,
                    alignment: 256,
                },
            )
            .unwrap();
        let region = allocation.region();
        let stats = memory.stats();
        let layout = MemoryLayout::Buffer {
            bytes: 128,
            alignment: 512,
        };
        assert_eq!(
            memory.grow(&mut allocation, layout.clone()),
            Err(OutOfMemory::Buffer)
        );
        assert_eq!(allocation.region(), region);
        assert_eq!(memory.stats(), stats);
        memory.free(suffix);
        let AllocationRegion::Buffer { offset, bytes } =
            memory.grow(&mut allocation, layout).unwrap()
        else {
            panic!("buffer resize returned another resource class");
        };
        assert_eq!(offset % 512, 0);
        assert_eq!(bytes, 128);
        memory.free(allocation);
        memory.free(prefix);
        assert_eq!(memory.stats().free_buffer_bytes, 1_024);
    }
}
