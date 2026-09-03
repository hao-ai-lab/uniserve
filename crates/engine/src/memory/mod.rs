//! Scheduler-side authority for logical GPU allocations and immutable worker placements.

use std::collections::{BTreeMap, HashMap, HashSet};

use uniserve_core::{BlockId, HashAlgo, RequestId};
use uniserve_worker_ipc::{BufferId, RequestKey, WorkerInfo};

use crate::kv::{BlockPool, BlockTable, EncoderCacheManager, KvCacheCoordinator};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocationId(u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryLayout {
    RequestSlot,
    Kv { tokens: u32, groups: u32 },
    Latent { units: u64 },
    Buffer { bytes: u64, alignment: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    RequestSlot { index: u32 },
    Kv { tables: Vec<Vec<BlockId>> },
    Latent { pages: Vec<u32>, units: u64 },
    Buffer { offset: u64, bytes: u64 },
}

#[derive(Debug)]
enum AllocationBacking {
    RequestSlot(u32),
    Kv(Vec<BlockTable>),
    Latent(RequestId),
    Buffer { offset: u64, bytes: u64 },
}

/// An owned logical allocation. The handle is intentionally not cloneable;
/// request retirement consumes it through [`Memory::free`].
#[derive(Debug)]
pub struct Allocation {
    id: AllocationId,
    owner: RequestKey,
    layout: MemoryLayout,
    placement: Placement,
    backing: AllocationBacking,
}

impl Allocation {
    pub const fn id(&self) -> AllocationId {
        self.id
    }

    pub const fn owner(&self) -> RequestKey {
        self.owner
    }

    pub fn layout(&self) -> &MemoryLayout {
        &self.layout
    }

    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    pub(crate) fn request_slot(&self) -> Option<u32> {
        match self.backing {
            AllocationBacking::RequestSlot(index) => Some(index),
            _ => None,
        }
    }

    pub(crate) fn kv_tables(&self) -> Option<&[BlockTable]> {
        match &self.backing {
            AllocationBacking::Kv(tables) => Some(tables),
            _ => None,
        }
    }

    pub(crate) fn kv_tables_mut(&mut self) -> Option<&mut Vec<BlockTable>> {
        match &mut self.backing {
            AllocationBacking::Kv(tables) => Some(tables),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OutOfMemory {
    #[error("request slot capacity is exhausted")]
    RequestSlots,
    #[error("KV page capacity is exhausted")]
    Kv,
    #[error("latent page capacity is exhausted")]
    Latent,
    #[error("persistent buffer capacity is exhausted")]
    Buffer,
    #[error("the requested allocation layout is invalid")]
    InvalidLayout,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryStats {
    pub live_allocations: usize,
    pub free_request_slots: usize,
    pub free_kv_pages: usize,
    pub used_latent_pages: usize,
    pub free_buffer_bytes: u64,
}

pub(crate) struct RequestSlotPool {
    free: Vec<u32>,
    live: Vec<bool>,
}

impl RequestSlotPool {
    fn new(capacity: usize) -> Self {
        let capacity = capacity.clamp(1, u32::MAX as usize);
        Self {
            free: (1..=capacity as u32).rev().collect(),
            live: vec![false; capacity + 1],
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.live.len().saturating_sub(1)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.free.is_empty()
    }

    pub(crate) fn acquire(&mut self) -> Option<u32> {
        let index = self.free.pop()?;
        self.live[index as usize] = true;
        Some(index)
    }

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

pub(crate) struct LatentPagePool {
    page_units: u32,
    free: Vec<u32>,
    owners: Vec<Option<RequestId>>,
    allocations: HashMap<RequestId, Vec<u32>>,
}

impl LatentPagePool {
    fn new(num_pages: u32, page_units: u32) -> Self {
        Self {
            page_units,
            free: (1..num_pages).rev().collect(),
            owners: vec![None; num_pages as usize],
            allocations: HashMap::new(),
        }
    }

    fn pages_needed(&self, units: u64) -> Option<usize> {
        if units == 0 {
            return Some(0);
        }
        let page_units = u64::from(self.page_units);
        if page_units == 0 {
            return None;
        }
        usize::try_from(units.div_ceil(page_units)).ok()
    }

    pub(crate) fn can_reserve(&self, request_id: RequestId, units: u64) -> bool {
        let Some(needed) = self.pages_needed(units) else {
            return false;
        };
        let held = self.allocations.get(&request_id).map_or(0, Vec::len);
        needed.saturating_sub(held) <= self.free.len()
    }

    pub(crate) fn reserve(&mut self, request_id: RequestId, units: u64) -> bool {
        if !self.can_reserve(request_id, units) {
            return false;
        }
        let needed = self.pages_needed(units).unwrap_or_default();
        let held = self.allocations.get(&request_id).map_or(0, Vec::len);
        let mut pages = Vec::with_capacity(needed.saturating_sub(held));
        for _ in held..needed {
            let page = self.free.pop().expect("latent free-page invariant");
            self.owners[page as usize] = Some(request_id);
            pages.push(page);
        }
        self.allocations
            .entry(request_id)
            .or_default()
            .extend(pages);
        true
    }

    pub(crate) fn pages_for(&self, request_id: RequestId) -> &[u32] {
        self.allocations
            .get(&request_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub(crate) fn release(&mut self, request_id: RequestId) {
        if let Some(pages) = self.allocations.remove(&request_id) {
            for page in pages.into_iter().rev() {
                self.owners[page as usize] = None;
                self.free.push(page);
            }
        }
    }

    pub(crate) fn used_pages(&self) -> usize {
        self.owners
            .iter()
            .skip(1)
            .filter(|owner| owner.is_some())
            .count()
    }
}

pub(crate) struct KvMemoryState {
    pub(crate) block_pool: BlockPool,
    pub(crate) coordinator: KvCacheCoordinator,
    pub(crate) usable_blocks: usize,
}

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
                    let shape = (group.kind, offset, group.num_blocks);
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
    fn new(capacity: u64) -> Self {
        let free = (capacity > 0)
            .then_some((0, capacity))
            .into_iter()
            .collect();
        Self { capacity, free }
    }

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

    fn free_bytes(&self) -> u64 {
        self.free.values().copied().sum()
    }
}

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

    pub fn from_worker_info(info: &WorkerInfo) -> Self {
        Self::with_buffer_capacity(info, info.buffer_pool_bytes, info.request_slots as usize)
    }

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

    fn allocation_id(&mut self) -> AllocationId {
        let id = AllocationId(self.next_allocation_id);
        self.next_allocation_id = self.next_allocation_id.saturating_add(1);
        let inserted = self.live.insert(id);
        debug_assert!(inserted, "allocation identity reused");
        id
    }

    pub fn alloc(
        &mut self,
        owner: RequestKey,
        layout: MemoryLayout,
    ) -> Result<Allocation, OutOfMemory> {
        let recorded_layout = layout.clone();
        let (placement, backing) = match layout {
            MemoryLayout::RequestSlot => {
                let index = self
                    .request_slots
                    .acquire()
                    .ok_or(OutOfMemory::RequestSlots)?;
                (
                    Placement::RequestSlot { index },
                    AllocationBacking::RequestSlot(index),
                )
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
                let placement = Placement::Kv {
                    tables: tables.iter().map(BlockTable::page_ids).collect(),
                };
                (placement, AllocationBacking::Kv(tables))
            }
            MemoryLayout::Latent { units } => {
                if !self.latent_pages.reserve(owner.request_id, units) {
                    return Err(OutOfMemory::Latent);
                }
                let pages = self.latent_pages.pages_for(owner.request_id).to_vec();
                (
                    Placement::Latent { pages, units },
                    AllocationBacking::Latent(owner.request_id),
                )
            }
            MemoryLayout::Buffer { bytes, alignment } => {
                if bytes == 0 || alignment == 0 || !alignment.is_power_of_two() {
                    return Err(OutOfMemory::InvalidLayout);
                }
                let offset = self
                    .buffers
                    .alloc(bytes, alignment)
                    .ok_or(OutOfMemory::Buffer)?;
                (
                    Placement::Buffer { offset, bytes },
                    AllocationBacking::Buffer { offset, bytes },
                )
            }
        };
        let id = self.allocation_id();
        Ok(Allocation {
            id,
            owner,
            layout: recorded_layout,
            placement,
            backing,
        })
    }

    pub fn grow(
        &mut self,
        allocation: &mut Allocation,
        layout: MemoryLayout,
    ) -> Result<Placement, OutOfMemory> {
        if !self.live.contains(&allocation.id) {
            return Err(OutOfMemory::InvalidLayout);
        }
        let placement = match (&mut allocation.backing, &layout) {
            (AllocationBacking::RequestSlot(index), MemoryLayout::RequestSlot) => {
                Placement::RequestSlot { index: *index }
            }
            (AllocationBacking::Kv(tables), MemoryLayout::Kv { tokens, groups }) => {
                let cache = self.cache.as_ref().ok_or(OutOfMemory::Kv)?;
                if *groups as usize != tables.len() {
                    return Err(OutOfMemory::InvalidLayout);
                }
                cache
                    .coordinator
                    .ensure_capacity(&cache.block_pool, tables, *tokens as usize)
                    .ok_or(OutOfMemory::Kv)?;
                Placement::Kv {
                    tables: tables.iter().map(BlockTable::page_ids).collect(),
                }
            }
            (AllocationBacking::Latent(request_id), MemoryLayout::Latent { units }) => {
                if !self.latent_pages.reserve(*request_id, *units) {
                    return Err(OutOfMemory::Latent);
                }
                Placement::Latent {
                    pages: self.latent_pages.pages_for(*request_id).to_vec(),
                    units: *units,
                }
            }
            (
                AllocationBacking::Buffer { offset, bytes },
                MemoryLayout::Buffer {
                    bytes: requested,
                    alignment,
                },
            ) => {
                if *requested <= *bytes {
                    Placement::Buffer {
                        offset: *offset,
                        bytes: *requested,
                    }
                } else {
                    if *alignment == 0 || !alignment.is_power_of_two() {
                        return Err(OutOfMemory::InvalidLayout);
                    }
                    let new_offset = if self.buffers.grow_in_place(*offset, *bytes, *requested) {
                        *offset
                    } else {
                        let new_offset = self
                            .buffers
                            .alloc(*requested, *alignment)
                            .ok_or(OutOfMemory::Buffer)?;
                        self.buffers.free(*offset, *bytes);
                        new_offset
                    };
                    *offset = new_offset;
                    *bytes = *requested;
                    Placement::Buffer {
                        offset: new_offset,
                        bytes: *requested,
                    }
                }
            }
            _ => return Err(OutOfMemory::InvalidLayout),
        };
        allocation.layout = layout;
        allocation.placement = placement.clone();
        Ok(placement)
    }

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
            AllocationBacking::Latent(request_id) => self.latent_pages.release(request_id),
            AllocationBacking::Buffer { offset, bytes } => self.buffers.free(offset, bytes),
        }
    }

    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            live_allocations: self.live.len(),
            free_request_slots: self.request_slots.free.len(),
            free_kv_pages: self.free_blocks(),
            used_latent_pages: self.latent_pages.used_pages(),
            free_buffer_bytes: self.buffers.free_bytes(),
        }
    }

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

    pub(crate) fn take_encoder_buffer(&mut self, buffer: BufferId) -> Option<Allocation> {
        self.encoder_buffers.remove(&buffer)
    }

    pub(crate) fn cache(&self) -> &KvMemoryState {
        self.cache
            .as_ref()
            .expect("generation scheduling requires worker KV resources")
    }

    pub(crate) fn free_blocks(&self) -> usize {
        self.cache
            .as_ref()
            .map_or(0, |state| state.block_pool.free_request_pages())
    }

    pub(crate) fn usable_blocks(&self) -> usize {
        self.cache.as_ref().map_or(0, |state| state.usable_blocks)
    }

    pub(crate) fn set_prefix_cache(&mut self, enabled: bool) {
        if let Some(cache) = self.cache.as_mut() {
            cache.coordinator.set_prefix_enabled(enabled);
        }
    }

    pub(crate) fn set_hash_algo(&mut self, algo: HashAlgo) {
        if let Some(cache) = self.cache.as_mut() {
            cache.coordinator.set_hash_algo(algo);
        }
    }

    pub(crate) fn reset_after_worker_loss(&mut self, info: &WorkerInfo) {
        let coordinator = self.cache.take().map(|state| state.coordinator);
        let mut cache = worker_kv_state(info);
        if let (Some(cache), Some(coordinator)) = (&mut cache, coordinator) {
            cache.coordinator = coordinator;
        }
        let encoder_cache_budget = self.encoder_cache.budget();
        let request_pool_capacity = self.request_slots.capacity();
        let mut replacement = Self::new(
            cache,
            encoder_cache_budget,
            request_pool_capacity,
            info.latent_pages,
            info.latent_page_units,
        );
        replacement.buffers = BufferPool::new(info.buffer_pool_bytes);
        replacement.next_allocation_id = self.next_allocation_id;
        *self = replacement;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            first.placement(),
            &Placement::Buffer {
                offset: 0,
                bytes: 200
            }
        );
        assert_eq!(
            second.placement(),
            &Placement::Buffer {
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
        let placement = first.placement().clone();
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
        assert_eq!(first.placement(), &placement);
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
            Placement::Buffer {
                offset: 0,
                bytes: 768,
            }
        );
        memory.free(first);
        assert_eq!(memory.stats().free_buffer_bytes, 1_024);
    }
}
