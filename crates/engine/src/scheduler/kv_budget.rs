//! Scheduler-owned residency budgets for KV pages, encoder entries, request slots, and latents.

use super::*;

pub(super) struct RequestSlotPool {
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

    pub(super) fn capacity(&self) -> usize {
        self.live.len().saturating_sub(1)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.free.is_empty()
    }

    pub(super) fn acquire(&mut self) -> Option<u32> {
        let index = self.free.pop()?;
        self.live[index as usize] = true;
        Some(index)
    }

    pub(super) fn release(&mut self, index: u32) -> Result<(), &'static str> {
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

pub(super) struct LatentPagePool {
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

    pub(super) fn can_reserve(&self, request_id: RequestId, units: u64) -> bool {
        let Some(needed) = self.pages_needed(units) else {
            return false;
        };
        let held = self.allocations.get(&request_id).map_or(0, Vec::len);
        needed.saturating_sub(held) <= self.free.len()
    }

    pub(super) fn reserve(&mut self, request_id: RequestId, units: u64) -> bool {
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

    pub(super) fn pages_for(&self, request_id: RequestId) -> &[u32] {
        self.allocations
            .get(&request_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub(super) fn release(&mut self, request_id: RequestId) {
        if let Some(pages) = self.allocations.remove(&request_id) {
            for page in pages.into_iter().rev() {
                self.owners[page as usize] = None;
                self.free.push(page);
            }
        }
    }

    pub(super) fn used_pages(&self) -> usize {
        self.owners
            .iter()
            .skip(1)
            .filter(|owner| owner.is_some())
            .count()
    }
}

pub(super) struct KvSchedulerState {
    pub(super) block_pool: BlockPool,
    pub(super) coordinator: KvCacheCoordinator,
    pub(super) usable_blocks: usize,
}

pub(super) fn worker_kv_state(info: &WorkerInfo) -> Option<KvSchedulerState> {
    info.uses_kv().then(|| {
        let block_pool = if info.groups.is_empty() {
            BlockPool::new(info.num_blocks as usize, info.block_size as usize)
        } else {
            let mut offset = 0_u32;
            let group_shapes = info
                .groups
                .iter()
                .map(|group| {
                    let shape = (group.kind, offset, group.num_blocks);
                    offset = offset.saturating_add(group.num_blocks);
                    shape
                })
                .collect::<Vec<_>>();
            BlockPool::with_groups(
                info.num_blocks as usize,
                info.block_size as usize,
                &group_shapes,
            )
        };
        let usable_blocks = block_pool.request_page_capacity();
        KvSchedulerState {
            block_pool,
            coordinator: KvCacheCoordinator::default(),
            usable_blocks,
        }
    })
}

pub(super) struct KvBudget {
    pub(super) cache: Option<KvSchedulerState>,
    pub(super) encoder_cache: EncoderCacheManager,
    pub(super) reserved_encoder_entries: usize,
    pub(super) request_slots: RequestSlotPool,
    pub(super) latent_pages: LatentPagePool,
    pub(super) reserved_blocks: usize,
}

impl KvBudget {
    pub(super) fn new(
        cache: Option<KvSchedulerState>,
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
        }
    }

    pub(super) fn cache(&self) -> &KvSchedulerState {
        self.cache
            .as_ref()
            .expect("generation scheduling requires worker KV resources")
    }

    pub(super) fn free_blocks(&self) -> usize {
        self.cache
            .as_ref()
            .map_or(0, |state| state.block_pool.free_request_pages())
    }

    pub(super) fn usable_blocks(&self) -> usize {
        self.cache.as_ref().map_or(0, |state| state.usable_blocks)
    }

    pub(super) fn set_prefix_cache(&mut self, enabled: bool) {
        if let Some(cache) = self.cache.as_mut() {
            cache.coordinator.set_prefix_enabled(enabled);
        }
    }

    pub(super) fn set_hash_algo(&mut self, algo: HashAlgo) {
        if let Some(cache) = self.cache.as_mut() {
            cache.coordinator.set_hash_algo(algo);
        }
    }

    pub(super) fn reset_after_worker_loss(&mut self, info: &WorkerInfo) {
        let coordinator = self.cache.take().map(|state| state.coordinator);
        let mut cache = worker_kv_state(info);
        if let (Some(cache), Some(coordinator)) = (&mut cache, coordinator) {
            cache.coordinator = coordinator;
        }
        let encoder_cache_budget = self.encoder_cache.budget();
        let request_pool_capacity = self.request_slots.capacity();
        *self = Self::new(
            cache,
            encoder_cache_budget,
            request_pool_capacity,
            info.num_latent_pages,
            info.latent_page_units,
        );
    }

    pub(super) fn release_transition(&mut self, id: RequestId, apply: &SchedulerApply) {
        for class in &apply.release_on_apply {
            if *class == uniserve_worker_ipc::ResourceClass::ImageLatent {
                self.latent_pages.release(id);
            }
        }
    }
}
