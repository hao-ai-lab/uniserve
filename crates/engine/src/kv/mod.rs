//! Scheduler-side ownership for paged key/value cache memory.
//!
//! [`BlockPool`] owns physical page availability and prefix-cache metadata.
//! [`BlockTable`] owns one cached sequence's logical-to-physical mapping. A
//! table contains reference-counted [`CacheBlockRef`] handles, so sharing a
//! prefix is ordinary Rust ownership and a page returns to the pool when its
//! last table owner is dropped.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use uniserve_core::{BlockId, HashAlgo, KvGroupKind, Modality};

mod encoder_cache;
mod freeq;

pub(crate) use encoder_cache::EncoderCacheManager;
use freeq::BlockMeta;

/// Incremental per-page prefix hash.
pub(crate) fn block_hash(
    parent: u64,
    group_id: u32,
    modality_tag: u8,
    tokens: &[u32],
    algo: HashAlgo,
) -> u64 {
    let token_count = tokens.len() as u32;
    match algo {
        HashAlgo::Fnv1a => {
            const OFFSET: u64 = 0xcbf29ce484222325;
            const PRIME: u64 = 0x00000100000001B3;
            let mut h = OFFSET;
            let mix = |bytes: &[u8], h: &mut u64| {
                for b in bytes {
                    *h ^= u64::from(*b);
                    *h = h.wrapping_mul(PRIME);
                }
            };
            mix(&parent.to_le_bytes(), &mut h);
            mix(&group_id.to_le_bytes(), &mut h);
            mix(&[modality_tag], &mut h);
            mix(&token_count.to_le_bytes(), &mut h);
            for token in tokens {
                mix(&token.to_le_bytes(), &mut h);
            }
            h
        }
        HashAlgo::Sha256 => {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(parent.to_le_bytes());
            hasher.update(group_id.to_le_bytes());
            hasher.update([modality_tag]);
            hasher.update(token_count.to_le_bytes());
            for token in tokens {
                hasher.update(token.to_le_bytes());
            }
            let digest = hasher.finalize();
            u64::from_le_bytes(
                digest[..8]
                    .try_into()
                    .expect("SHA-256 prefix is eight bytes"),
            )
        }
    }
}

pub(crate) fn modality_tag(modality: Modality) -> u8 {
    match modality {
        Modality::Und => 0,
        Modality::Gen => 1,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockState {
    Free,
    Reserved,
    Active,
    Cached,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CacheEvent {
    BlockStored { hash: u64, block: BlockId },
    BlockRemoved { hash: u64, block: BlockId },
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockPoolStats {
    pub(crate) total: usize,
    pub(crate) free: usize,
    pub(crate) allocations: u64,
    pub(crate) evictions: u64,
    pub(crate) blocks_stored: u64,
}

struct Group {
    kind: KvGroupKind,
    fq_head: Option<u32>,
    fq_tail: Option<u32>,
    free_count: usize,
    total: usize,
}

struct PoolInner {
    block_size: usize,
    num_blocks: usize,
    meta: Vec<BlockMeta>,
    groups: Vec<Group>,
    block_group: Vec<u32>,
    hash_to_block: HashMap<u64, BlockId>,
    events: VecDeque<CacheEvent>,
    events_cap: usize,
    stats: BlockPoolStats,
}

impl PoolInner {
    fn fq_push_back(&mut self, group: usize, id: BlockId) {
        let index = id.0;
        let tail = self.groups[group].fq_tail;
        self.meta[index as usize].fq_prev = tail;
        self.meta[index as usize].fq_next = None;
        self.meta[index as usize].in_fq = true;
        if let Some(tail) = tail {
            self.meta[tail as usize].fq_next = Some(index);
        } else {
            self.groups[group].fq_head = Some(index);
        }
        self.groups[group].fq_tail = Some(index);
        self.groups[group].free_count += 1;
    }

    fn fq_unlink_index(&mut self, group: usize, index: u32) {
        let previous = self.meta[index as usize].fq_prev;
        let next = self.meta[index as usize].fq_next;
        if let Some(previous) = previous {
            self.meta[previous as usize].fq_next = next;
        } else {
            self.groups[group].fq_head = next;
        }
        if let Some(next) = next {
            self.meta[next as usize].fq_prev = previous;
        } else {
            self.groups[group].fq_tail = previous;
        }
        let meta = &mut self.meta[index as usize];
        meta.fq_prev = None;
        meta.fq_next = None;
        meta.in_fq = false;
        self.groups[group].free_count = self.groups[group].free_count.saturating_sub(1);
    }

    fn fq_pop_front(&mut self, group: usize) -> Option<BlockId> {
        let index = self.groups[group].fq_head?;
        self.fq_unlink_index(group, index);
        Some(BlockId(index))
    }

    fn fq_unlink(&mut self, id: BlockId) {
        if self.meta[id.0 as usize].in_fq {
            let group = self.block_group[id.0 as usize] as usize;
            self.fq_unlink_index(group, id.0);
        }
    }

    fn total_free(&self) -> usize {
        self.groups.iter().map(|group| group.free_count).sum()
    }

    fn push_event(&mut self, event: CacheEvent) {
        if self.events.len() == self.events_cap {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    fn release_ref(&mut self, id: BlockId) {
        let meta = &mut self.meta[id.0 as usize];
        debug_assert!(meta.ref_cnt > 0, "cache page reference count underflow");
        meta.ref_cnt = meta.ref_cnt.saturating_sub(1);
        if meta.ref_cnt != 0 || meta.in_fq {
            return;
        }
        let group = self.block_group[id.0 as usize] as usize;
        self.meta[id.0 as usize].state = if self.meta[id.0 as usize].hash.is_some() {
            BlockState::Cached
        } else {
            BlockState::Free
        };
        self.fq_push_back(group, id);
        self.stats.free = self.total_free();
    }
}

/// A reference-counted physical page owned by a scheduler [`BlockTable`].
pub(crate) struct CacheBlockRef {
    id: BlockId,
    pool: Weak<Mutex<PoolInner>>,
}

impl CacheBlockRef {
    pub(crate) fn id(&self) -> BlockId {
        self.id
    }
}

impl Clone for CacheBlockRef {
    fn clone(&self) -> Self {
        if let Some(pool) = self.pool.upgrade() {
            let mut inner = lock(&pool);
            let meta = &mut inner.meta[self.id.0 as usize];
            meta.ref_cnt = meta
                .ref_cnt
                .checked_add(1)
                .expect("cache page reference count overflow");
        }
        Self {
            id: self.id,
            pool: self.pool.clone(),
        }
    }
}

impl Drop for CacheBlockRef {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            lock(&pool).release_ref(self.id);
        }
    }
}

impl fmt::Debug for CacheBlockRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("CacheBlockRef")
            .field(&self.id)
            .finish()
    }
}

impl PartialEq for CacheBlockRef {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && Weak::ptr_eq(&self.pool, &other.pool)
    }
}

impl Eq for CacheBlockRef {}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum BlockPoolConfigError {
    #[error("physical KV capacity and every cache group must be positive")]
    EmptyCapacity,
    #[error("cache group {group} range overflows")]
    RangeOverflow { group: usize },
    #[error("cache group {group} range [{first}, {end}) exceeds capacity {capacity}")]
    RangeExceedsCapacity {
        group: usize,
        first: u32,
        end: u64,
        capacity: usize,
    },
    #[error("cache group {group} page {page} overlaps another group")]
    Overlap { group: usize, page: usize },
    #[error("physical KV page {page} is not covered by a cache group")]
    UncoveredPage { page: usize },
}

/// Physical KV pages, free capacity, page references, and prefix-cache metadata.
#[derive(Clone)]
pub(crate) struct BlockPool {
    inner: Arc<Mutex<PoolInner>>,
}

impl BlockPool {
    pub(crate) fn new(num_blocks: usize, block_size: usize) -> Self {
        Self::with_groups(
            num_blocks,
            block_size,
            &[(
                KvGroupKind::Full,
                0,
                u32::try_from(num_blocks).expect("physical KV page capacity exceeds u32"),
            )],
        )
    }

    pub(crate) fn validate_group_specs(
        num_blocks: usize,
        group_specs: &[(KvGroupKind, u32, u32)],
    ) -> Result<(), BlockPoolConfigError> {
        if num_blocks == 0 || block_size_invalid(group_specs) {
            return Err(BlockPoolConfigError::EmptyCapacity);
        }
        let mut covered = vec![false; num_blocks];
        for (group, (_, first, count)) in group_specs.iter().enumerate() {
            let end = u64::from(*first)
                .checked_add(u64::from(*count))
                .ok_or(BlockPoolConfigError::RangeOverflow { group })?;
            if end > num_blocks as u64 {
                return Err(BlockPoolConfigError::RangeExceedsCapacity {
                    group,
                    first: *first,
                    end,
                    capacity: num_blocks,
                });
            }
            for (page, occupied) in covered
                .iter_mut()
                .enumerate()
                .take(end as usize)
                .skip(*first as usize)
            {
                if *occupied {
                    return Err(BlockPoolConfigError::Overlap { group, page });
                }
                *occupied = true;
            }
        }
        if let Some(page) = covered.iter().position(|occupied| !occupied) {
            return Err(BlockPoolConfigError::UncoveredPage { page });
        }
        Ok(())
    }

    pub(crate) fn with_groups(
        num_blocks: usize,
        block_size: usize,
        group_specs: &[(KvGroupKind, u32, u32)],
    ) -> Self {
        assert!(block_size > 0, "KV block size must be positive");
        if let Err(error) = Self::validate_group_specs(num_blocks, group_specs) {
            panic!("invalid KV cache groups: {error}");
        }
        let meta = (0..num_blocks)
            .map(|_| BlockMeta::new(BlockState::Free))
            .collect::<Vec<_>>();
        let mut block_group = vec![0; num_blocks];
        let mut groups = Vec::with_capacity(group_specs.len());
        for (group, (kind, first, count)) in group_specs.iter().enumerate() {
            for page in *first..(*first + *count) {
                block_group[page as usize] = group as u32;
            }
            groups.push(Group {
                kind: *kind,
                fq_head: None,
                fq_tail: None,
                free_count: 0,
                total: *count as usize,
            });
        }
        let mut inner = PoolInner {
            block_size,
            num_blocks,
            meta,
            groups,
            block_group,
            hash_to_block: HashMap::new(),
            events: VecDeque::new(),
            events_cap: 4096,
            stats: BlockPoolStats {
                total: num_blocks.saturating_sub(1),
                ..BlockPoolStats::default()
            },
        };
        for (group, (_, first, count)) in group_specs.iter().enumerate() {
            for page in *first..(*first + *count) {
                if page != 0 {
                    inner.fq_push_back(group, BlockId(page));
                }
            }
            if *first == 0 {
                inner.groups[group].total = inner.groups[group].total.saturating_sub(1);
            }
        }
        inner.stats.free = inner.total_free();
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    pub(crate) fn block_size(&self) -> usize {
        lock(&self.inner).block_size
    }

    pub(crate) fn num_blocks(&self) -> usize {
        lock(&self.inner).num_blocks
    }

    pub(crate) fn num_groups(&self) -> usize {
        lock(&self.inner).groups.len()
    }

    pub(crate) fn group_kind(&self, group: usize) -> Option<KvGroupKind> {
        lock(&self.inner).groups.get(group).map(|value| value.kind)
    }

    pub(crate) fn group_capacity(&self, group: usize) -> usize {
        lock(&self.inner)
            .groups
            .get(group)
            .map_or(0, |value| value.total)
    }

    pub(crate) fn request_page_capacity(&self) -> usize {
        let inner = lock(&self.inner);
        inner.groups.iter().map(|group| group.total).sum()
    }

    pub(crate) fn free_blocks(&self) -> usize {
        lock(&self.inner).total_free()
    }

    pub(crate) fn free_blocks_in_group(&self, group: usize) -> usize {
        lock(&self.inner)
            .groups
            .get(group)
            .map_or(0, |value| value.free_count)
    }

    pub(crate) fn free_request_pages(&self) -> usize {
        let inner = lock(&self.inner);
        inner
            .groups
            .iter()
            .map(|group| group.free_count)
            .min()
            .unwrap_or(0)
    }

    pub(crate) fn blocks_needed(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_size())
    }

    pub(crate) fn stats(&self) -> BlockPoolStats {
        lock(&self.inner).stats
    }

    pub(crate) fn allocate(&self, group: usize, count: usize) -> Option<Vec<CacheBlockRef>> {
        let mut inner = lock(&self.inner);
        if group >= inner.groups.len() || count > inner.groups[group].free_count {
            return None;
        }
        let mut refs = Vec::with_capacity(count);
        for _ in 0..count {
            let page = inner.fq_pop_front(group).expect("free-count invariant");
            if let Some(hash) = inner.meta[page.0 as usize].hash.take() {
                inner.meta[page.0 as usize].tokens.clear();
                if inner.hash_to_block.get(&hash) == Some(&page) {
                    inner.hash_to_block.remove(&hash);
                }
                inner.stats.evictions += 1;
                inner.push_event(CacheEvent::BlockRemoved { hash, block: page });
            }
            let meta = &mut inner.meta[page.0 as usize];
            meta.state = BlockState::Reserved;
            meta.ref_cnt = 1;
            refs.push(CacheBlockRef {
                id: page,
                pool: Arc::downgrade(&self.inner),
            });
        }
        inner.stats.allocations += 1;
        inner.stats.free = inner.total_free();
        Some(refs)
    }

    pub(crate) fn activate(&self, blocks: &[CacheBlockRef]) {
        let mut inner = lock(&self.inner);
        for block in blocks {
            if Arc::ptr_eq(
                &self.inner,
                &block.pool.upgrade().expect("cache pool is live"),
            ) && inner.meta[block.id.0 as usize].state == BlockState::Reserved
            {
                inner.meta[block.id.0 as usize].state = BlockState::Active;
            }
        }
    }

    pub(crate) fn ref_count(&self, block: BlockId) -> u32 {
        lock(&self.inner).meta[block.0 as usize].ref_cnt
    }

    pub(crate) fn block_state(&self, block: BlockId) -> BlockState {
        lock(&self.inner).meta[block.0 as usize].state
    }

    pub(crate) fn block_group(&self, block: BlockId) -> Option<usize> {
        lock(&self.inner)
            .block_group
            .get(block.0 as usize)
            .map(|group| *group as usize)
    }

    pub(crate) fn cache_block(&self, block: &CacheBlockRef, hash: u64, tokens: &[u32]) {
        let mut inner = lock(&self.inner);
        if inner.hash_to_block.contains_key(&hash) {
            return;
        }
        let meta = &mut inner.meta[block.id.0 as usize];
        meta.hash = Some(hash);
        meta.tokens.clear();
        meta.tokens.extend_from_slice(tokens);
        inner.hash_to_block.insert(hash, block.id);
        inner.stats.blocks_stored += 1;
        inner.push_event(CacheEvent::BlockStored {
            hash,
            block: block.id,
        });
    }

    pub(crate) fn lookup_cached(&self, hash: u64, tokens: &[u32]) -> Option<BlockId> {
        let inner = lock(&self.inner);
        let block = *inner.hash_to_block.get(&hash)?;
        (inner.meta[block.0 as usize].tokens == tokens).then_some(block)
    }

    pub(crate) fn acquire_cached(
        &self,
        block: BlockId,
        hash: u64,
        tokens: &[u32],
    ) -> Option<CacheBlockRef> {
        let mut inner = lock(&self.inner);
        let meta = inner.meta.get(block.0 as usize)?;
        if meta.hash != Some(hash) || meta.tokens != tokens {
            return None;
        }
        inner.fq_unlink(block);
        let meta = &mut inner.meta[block.0 as usize];
        meta.ref_cnt = meta.ref_cnt.checked_add(1)?;
        meta.state = BlockState::Active;
        inner.stats.free = inner.total_free();
        Some(CacheBlockRef {
            id: block,
            pool: Arc::downgrade(&self.inner),
        })
    }

    pub(crate) fn cached_blocks(&self) -> usize {
        lock(&self.inner).hash_to_block.len()
    }

    pub(crate) fn drain_events(&self) -> Vec<CacheEvent> {
        lock(&self.inner).events.drain(..).collect()
    }
}

impl fmt::Debug for BlockPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlockPool")
            .field("num_blocks", &self.num_blocks())
            .field("block_size", &self.block_size())
            .field("num_groups", &self.num_groups())
            .finish()
    }
}

/// One cached sequence's logical-page to physical-page mapping.
#[derive(Debug, Clone)]
pub(crate) struct BlockTable {
    group_id: usize,
    block_size: usize,
    blocks: Vec<CacheBlockRef>,
    token_starts: Option<Vec<usize>>,
}

impl BlockTable {
    pub(crate) fn new(group_id: usize, block_size: usize) -> Self {
        assert!(block_size > 0, "KV block size must be positive");
        Self {
            group_id,
            block_size,
            blocks: Vec::new(),
            token_starts: None,
        }
    }

    pub(crate) fn group_id(&self) -> usize {
        self.group_id
    }

    pub(crate) fn len(&self) -> usize {
        self.blocks.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub(crate) fn capacity_tokens(&self) -> usize {
        self.blocks.len() * self.block_size
    }

    pub(crate) fn page_ids(&self) -> Vec<BlockId> {
        self.blocks.iter().map(CacheBlockRef::id).collect()
    }

    pub(crate) fn block(&self, index: usize) -> Option<&CacheBlockRef> {
        self.blocks.get(index)
    }

    pub(crate) fn contains(&self, block: BlockId) -> bool {
        self.blocks.iter().any(|candidate| candidate.id() == block)
    }

    pub(crate) fn append_cached(&mut self, block: CacheBlockRef) -> bool {
        if self.contains(block.id()) {
            return false;
        }
        self.blocks.push(block);
        self.token_starts = None;
        true
    }

    pub(crate) fn ensure_capacity(
        &mut self,
        pool: &BlockPool,
        total_tokens: usize,
    ) -> Option<Vec<BlockId>> {
        let required = total_tokens.div_ceil(self.block_size);
        let additional = required.saturating_sub(self.blocks.len());
        let acquired = pool.allocate(self.group_id, additional)?;
        let page_ids = acquired.iter().map(CacheBlockRef::id).collect();
        self.blocks.extend(acquired);
        self.token_starts = None;
        Some(page_ids)
    }

    pub(crate) fn activate(&self, pool: &BlockPool) {
        pool.activate(&self.blocks);
    }

    pub(crate) fn trim(&mut self, pool: &BlockPool, position_tokens: usize) {
        let Some(KvGroupKind::SlidingWindow { window, sink }) = pool.group_kind(self.group_id)
        else {
            return;
        };
        let starts = self.token_starts.clone();
        let mut kept = Vec::with_capacity(self.blocks.len());
        let mut kept_starts = Vec::with_capacity(self.blocks.len());
        for (index, block) in self.blocks.drain(..).enumerate() {
            let start = starts
                .as_ref()
                .and_then(|values| values.get(index).copied())
                .unwrap_or(index * self.block_size);
            let end = start + self.block_size;
            if start < sink as usize || end > position_tokens.saturating_sub(window as usize) {
                kept.push(block);
                kept_starts.push(start);
            }
        }
        self.blocks = kept;
        self.token_starts = Some(kept_starts);
    }

    pub(crate) fn clear(&mut self) {
        self.blocks.clear();
        self.token_starts = None;
    }
}

/// Prefix lookup result for one scheduler-owned sequence.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PrefixLookup {
    pub(crate) block_hashes: Vec<Vec<u64>>,
    pub(crate) cached_blocks: usize,
}

/// Read-only prefix-cache probe result across every physical cache group.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PrefixHit {
    pub(crate) cached_blocks: usize,
    pub(crate) cached_free_blocks: Vec<usize>,
}

/// Coordinates capacity and prefix state across physical layout groups.
#[derive(Debug)]
pub(crate) struct KvCacheCoordinator {
    prefix_enabled: bool,
    hash_algo: HashAlgo,
    hash_seed: u64,
}

impl Default for KvCacheCoordinator {
    fn default() -> Self {
        Self {
            prefix_enabled: true,
            hash_algo: HashAlgo::Fnv1a,
            hash_seed: 0,
        }
    }
}

impl KvCacheCoordinator {
    pub(crate) fn set_prefix_enabled(&mut self, enabled: bool) {
        self.prefix_enabled = enabled;
    }

    pub(crate) fn set_hash_algo(&mut self, algorithm: HashAlgo) {
        self.hash_algo = algorithm;
    }

    /// Atomically grows every group table to the common token boundary and
    /// returns only the pages acquired by this allocation event.
    pub(crate) fn ensure_capacity(
        &self,
        pool: &BlockPool,
        tables: &mut [BlockTable],
        total_tokens: usize,
    ) -> Option<Vec<(u32, Vec<BlockId>)>> {
        if tables.len() != pool.num_groups()
            || tables
                .iter()
                .enumerate()
                .any(|(group, table)| table.group_id != group)
        {
            return None;
        }
        let required = pool.blocks_needed(total_tokens);
        if tables.iter().enumerate().any(|(group, table)| {
            required.saturating_sub(table.len()) > pool.free_blocks_in_group(group)
        }) {
            return None;
        }
        let original_lengths = tables.iter().map(BlockTable::len).collect::<Vec<_>>();
        let mut updates = Vec::with_capacity(tables.len());
        for (group, table) in tables.iter_mut().enumerate() {
            let Some(pages) = table.ensure_capacity(pool, total_tokens) else {
                for (table, original) in tables.iter_mut().zip(original_lengths) {
                    table.blocks.truncate(original);
                }
                return None;
            };
            updates.push((group as u32, pages));
        }
        Some(updates)
    }

    pub(crate) fn probe_prefix(
        &self,
        pool: &BlockPool,
        prompt: &[u32],
        cache_read: bool,
        has_images: bool,
        isolation_key: Option<u64>,
    ) -> PrefixHit {
        let groups = pool.num_groups();
        let mut hit = PrefixHit {
            cached_free_blocks: vec![0; groups],
            ..PrefixHit::default()
        };
        if !self.prefix_enabled || !cache_read || has_images {
            return hit;
        }
        let block_size = pool.block_size();
        let limit = prefix_lookup_limit(prompt.len(), block_size);
        if limit == 0 {
            return hit;
        }
        let hashes = self.prefix_hashes(prompt, groups, block_size, isolation_key);
        for index in 0..limit {
            let tokens = &prompt[index * block_size..(index + 1) * block_size];
            let mut blocks = Vec::with_capacity(groups);
            for (group, group_hashes) in hashes.iter().enumerate() {
                let Some(block) = pool.lookup_cached(group_hashes[index], tokens) else {
                    return hit;
                };
                if pool.block_group(block) != Some(group) {
                    return hit;
                }
                blocks.push(block);
            }
            for (group, block) in blocks.into_iter().enumerate() {
                if pool.block_state(block) == BlockState::Cached {
                    hit.cached_free_blocks[group] += 1;
                }
            }
            hit.cached_blocks += 1;
        }
        hit
    }

    pub(crate) fn acquire_prefix(
        &self,
        pool: &BlockPool,
        tables: &mut [BlockTable],
        prompt: &[u32],
        cache_read: bool,
        has_images: bool,
        isolation_key: Option<u64>,
    ) -> Option<PrefixLookup> {
        let groups = pool.num_groups();
        if tables.len() != groups
            || tables
                .iter()
                .enumerate()
                .any(|(group, table)| table.group_id() != group)
        {
            return None;
        }
        if !self.prefix_enabled || has_images {
            return Some(PrefixLookup::default());
        }
        let block_size = pool.block_size();
        let hashes = self.prefix_hashes(prompt, groups, block_size, isolation_key);
        let mut result = PrefixLookup {
            block_hashes: hashes,
            cached_blocks: 0,
        };
        if !cache_read {
            return Some(result);
        }
        for index in 0..prefix_lookup_limit(prompt.len(), block_size) {
            let tokens = &prompt[index * block_size..(index + 1) * block_size];
            let mut candidates = Vec::with_capacity(groups);
            for (group, table) in tables.iter().enumerate() {
                let hash = result.block_hashes[group][index];
                let Some(block) = pool.lookup_cached(hash, tokens) else {
                    return Some(result);
                };
                if pool.block_group(block) != Some(group) || table.contains(block) {
                    return Some(result);
                }
                candidates.push((block, hash));
            }
            let mut references = Vec::with_capacity(groups);
            for (block, hash) in candidates {
                let Some(reference) = pool.acquire_cached(block, hash, tokens) else {
                    return Some(result);
                };
                references.push(reference);
            }
            for (table, reference) in tables.iter_mut().zip(references) {
                if !table.append_cached(reference) {
                    return None;
                }
            }
            result.cached_blocks += 1;
        }
        Some(result)
    }

    pub(crate) fn cache_prefix(
        &self,
        pool: &BlockPool,
        tables: &[BlockTable],
        prompt: &[u32],
        hashes: &[Vec<u64>],
        cache_write: bool,
    ) -> bool {
        if !self.prefix_enabled || !cache_write {
            return true;
        }
        let groups = pool.num_groups();
        if tables.len() != groups || hashes.len() != groups {
            return false;
        }
        let block_size = pool.block_size();
        for (group, (table, group_hashes)) in tables.iter().zip(hashes).enumerate() {
            if table.group_id() != group {
                return false;
            }
            for (index, hash) in group_hashes.iter().enumerate() {
                let start = index * block_size;
                let end = start + block_size;
                let (Some(block), Some(tokens)) = (table.block(index), prompt.get(start..end))
                else {
                    continue;
                };
                pool.cache_block(block, *hash, tokens);
            }
        }
        true
    }

    fn prefix_hashes(
        &self,
        prompt: &[u32],
        groups: usize,
        block_size: usize,
        isolation_key: Option<u64>,
    ) -> Vec<Vec<u64>> {
        let block_count = prompt.len() / block_size;
        let seed = isolation_key.map_or(self.hash_seed, |key| self.hash_seed ^ key.rotate_left(17));
        let mut parents = vec![seed; groups];
        let mut hashes = vec![Vec::with_capacity(block_count); groups];
        for tokens in prompt.chunks_exact(block_size) {
            for group in 0..groups {
                let hash = block_hash(
                    parents[group],
                    group as u32,
                    modality_tag(Modality::Und),
                    tokens,
                    self.hash_algo,
                );
                hashes[group].push(hash);
                parents[group] = hash;
            }
        }
        hashes
    }
}

fn prefix_lookup_limit(prompt_tokens: usize, block_size: usize) -> usize {
    let full_blocks = prompt_tokens / block_size;
    if prompt_tokens.is_multiple_of(block_size) {
        full_blocks.saturating_sub(1)
    } else {
        full_blocks
    }
}

fn block_size_invalid(group_specs: &[(KvGroupKind, u32, u32)]) -> bool {
    group_specs.is_empty() || group_specs.iter().any(|(_, _, count)| *count == 0)
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_tables_keep_pages_live_until_the_last_owner_releases() {
        let pool = BlockPool::new(4, 4);
        let mut first = BlockTable::new(0, 4);
        assert_eq!(first.ensure_capacity(&pool, 4), Some(vec![BlockId(1)]));
        let second = first.clone();
        assert_eq!(pool.ref_count(BlockId(1)), 2);
        first.clear();
        assert_eq!(pool.ref_count(BlockId(1)), 1);
        assert_eq!(pool.free_blocks(), 2);
        drop(second);
        assert_eq!(pool.ref_count(BlockId(1)), 0);
        assert_eq!(pool.free_blocks(), 3);
    }

    #[test]
    fn coordinator_allocates_complete_group_tables_atomically() {
        let pool = BlockPool::with_groups(
            7,
            4,
            &[(KvGroupKind::Full, 0, 4), (KvGroupKind::Full, 4, 3)],
        );
        let mut tables = vec![BlockTable::new(0, 4), BlockTable::new(1, 4)];
        let coordinator = KvCacheCoordinator::default();
        assert!(coordinator.ensure_capacity(&pool, &mut tables, 8).is_some());
        assert_eq!(tables[0].len(), 2);
        assert_eq!(tables[1].len(), 2);
        assert!(
            coordinator
                .ensure_capacity(&pool, &mut tables, 16)
                .is_none()
        );
        assert_eq!(tables[0].len(), 2);
        assert_eq!(tables[1].len(), 2);
    }

    #[test]
    fn cached_prefix_reference_shares_the_physical_page() {
        let pool = BlockPool::new(4, 4);
        let mut first = BlockTable::new(0, 4);
        first.ensure_capacity(&pool, 4).unwrap();
        let hash = block_hash(0, 0, 0, &[1, 2, 3, 4], HashAlgo::Fnv1a);
        pool.cache_block(first.block(0).unwrap(), hash, &[1, 2, 3, 4]);
        let block = pool.lookup_cached(hash, &[1, 2, 3, 4]).unwrap();
        let reference = pool.acquire_cached(block, hash, &[1, 2, 3, 4]).unwrap();
        let mut second = BlockTable::new(0, 4);
        assert!(second.append_cached(reference));
        assert_eq!(first.page_ids(), second.page_ids());
        assert_eq!(pool.ref_count(block), 2);
    }

    #[test]
    fn group_validation_rejects_gaps_and_overlap() {
        use KvGroupKind::Full;
        assert!(BlockPool::validate_group_specs(8, &[(Full, 0, 3), (Full, 3, 5)]).is_ok());
        assert!(BlockPool::validate_group_specs(8, &[(Full, 0, 4), (Full, 3, 5)]).is_err());
        assert!(BlockPool::validate_group_specs(8, &[(Full, 0, 7)]).is_err());
    }

    #[test]
    fn prefix_reuse_advances_only_at_a_boundary_cached_in_every_group() {
        let pool = BlockPool::with_groups(
            7,
            4,
            &[(KvGroupKind::Full, 0, 4), (KvGroupKind::Full, 4, 3)],
        );
        let coordinator = KvCacheCoordinator::default();
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut source = vec![BlockTable::new(0, 4), BlockTable::new(1, 4)];
        coordinator
            .ensure_capacity(&pool, &mut source, prompt.len())
            .unwrap();
        let hashes = coordinator.prefix_hashes(&prompt, 2, 4, None);
        assert!(coordinator.cache_prefix(&pool, &source, &prompt, &hashes, true));

        let hit = coordinator.probe_prefix(&pool, &prompt, true, false, None);
        assert_eq!(hit.cached_blocks, 1);
        assert_eq!(hit.cached_free_blocks, vec![0, 0]);

        let mut target = vec![BlockTable::new(0, 4), BlockTable::new(1, 4)];
        let lookup = coordinator
            .acquire_prefix(&pool, &mut target, &prompt, true, false, None)
            .unwrap();
        assert_eq!(lookup.cached_blocks, 1);
        assert_eq!(target[0].page_ids(), source[0].page_ids()[..1]);
        assert_eq!(target[1].page_ids(), source[1].page_ids()[..1]);
    }
}
