//! Scheduler-owned paged key/value cache allocation and prefix reuse.
//!
//! The scheduler chooses every physical KV page before a call is dispatched;
//! workers read and write only the pages named in a call's block tables.
//! [`BlockPool`] owns physical page availability and prefix-cache metadata.
//! [`BlockTable`] maps one sequence's logical blocks to reference-counted
//! physical pages. Dropping the final page reference returns it to the pool.
//! [`KvCacheCoordinator`] applies allocation and prefix reuse across every
//! cache group at once.
//!
//! Key concepts:
//!
//! - Cache groups: the worker reports a page count and attention kind
//!   (`KvGroupKind`) per group, and the groups partition the physical page
//!   range into consecutive subranges. A request holds one table per group,
//!   and all of its tables cover the same number of logical blocks, so one
//!   logical block consumes one page in every group.
//! - Page zero: the worker treats physical page zero as its padding sentinel,
//!   so the pool never allocates it and excludes it from capacity.
//! - Free queues: each group keeps an intrusive doubly linked queue of
//!   unreferenced pages. Allocation takes pages from the head and released
//!   pages join the tail. An unreferenced page that still holds a published
//!   prefix ([`BlockState::Cached`]) therefore stays reusable until
//!   allocation reaches it, and allocation reuses pages in release order. A
//!   prefix hit on an unreferenced page unlinks it from wherever it sits in
//!   the queue.
//! - Prefix identity: a published page is keyed by a chained 64-bit hash, but
//!   a match also requires equal tokens and the same loaded `WorkerEndpoint`.
//!   The token comparison guards against hash collisions, and the endpoint
//!   comparison confines reuse to the loaded worker incarnation that computed
//!   and retains the page.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use uniserve_core::{BlockId, HashAlgo, Modality};
use uniserve_worker_ipc::WorkerEndpoint;

mod encoder_cache;
mod freeq;

pub(crate) use encoder_cache::EncoderCacheManager;
use freeq::BlockMeta;

/// Computes an incremental per-page prefix hash.
///
/// `parent` is the hash of the preceding page (or the chain seed for the
/// first page), so equal hashes identify equal complete prefixes up to hash
/// collisions. `group_id` and `modality_tag` separate chains that hash the
/// same tokens for different cache groups or modalities. Both algorithms
/// produce a 64-bit key, so callers must still compare exact tokens before
/// reusing a page.
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
            // The page hash is the little-endian first eight digest bytes.
            let digest: [u8; 32] = hasher.finalize().into();
            let [b0, b1, b2, b3, b4, b5, b6, b7, ..] = digest;
            u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
        }
    }
}

/// Returns the stable byte tag included in modality-specific prefix hashes.
pub(crate) fn modality_tag(modality: Modality) -> u8 {
    match modality {
        Modality::Und => 0,
        Modality::Gen => 1,
    }
}

/// Ownership state of one physical cache page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockState {
    /// Unreferenced and unpublished; page zero stays in this state.
    Free,
    /// Referenced by a table whose pages have been allocated but not yet
    /// activated by the scheduler.
    Reserved,
    /// Referenced by a table after activation, or acquired from the prefix
    /// cache.
    Active,
    /// Unreferenced but still published in the prefix index. The page sits in
    /// its group's free queue and is evicted when allocation reaches it.
    Cached,
}

/// Cache observation emitted by a block-pool call.
///
/// Events are retained in a bounded history that drops the oldest entry when
/// full, so a consumer that drains infrequently can miss events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CacheEvent {
    BlockStored { hash: u64, block: BlockId },
    BlockRemoved { hash: u64, block: BlockId },
}

/// Aggregate cache counters retained by the block pool.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockPoolStats {
    /// Allocatable pages across all groups, excluding the page-zero sentinel.
    pub(crate) total: usize,
    /// Pages in any free queue, including [`BlockState::Cached`] pages.
    pub(crate) free: usize,
    /// Successful `BlockPool::allocate` calls, not pages.
    pub(crate) allocations: u64,
    /// Pages removed from the prefix index, whether by allocation reuse,
    /// republication, or worker invalidation.
    pub(crate) evictions: u64,
    pub(crate) blocks_stored: u64,
}

/// Free-queue anchor and capacity of one cache group.
struct Group {
    fq_head: Option<u32>,
    fq_tail: Option<u32>,
    /// Queued pages, including [`BlockState::Cached`] ones.
    free_count: usize,
    /// Allocatable pages in the group, excluding the page-zero sentinel.
    total: usize,
}

/// Pool state guarded by one mutex.
///
/// `CacheBlockRef::drop` locks this state, and `std::sync::Mutex` is not
/// reentrant, so no page reference may be dropped while it is locked.
struct PoolInner {
    block_size: usize,
    num_blocks: usize,
    /// Per-page metadata indexed by page id.
    meta: Vec<BlockMeta>,
    groups: Vec<Group>,
    /// Owning group index for every page id.
    block_group: Vec<u32>,
    /// Every published physical copy of a hash. Copies differ by source
    /// worker or, after a hash collision, by tokens.
    hash_to_blocks: HashMap<u64, Vec<BlockId>>,
    events: VecDeque<CacheEvent>,
    events_cap: usize,
    stats: BlockPoolStats,
}

impl PoolInner {
    /// Appends a block to its group free queue.
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

    /// Unlinks a block index from its group free queue.
    ///
    /// The caller guarantees that `index` is queued in `group`.
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

    /// Unlinks a block from its group free queue.
    fn fq_unlink(&mut self, id: BlockId) {
        if self.meta[id.0 as usize].in_fq {
            let group = self.block_group[id.0 as usize] as usize;
            self.fq_unlink_index(group, id.0);
        }
    }

    /// Returns the total number of free blocks.
    fn total_free(&self) -> usize {
        self.groups.iter().map(|group| group.free_count).sum()
    }

    /// Appends a cache event while respecting the bounded event history.
    fn push_event(&mut self, event: CacheEvent) {
        if self.events.len() == self.events_cap {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    /// Removes only this physical copy from the prefix index, preserving active references.
    ///
    /// Does nothing for an unpublished page. Otherwise an unreferenced page
    /// reverts from `Cached` to `Free` and keeps its free-queue position,
    /// other copies of the same hash stay published, and the call counts an
    /// eviction and records a `BlockRemoved` event.
    fn remove_cached(&mut self, block: BlockId) {
        let meta = &mut self.meta[block.0 as usize];
        let Some(hash) = meta.hash.take() else {
            return;
        };
        meta.source = None;
        meta.tokens.clear();
        if meta.state == BlockState::Cached {
            meta.state = BlockState::Free;
        }
        if let Some(copies) = self.hash_to_blocks.get_mut(&hash) {
            copies.retain(|candidate| *candidate != block);
            if copies.is_empty() {
                self.hash_to_blocks.remove(&hash);
            }
        }
        self.stats.evictions += 1;
        self.push_event(CacheEvent::BlockRemoved { hash, block });
    }

    /// Releases one reference to a KV block.
    ///
    /// The final release queues the page at its group's tail. A published
    /// page stays in the prefix index as `Cached`, so a later request can
    /// reacquire it until allocation reaches it.
    fn release_ref(&mut self, id: BlockId) {
        let meta = &mut self.meta[id.0 as usize];
        debug_assert!(meta.ref_cnt > 0, "cache page reference count underflow");
        meta.ref_cnt = meta.ref_cnt.saturating_sub(1);

        // The `in_fq` check keeps a page from being queued twice.
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
///
/// Each handle accounts for one unit of the page's reference count, and
/// dropping it releases that unit. The handle holds the pool weakly, so a
/// handle that outlives its pool drops without effect. Dropping a handle
/// locks the pool: never drop one while holding the pool's lock.
pub(crate) struct CacheBlockRef {
    id: BlockId,
    pool: Weak<Mutex<PoolInner>>,
}

impl CacheBlockRef {
    /// Returns the block identifier.
    pub(crate) fn id(&self) -> BlockId {
        self.id
    }
}

impl Drop for CacheBlockRef {
    /// Returns this reference to the pool, if the pool still exists.
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            lock(&pool).release_ref(self.id);
        }
    }
}

impl fmt::Debug for CacheBlockRef {
    /// Formats the value for diagnostic output.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("CacheBlockRef")
            .field(&self.id)
            .finish()
    }
}

impl PartialEq for CacheBlockRef {
    /// Returns whether both handles identify the same allocation.
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && Weak::ptr_eq(&self.pool, &other.pool)
    }
}

impl Eq for CacheBlockRef {}

/// Invalid block-pool geometry or group configuration.
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
///
/// Clones share one pool. Each method locks the pool independently, so a
/// sequence of calls is not atomic as a whole.
#[derive(Clone)]
pub(crate) struct BlockPool {
    inner: Arc<Mutex<PoolInner>>,
}

impl BlockPool {
    /// Creates a block pool with one full-attention cache group.
    pub(crate) fn new(num_blocks: u32, block_size: usize) -> Self {
        Self::with_groups(num_blocks as usize, block_size, &[(0, num_blocks)])
    }

    /// Validates that cache groups partition the complete physical page range exactly once.
    ///
    /// Each spec is `(first_page, page_count)` for one group, in group order.
    pub(crate) fn validate_group_specs(
        num_blocks: usize,
        group_specs: &[(u32, u32)],
    ) -> Result<(), BlockPoolConfigError> {
        if num_blocks == 0
            || group_specs.is_empty()
            || group_specs.iter().any(|(_, count)| *count == 0)
        {
            return Err(BlockPoolConfigError::EmptyCapacity);
        }
        let mut covered = vec![false; num_blocks];
        for (group, (first, count)) in group_specs.iter().enumerate() {
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

    /// Creates a block pool from a validated physical cache-group partition.
    ///
    /// # Panics
    ///
    /// Panics when the block size is zero or group specifications do not partition the pool.
    pub(crate) fn with_groups(
        num_blocks: usize,
        block_size: usize,
        group_specs: &[(u32, u32)],
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
        for (group, (first, count)) in group_specs.iter().enumerate() {
            for page in *first..(*first + *count) {
                block_group[page as usize] = group as u32;
            }
            groups.push(Group {
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
            hash_to_blocks: HashMap::new(),
            events: VecDeque::new(),
            events_cap: 4096,
            stats: BlockPoolStats {
                total: num_blocks.saturating_sub(1),
                ..BlockPoolStats::default()
            },
        };

        // Page zero is the worker's padding sentinel: it is never queued, and
        // the group that contains it loses one page of capacity.
        for (group, (first, count)) in group_specs.iter().enumerate() {
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

    /// Returns the KV block size in tokens.
    pub(crate) fn block_size(&self) -> usize {
        lock(&self.inner).block_size
    }

    /// Returns the total number of KV blocks.
    pub(crate) fn num_blocks(&self) -> usize {
        lock(&self.inner).num_blocks
    }

    /// Returns the number of KV groups.
    pub(crate) fn num_groups(&self) -> usize {
        lock(&self.inner).groups.len()
    }

    /// Returns the block capacity of a KV group.
    pub(crate) fn group_capacity(&self, group: usize) -> usize {
        lock(&self.inner)
            .groups
            .get(group)
            .map_or(0, |value| value.total)
    }

    /// Returns the allocatable pages summed over every group, excluding the
    /// page-zero sentinel.
    ///
    /// With more than one group this exceeds the number of logical blocks a
    /// single request can hold, which is bounded by the smallest group.
    pub(crate) fn request_page_capacity(&self) -> usize {
        let inner = lock(&self.inner);
        inner.groups.iter().map(|group| group.total).sum()
    }

    /// Returns the number of free blocks in a KV group.
    pub(crate) fn free_blocks_in_group(&self, group: usize) -> usize {
        lock(&self.inner)
            .groups
            .get(group)
            .map_or(0, |value| value.free_count)
    }

    /// Returns the number of logical blocks that free pages can back.
    ///
    /// Every logical block takes one page in every group, so the scarcest
    /// group bounds the result. Free pages include `Cached` ones, which
    /// allocation evicts when it reaches them.
    pub(crate) fn free_request_pages(&self) -> usize {
        let inner = lock(&self.inner);
        inner
            .groups
            .iter()
            .map(|group| group.free_count)
            .min()
            .unwrap_or(0)
    }

    /// Returns the number of blocks needed to hold `tokens` tokens.
    pub(crate) fn blocks_needed(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_size())
    }

    /// Returns a snapshot of the current statistics.
    pub(crate) fn stats(&self) -> BlockPoolStats {
        lock(&self.inner).stats
    }

    /// Reserves free pages from one cache group, evicting cached contents as necessary.
    ///
    /// Takes `count` pages from the head of the group's free queue, removes
    /// any prefix identity they still carry, and returns them `Reserved` with
    /// one reference each. Returns `None`, changing nothing, when the group
    /// does not exist or has fewer than `count` free pages.
    pub(crate) fn allocate(&self, group: usize, count: usize) -> Option<Vec<CacheBlockRef>> {
        let mut inner = lock(&self.inner);
        if group >= inner.groups.len() || count > inner.groups[group].free_count {
            return None;
        }

        // Select the first `count` pages of the free queue before changing any
        // state, so a queue shorter than its count fails with the pool intact.
        let mut pages = Vec::with_capacity(count);
        let mut next = inner.groups[group].fq_head;
        while pages.len() < count {
            let index = next?;
            pages.push(BlockId(index));
            next = inner.meta[index as usize].fq_next;
        }

        // Page references are built only after selection succeeds: dropping
        // one here would re-enter this pool's lock.
        let mut refs = Vec::with_capacity(count);
        for page in pages {
            inner.fq_unlink_index(group, page.0);
            inner.remove_cached(page);
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

    /// Marks this pool's `Reserved` pages among `blocks` as `Active`.
    ///
    /// Pages in any other state, or from another pool, are left unchanged.
    pub(crate) fn activate(&self, blocks: &[CacheBlockRef]) {
        let mut inner = lock(&self.inner);
        for block in blocks {
            // Only this pool's pages are activated; the block's pool is compared
            // by address, which needs no upgrade.
            if std::ptr::eq(block.pool.as_ptr(), Arc::as_ptr(&self.inner))
                && inner.meta[block.id.0 as usize].state == BlockState::Reserved
            {
                inner.meta[block.id.0 as usize].state = BlockState::Active;
            }
        }
    }

    /// Returns the block lifecycle state.
    pub(crate) fn block_state(&self, block: BlockId) -> BlockState {
        lock(&self.inner).meta[block.0 as usize].state
    }

    /// Returns the block KV group.
    pub(crate) fn block_group(&self, block: BlockId) -> Option<usize> {
        lock(&self.inner)
            .block_group
            .get(block.0 as usize)
            .map(|group| *group as usize)
    }

    /// Publishes one physical prefix copy under its loaded Worker identity.
    ///
    /// Callers publish only pages whose contents the worker has already
    /// computed. The call does nothing when `source` already publishes these
    /// exact tokens under `hash` on some page. Otherwise any earlier identity
    /// of `block` is removed first, so a page carries at most one hash.
    pub(crate) fn cache_block(
        &self,
        block: &CacheBlockRef,
        hash: u64,
        tokens: &[u32],
        source: &Arc<WorkerEndpoint>,
    ) {
        let mut inner = lock(&self.inner);

        // At most one copy exists per hash, source, and token content.
        if inner.hash_to_blocks.get(&hash).is_some_and(|copies| {
            copies.iter().any(|candidate| {
                let meta = &inner.meta[candidate.0 as usize];
                meta.source.as_deref() == Some(source.as_ref()) && meta.tokens == tokens
            })
        }) {
            return;
        }

        inner.remove_cached(block.id);
        let meta = &mut inner.meta[block.id.0 as usize];
        meta.hash = Some(hash);
        meta.source = Some(Arc::clone(source));
        meta.tokens.clear();
        meta.tokens.extend_from_slice(tokens);
        inner.hash_to_blocks.entry(hash).or_default().push(block.id);
        inner.stats.blocks_stored += 1;
        inner.push_event(CacheEvent::BlockStored {
            hash,
            block: block.id,
        });
    }

    /// Finds a prefix copy retained by this exact loaded Worker.
    ///
    /// The lookup acquires nothing: the page can be evicted before the caller
    /// acts on it, so reuse must go through [`BlockPool::acquire_cached`],
    /// which checks the identity again.
    pub(crate) fn lookup_cached(
        &self,
        hash: u64,
        tokens: &[u32],
        source: &WorkerEndpoint,
    ) -> Option<BlockId> {
        let inner = lock(&self.inner);
        inner
            .hash_to_blocks
            .get(&hash)?
            .iter()
            .copied()
            .find(|block| {
                let meta = &inner.meta[block.0 as usize];
                meta.source.as_deref() == Some(source) && meta.tokens == tokens
            })
    }

    /// Acquires an active reference when a cached page still matches its hash and token payload.
    ///
    /// An unreferenced `Cached` page leaves its free queue and stops counting
    /// as free capacity. Returns `None` when the page is out of range, its
    /// identity no longer matches, or its reference count would overflow.
    pub(crate) fn acquire_cached(
        &self,
        block: BlockId,
        hash: u64,
        tokens: &[u32],
        source: &WorkerEndpoint,
    ) -> Option<CacheBlockRef> {
        let mut inner = lock(&self.inner);
        let meta = inner.meta.get(block.0 as usize)?;
        if meta.hash != Some(hash)
            || meta.tokens != tokens
            || meta.source.as_deref() != Some(source)
        {
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

    /// Returns the number of published prefix pages, referenced or not.
    pub(crate) fn cached_blocks(&self) -> usize {
        lock(&self.inner)
            .hash_to_blocks
            .values()
            .map(Vec::len)
            .sum()
    }

    /// Revokes cache lookup for one lost incarnation without freeing live request pages.
    ///
    /// Referenced pages keep their owners and reference counts; only their
    /// prefix identity is removed. Unreferenced pages turn from `Cached` into
    /// `Free` without changing the free capacity.
    pub(crate) fn invalidate_source(&self, source: &WorkerEndpoint) {
        let mut inner = lock(&self.inner);
        for index in 0..inner.meta.len() {
            if inner.meta[index].source.as_deref() == Some(source) {
                inner.remove_cached(BlockId(index as u32));
            }
        }
    }

    /// Drains pending cache events in publication order.
    pub(crate) fn drain_events(&self) -> Vec<CacheEvent> {
        lock(&self.inner).events.drain(..).collect()
    }
}

impl fmt::Debug for BlockPool {
    /// Formats the value for diagnostic output.
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
#[derive(Debug)]
pub(crate) struct BlockTable {
    group_id: usize,
    block_size: usize,
    blocks: Vec<CacheBlockRef>,
}

impl BlockTable {
    /// Creates an empty block table for one KV group.
    pub(crate) fn new(group_id: usize, block_size: usize) -> Self {
        assert!(block_size > 0, "KV block size must be positive");
        Self {
            group_id,
            block_size,
            blocks: Vec::new(),
        }
    }

    /// Returns the KV group identifier.
    pub(crate) fn group_id(&self) -> usize {
        self.group_id
    }

    /// Returns the number of logical blocks the table maps.
    pub(crate) fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the table capacity in tokens.
    pub(crate) fn capacity_tokens(&self) -> usize {
        self.blocks.len() * self.block_size
    }

    /// Returns the table page identifiers.
    pub(crate) fn page_ids(&self) -> Vec<BlockId> {
        self.blocks.iter().map(CacheBlockRef::id).collect()
    }

    /// Returns the block at the requested logical position.
    pub(crate) fn block(&self, index: usize) -> Option<&CacheBlockRef> {
        self.blocks.get(index)
    }

    /// Returns whether the table already references physical page `block`.
    pub(crate) fn contains(&self, block: BlockId) -> bool {
        self.blocks.iter().any(|candidate| candidate.id() == block)
    }

    /// Appends an acquired prefix page as the next logical block.
    ///
    /// Returns `false`, dropping `block` and thereby releasing its reference,
    /// when the table already references that page.
    pub(crate) fn append_cached(&mut self, block: CacheBlockRef) -> bool {
        if self.contains(block.id()) {
            return false;
        }
        self.blocks.push(block);
        true
    }

    /// Grows the table until it covers `total_tokens` tokens.
    ///
    /// Returns the page ids allocated by this call, which is empty when the
    /// table already covers the tokens. Returns `None`, leaving the table
    /// unchanged, when the pool cannot allocate the missing pages from the
    /// table's group. The table never shrinks here.
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
        Some(page_ids)
    }

    /// Marks the table's `Reserved` pages as `Active`.
    pub(crate) fn activate(&self, pool: &BlockPool) {
        pool.activate(&self.blocks);
    }
}

/// Prefix lookup result for one scheduler-owned sequence.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PrefixLookup {
    /// Chained hash of every complete prompt block, indexed by group and then
    /// by logical block. The scheduler keeps these to publish the prompt
    /// through [`KvCacheCoordinator::cache_prefix`] after prefill. Empty when
    /// prefix caching does not apply to the request.
    pub(crate) block_hashes: Vec<Vec<u64>>,
    /// Leading logical blocks acquired from the prefix cache.
    pub(crate) cached_blocks: usize,
}

/// Read-only prefix-cache probe result across every physical cache group.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PrefixHit {
    /// Leading logical blocks cached in every group.
    pub(crate) cached_blocks: usize,
    /// Per group, the hit pages that are currently unreferenced (`Cached`).
    ///
    /// Those pages count as free capacity until acquired, so admission
    /// subtracts them from the group's free pages.
    pub(crate) cached_free_blocks: Vec<usize>,
}

/// Coordinates capacity and prefix state across physical layout groups.
///
/// Callers pass one [`BlockTable`] per group, ordered by group id.
/// `ensure_capacity` and `acquire_prefix` return `None` for any other table
/// set; `cache_prefix` returns `false` for one unless it returns early
/// because caching is disabled.
#[derive(Debug)]
pub(crate) struct KvCacheCoordinator {
    prefix_enabled: bool,
    hash_algo: HashAlgo,
    /// Root of every hash chain; an isolation key is mixed into it.
    hash_seed: u64,
}

impl Default for KvCacheCoordinator {
    /// Enables prefix caching with FNV-1a hashing and a zero chain seed.
    fn default() -> Self {
        Self {
            prefix_enabled: true,
            hash_algo: HashAlgo::Fnv1a,
            hash_seed: 0,
        }
    }
}

impl KvCacheCoordinator {
    /// Enables or disables prefix-cache lookup and publication.
    pub(crate) fn set_prefix_enabled(&mut self, enabled: bool) {
        self.prefix_enabled = enabled;
    }

    /// Sets the prefix-cache hashing algorithm.
    pub(crate) fn set_hash_algo(&mut self, algorithm: HashAlgo) {
        self.hash_algo = algorithm;
    }

    /// Grows every group table atomically to the common token boundary and
    /// returns only the pages acquired by this allocation event.
    ///
    /// Each entry of the result is `(group, new_page_ids)`. Returns `None`,
    /// with every table at its original length, when the tables do not match
    /// the pool's groups or any group lacks free pages.
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

        // Check every group before allocating from any, so a capacity shortage
        // fails without touching the pool.
        let required = pool.blocks_needed(total_tokens);
        if tables.iter().enumerate().any(|(group, table)| {
            required.saturating_sub(table.len()) > pool.free_blocks_in_group(group)
        }) {
            return None;
        }

        // If a later group still fails, truncating the earlier tables drops
        // their new references and returns those pages to the pool. No pool
        // lock is held here, so the drops cannot deadlock.
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

    /// Measures the longest complete cached prefix without acquiring page references.
    ///
    /// Admission uses the result to estimate the pages a request needs before
    /// committing to it. A page found here can be evicted before
    /// [`KvCacheCoordinator::acquire_prefix`] runs, so the hit is an estimate.
    /// Returns an empty hit when prefix caching is disabled, `cache_read` is
    /// false, or the request carries images.
    pub(crate) fn probe_prefix(
        &self,
        pool: &BlockPool,
        prompt: &[u32],
        cache_read: bool,
        has_images: bool,
        isolation_key: Option<u64>,
        source: &WorkerEndpoint,
    ) -> PrefixHit {
        let groups = pool.num_groups();
        let mut hit = PrefixHit {
            cached_free_blocks: vec![0; groups],
            ..PrefixHit::default()
        };
        // Prefix hashes cover token ids only, not image content, so requests
        // with images are excluded from prefix reuse.
        if !self.prefix_enabled || !cache_read || has_images {
            return hit;
        }

        let block_size = pool.block_size();
        let limit = prefix_lookup_limit(prompt.len(), block_size);
        if limit == 0 {
            return hit;
        }
        let hashes = self.prefix_hashes(prompt, groups, block_size, isolation_key);

        // A logical block counts only when every group holds it; the first
        // miss in any group ends the prefix.
        for index in 0..limit {
            let tokens = &prompt[index * block_size..(index + 1) * block_size];
            let mut blocks = Vec::with_capacity(groups);
            for (group, group_hashes) in hashes.iter().enumerate() {
                let Some(block) = pool.lookup_cached(group_hashes[index], tokens, source) else {
                    return hit;
                };

                // The group id is part of each hash, so a page from another
                // group can match only through a hash collision.
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

    /// Acquires the longest cache prefix present across every KV group atomically per block.
    ///
    /// Reused pages are appended as leading logical blocks, so the tables are
    /// expected to be empty. A block is taken only when every group's page is
    /// found and acquired; otherwise the scan stops and any references already
    /// taken for that block are dropped, releasing them. The returned hashes
    /// are computed even when `cache_read` is false, so the prompt can still
    /// be published after prefill. Returns `None` when the tables do not match
    /// the pool's groups.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn acquire_prefix(
        &self,
        pool: &BlockPool,
        tables: &mut [BlockTable],
        prompt: &[u32],
        cache_read: bool,
        has_images: bool,
        isolation_key: Option<u64>,
        source: &WorkerEndpoint,
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

        // Empty hashes also keep image requests out of later publication; see
        // `probe_prefix` for why images are excluded.
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

            // Find every group's page before acquiring any. Rejecting a page
            // the table already holds keeps `append_cached` below from failing.
            let mut candidates = Vec::with_capacity(groups);
            for (group, table) in tables.iter().enumerate() {
                let hash = result.block_hashes[group][index];
                let Some(block) = pool.lookup_cached(hash, tokens, source) else {
                    return Some(result);
                };
                if pool.block_group(block) != Some(group) || table.contains(block) {
                    return Some(result);
                }
                candidates.push((block, hash));
            }

            // `acquire_cached` rechecks each page's identity and fails if it
            // no longer matches.
            let mut references = Vec::with_capacity(groups);
            for (block, hash) in candidates {
                let Some(reference) = pool.acquire_cached(block, hash, tokens, source) else {
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

    /// Publishes full prompt blocks from every KV group into the prefix cache.
    ///
    /// `hashes` is the `PrefixLookup::block_hashes` from
    /// [`KvCacheCoordinator::acquire_prefix`]; callers publish only after the
    /// worker has written the prompt's pages. Blocks without both a table page
    /// and complete prompt tokens are skipped. Returns `true` without
    /// publishing when prefix caching is disabled or `cache_write` is false.
    /// Otherwise returns `false` when the tables or hashes do not match the
    /// pool's groups, which includes the empty hashes of a request excluded
    /// from prefix caching. A misordered table is detected only after the
    /// groups before it have published their blocks.
    pub(crate) fn cache_prefix(
        &self,
        pool: &BlockPool,
        tables: &[BlockTable],
        prompt: &[u32],
        hashes: &[Vec<u64>],
        cache_write: bool,
        source: &Arc<WorkerEndpoint>,
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
                pool.cache_block(block, *hash, tokens, source);
            }
        }
        true
    }

    /// Computes group-specific chained hashes for each complete prompt block.
    ///
    /// A trailing partial block is not hashed, so only complete pages are ever
    /// published. An isolation key is rotated and XORed into the chain seed,
    /// so requests with different keys compute different hash chains and,
    /// barring a hash collision, reuse none of each other's pages. A key of
    /// zero leaves the seed unchanged and therefore shares pages with
    /// requests that carry no key.
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

/// Returns the maximum prefix length eligible for lookup, in blocks.
///
/// When the prompt ends exactly on a block boundary, its last block is
/// excluded so that at least one prompt token is computed and prefill still
/// produces logits for the first sampled token. Admission counts prefix-cache
/// queries with the same rule.
fn prefix_lookup_limit(prompt_tokens: usize, block_size: usize) -> usize {
    let full_blocks = prompt_tokens / block_size;
    if prompt_tokens.is_multiple_of(block_size) {
        full_blocks.saturating_sub(1)
    } else {
        full_blocks
    }
}

/// Locks the shared state and recovers it after poisoning.
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
        let hash = block_hash(0, 0, 0, &[1, 2, 3, 4], HashAlgo::Fnv1a);
        let source = Arc::new(uniserve_worker_ipc::WorkerInfo::default().endpoint);
        pool.cache_block(first.block(0).unwrap(), hash, &[1, 2, 3, 4], &source);
        let mut second = BlockTable::new(0, 4);
        let shared = pool
            .acquire_cached(BlockId(1), hash, &[1, 2, 3, 4], &source)
            .unwrap();
        assert!(second.append_cached(shared));
        assert!(pool.allocate(0, 3).is_none());
        drop(first);
        assert!(pool.allocate(0, 3).is_none());
        drop(second);
        assert_eq!(pool.allocate(0, 3).unwrap().len(), 3);
    }

    #[test]
    fn coordinator_allocates_complete_group_tables_atomically() {
        let pool = BlockPool::with_groups(7, 4, &[(0, 4), (4, 3)]);
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
        let source = Arc::new(uniserve_worker_ipc::WorkerInfo::default().endpoint);
        pool.cache_block(first.block(0).unwrap(), hash, &[1, 2, 3, 4], &source);
        let block = pool.lookup_cached(hash, &[1, 2, 3, 4], &source).unwrap();
        let reference = pool
            .acquire_cached(block, hash, &[1, 2, 3, 4], &source)
            .unwrap();
        let mut second = BlockTable::new(0, 4);
        assert!(second.append_cached(reference));
        assert_eq!(first.page_ids(), second.page_ids());
    }

    #[test]
    fn group_validation_rejects_gaps_and_overlap() {
        assert!(BlockPool::validate_group_specs(8, &[(0, 3), (3, 5)]).is_ok());
        assert!(BlockPool::validate_group_specs(8, &[(0, 4), (3, 5)]).is_err());
        assert!(BlockPool::validate_group_specs(8, &[(0, 7)]).is_err());
    }

    #[test]
    fn prefix_reuse_advances_only_at_a_boundary_cached_in_every_group() {
        let pool = BlockPool::with_groups(7, 4, &[(0, 4), (4, 3)]);
        let coordinator = KvCacheCoordinator::default();
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut source = vec![BlockTable::new(0, 4), BlockTable::new(1, 4)];
        coordinator
            .ensure_capacity(&pool, &mut source, prompt.len())
            .unwrap();
        let hashes = coordinator.prefix_hashes(&prompt, 2, 4, None);
        let endpoint = Arc::new(uniserve_worker_ipc::WorkerInfo::default().endpoint);
        assert!(coordinator.cache_prefix(&pool, &source, &prompt, &hashes, true, &endpoint));

        let hit = coordinator.probe_prefix(&pool, &prompt, true, false, None, &endpoint);
        assert_eq!(hit.cached_blocks, 1);
        assert_eq!(hit.cached_free_blocks, vec![0, 0]);

        let mut target = vec![BlockTable::new(0, 4), BlockTable::new(1, 4)];
        let lookup = coordinator
            .acquire_prefix(&pool, &mut target, &prompt, true, false, None, &endpoint)
            .unwrap();
        assert_eq!(lookup.cached_blocks, 1);
        assert_eq!(target[0].page_ids(), source[0].page_ids()[..1]);
        assert_eq!(target[1].page_ids(), source[1].page_ids()[..1]);
    }

    #[test]
    fn prefix_copies_follow_worker_incarnations_without_releasing_active_pages() {
        let pool = BlockPool::new(5, 4);
        let coordinator = KvCacheCoordinator::default();
        let prompt = [1, 2, 3, 4, 5];
        let hashes = coordinator.prefix_hashes(&prompt, 1, 4, None);
        let first = Arc::new(uniserve_worker_ipc::WorkerInfo::default().endpoint);
        let second = Arc::new(WorkerEndpoint {
            worker_id: "replica".into(),
            incarnation: "replica-loaded".into(),
            ..first.as_ref().clone()
        });
        let mut tables = [vec![BlockTable::new(0, 4)], vec![BlockTable::new(0, 4)]];
        for (source, table) in [&first, &second].into_iter().zip(&mut tables) {
            coordinator.ensure_capacity(&pool, table, 4).unwrap();
            assert!(coordinator.cache_prefix(&pool, table, &prompt, &hashes, true, source));
            assert_eq!(
                coordinator
                    .probe_prefix(&pool, &prompt, true, false, None, source)
                    .cached_blocks,
                1
            );
        }
        let replacement = WorkerEndpoint {
            incarnation: "reloaded".into(),
            ..first.as_ref().clone()
        };
        assert_eq!(
            coordinator
                .probe_prefix(&pool, &prompt, true, false, None, &replacement)
                .cached_blocks,
            0
        );

        let free_before = pool.free_blocks_in_group(0);
        pool.invalidate_source(&first);
        assert_eq!(pool.free_blocks_in_group(0), free_before);
        assert_eq!(
            coordinator
                .probe_prefix(&pool, &prompt, true, false, None, &first)
                .cached_blocks,
            0
        );
        let mut consumer = vec![BlockTable::new(0, 4)];
        let reused = coordinator
            .acquire_prefix(&pool, &mut consumer, &prompt, true, false, None, &second)
            .unwrap();
        assert_eq!(reused.cached_blocks, 1);
        assert_eq!(consumer[0].page_ids(), tables[1][0].page_ids());
        drop(tables[0].pop());
        assert_eq!(pool.free_blocks_in_group(0), free_before + 1);
    }
}
