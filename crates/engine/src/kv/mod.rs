//! Scheduler-owned paged key/value unit allocation, window release, and prefix
//! reuse.
//!
//! The scheduler chooses every physical KV unit before a call is dispatched;
//! workers read and write only the units named in a call's block tables.
//! [`BlockPool`] owns physical unit availability and prefix-cache metadata.
//! [`BlockTable`] maps one sequence's logical pages of one cache group to
//! reference-counted pages of units. Dropping the final reference returns the
//! units to the pool. [`KvCacheCoordinator`] applies allocation, window
//! release and prefix reuse across every cache group of a request at once.
//!
//! Key concepts:
//!
//! - Unit pool: the worker stores every cache group in one pool of equally
//!   sized units. A group's logical page holds `page_tokens` tokens of every
//!   layer in the group and occupies `units_per_page` units, so groups with
//!   different page shapes share one free queue and one capacity, and
//!   admission and growth compare the sum of every group's unit demand with
//!   the free units.
//! - Unit zero: the worker treats unit zero as its padding sentinel, valid
//!   memory in every plane, so the pool never allocates it and excludes it
//!   from capacity.
//! - Free queue: an intrusive doubly linked queue of unreferenced units.
//!   Allocation takes units from the head and released units join the tail,
//!   one page's units consecutively. An unreferenced unit of a published
//!   prefix page ([`UnitState::Cached`]) therefore stays reusable until
//!   allocation reaches it; allocation then evicts the whole page. A prefix
//!   hit on an unreferenced page unlinks its units from wherever they sit.
//! - Sliding windows: a table of a sliding-window group keeps pages from its
//!   `start_page` on. Pages before it are retired, and a worker never reads
//!   them. After a call completes, [`KvCacheCoordinator::release_window`]
//!   retires the pages that lie entirely before the oldest history position
//!   any later reader needs.
//! - Prefix identity: a published page is keyed by a chained 64-bit hash per
//!   group, but a match also requires equal tokens, the same group and the
//!   same loaded `WorkerEndpoint`. The token comparison guards against hash
//!   collisions, and the endpoint comparison confines reuse to the loaded
//!   worker incarnation that computed and retains the page. A reusable
//!   prefix must hold every full-attention page before it and each
//!   sliding-window group's window of pages just before it.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use uniserve_core::{HashAlgo, KvCacheGroup, Modality, UnitId};
use uniserve_worker_ipc::WorkerEndpoint;

mod encoder_cache;
mod freeq;

pub(crate) use encoder_cache::EncoderCacheManager;
use freeq::UnitMeta;

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

/// Ownership state of one physical KV unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnitState {
    /// Unreferenced and unpublished; unit zero stays in this state.
    Free,
    /// Referenced by a table whose pages have been allocated but not yet
    /// activated by the scheduler.
    Reserved,
    /// Referenced by a table after activation, or acquired from the prefix
    /// cache.
    Active,
    /// Unreferenced but still part of a published prefix page. The unit sits
    /// in the free queue, and allocation evicts its page when it reaches it.
    Cached,
}

/// Aggregate cache counters retained by the block pool.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockPoolStats {
    /// Allocatable units, excluding the unit-zero sentinel.
    pub(crate) total: usize,
    /// Units in the free queue, including [`UnitState::Cached`] ones.
    pub(crate) free: usize,
    /// Successful `BlockPool::allocate` calls, not units.
    pub(crate) allocations: u64,
    /// Pages removed from the prefix index, whether by allocation reuse,
    /// republication, or worker invalidation.
    pub(crate) evictions: u64,
    /// Pages published to the prefix index.
    pub(crate) pages_stored: u64,
}

/// Identity of one published prefix page.
struct PublishedPage {
    hash: u64,
    group: u32,
    /// Exact token content used to disambiguate equal prefix hashes.
    tokens: Vec<u32>,
    /// Loaded Worker whose rank group retains the page.
    source: Arc<WorkerEndpoint>,
    /// The page's units in page order; the first one keys the page.
    units: Box<[UnitId]>,
}

/// Pool state guarded by one mutex.
///
/// `PageRef::drop` locks this state, and `std::sync::Mutex` is not reentrant,
/// so no page reference may be dropped while it is locked.
struct PoolInner {
    /// Per-unit metadata indexed by unit id.
    meta: Vec<UnitMeta>,
    fq_head: Option<u32>,
    fq_tail: Option<u32>,
    /// Queued units, including [`UnitState::Cached`] ones.
    free_count: usize,
    /// Published pages keyed by their first unit.
    pages: HashMap<u32, PublishedPage>,
    /// First units of every published page copy of a hash. Copies differ by
    /// group, source worker or, after a hash collision, by tokens.
    hash_to_pages: HashMap<u64, Vec<u32>>,
    stats: BlockPoolStats,
}

impl PoolInner {
    /// Appends a unit to the free queue.
    fn fq_push_back(&mut self, index: u32) {
        let tail = self.fq_tail;
        let meta = &mut self.meta[index as usize];
        meta.fq_prev = tail;
        meta.fq_next = None;
        meta.in_fq = true;
        if let Some(tail) = tail {
            self.meta[tail as usize].fq_next = Some(index);
        } else {
            self.fq_head = Some(index);
        }
        self.fq_tail = Some(index);
        self.free_count += 1;
    }

    /// Unlinks a queued unit from the free queue; does nothing otherwise.
    fn fq_unlink(&mut self, index: u32) {
        if !self.meta[index as usize].in_fq {
            return;
        }
        let previous = self.meta[index as usize].fq_prev;
        let next = self.meta[index as usize].fq_next;
        if let Some(previous) = previous {
            self.meta[previous as usize].fq_next = next;
        } else {
            self.fq_head = next;
        }
        if let Some(next) = next {
            self.meta[next as usize].fq_prev = previous;
        } else {
            self.fq_tail = previous;
        }
        let meta = &mut self.meta[index as usize];
        meta.fq_prev = None;
        meta.fq_next = None;
        meta.in_fq = false;
        self.free_count = self.free_count.saturating_sub(1);
    }

    /// Removes one published page copy from the prefix index, preserving
    /// references.
    ///
    /// Does nothing when `head` keys no published page. Otherwise every unit
    /// of the page loses its identity, unreferenced units revert from
    /// `Cached` to `Free` and keep their free-queue position, other copies of
    /// the same hash stay published, and the call counts one eviction.
    fn remove_page(&mut self, head: u32) {
        let Some(page) = self.pages.remove(&head) else {
            return;
        };
        for unit in page.units.iter() {
            let meta = &mut self.meta[unit.0 as usize];
            meta.published = None;
            if meta.state == UnitState::Cached {
                meta.state = UnitState::Free;
            }
        }
        if let Some(copies) = self.hash_to_pages.get_mut(&page.hash) {
            copies.retain(|candidate| *candidate != head);
            if copies.is_empty() {
                self.hash_to_pages.remove(&page.hash);
            }
        }
        self.stats.evictions += 1;
    }

    /// Releases one reference to every unit of a page.
    ///
    /// The final release queues the units at the tail in page order. Units of
    /// a published page stay in the prefix index as `Cached`, so a later
    /// request can reacquire the page until allocation reaches it.
    fn release_page(&mut self, units: &[UnitId]) {
        for unit in units {
            let meta = &mut self.meta[unit.0 as usize];
            debug_assert!(meta.ref_cnt > 0, "KV unit reference count underflow");
            meta.ref_cnt = meta.ref_cnt.saturating_sub(1);

            // The `in_fq` check keeps a unit from being queued twice.
            if meta.ref_cnt != 0 || meta.in_fq {
                continue;
            }
            meta.state = if meta.published.is_some() {
                UnitState::Cached
            } else {
                UnitState::Free
            };
            self.fq_push_back(unit.0);
        }
        self.stats.free = self.free_count;
    }

    /// Returns the first unit of a published copy with this exact identity.
    fn find_page(
        &self,
        hash: u64,
        group: u32,
        tokens: &[u32],
        source: &WorkerEndpoint,
    ) -> Option<u32> {
        self.hash_to_pages.get(&hash)?.iter().copied().find(|head| {
            self.pages.get(head).is_some_and(|page| {
                page.group == group && page.source.as_ref() == source && page.tokens == tokens
            })
        })
    }
}

/// A counted reference to one logical page: `units_per_page` physical units
/// of one cache group, in page order.
///
/// Each handle accounts for one reference to every unit of the page, and
/// dropping it releases them. The handle holds the pool weakly, so a handle
/// that outlives its pool drops without effect. Dropping a handle locks the
/// pool: never drop one while holding the pool's lock.
pub(crate) struct PageRef {
    units: Box<[UnitId]>,
    pool: Weak<Mutex<PoolInner>>,
}

impl PageRef {
    /// Returns the page's units in page order.
    pub(crate) fn units(&self) -> &[UnitId] {
        &self.units
    }
}

impl Drop for PageRef {
    /// Returns this reference to the pool, if the pool still exists.
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            lock(&pool).release_page(&self.units);
        }
    }
}

impl fmt::Debug for PageRef {
    /// Formats the value for diagnostic output.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("PageRef").field(&self.units).finish()
    }
}

/// Physical KV units, free capacity, unit references, and prefix-cache
/// metadata.
///
/// Clones share one pool. Each method locks the pool independently, so a
/// sequence of calls is not atomic as a whole.
#[derive(Clone)]
pub(crate) struct BlockPool {
    inner: Arc<Mutex<PoolInner>>,
}

impl BlockPool {
    /// Creates a pool of `num_units` units, including the unit-zero sentinel.
    ///
    /// # Panics
    ///
    /// Panics when the pool holds no allocatable unit.
    pub(crate) fn new(num_units: u32) -> Self {
        assert!(num_units > 1, "a KV unit pool requires an allocatable unit");
        let mut inner = PoolInner {
            meta: (0..num_units).map(|_| UnitMeta::new()).collect(),
            fq_head: None,
            fq_tail: None,
            free_count: 0,
            pages: HashMap::new(),
            hash_to_pages: HashMap::new(),
            stats: BlockPoolStats {
                total: num_units as usize - 1,
                ..BlockPoolStats::default()
            },
        };

        // Unit zero is the worker's padding sentinel and is never queued.
        for unit in 1..num_units {
            inner.fq_push_back(unit);
        }
        inner.stats.free = inner.free_count;
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    /// Returns the allocatable units, excluding the unit-zero sentinel.
    pub(crate) fn usable_units(&self) -> usize {
        lock(&self.inner).stats.total
    }

    /// Returns the queued units, including `Cached` ones, which allocation
    /// evicts when it reaches them.
    pub(crate) fn free_units(&self) -> usize {
        lock(&self.inner).free_count
    }

    /// Returns a snapshot of the current statistics.
    pub(crate) fn stats(&self) -> BlockPoolStats {
        lock(&self.inner).stats
    }

    /// Reserves `count` pages of `units_per_page` units each, evicting cached
    /// contents as necessary.
    ///
    /// Takes the units from the head of the free queue in order. A unit that
    /// still belongs to a published page evicts that page. Returns the pages
    /// `Reserved` with one reference each, or `None`, changing nothing, when
    /// fewer units are free than the pages need.
    pub(crate) fn allocate(&self, count: usize, units_per_page: usize) -> Option<Vec<PageRef>> {
        let mut inner = lock(&self.inner);
        let needed = count.checked_mul(units_per_page)?;
        if needed > inner.free_count || units_per_page == 0 {
            return None;
        }

        // Page references are built only after selection succeeds: dropping
        // one here would re-enter this pool's lock.
        let mut pages = Vec::with_capacity(count);
        for _ in 0..count {
            let mut units = Vec::with_capacity(units_per_page);
            for _ in 0..units_per_page {
                // The free count covers every unit taken here.
                let Some(index) = inner.fq_head else {
                    break;
                };
                inner.fq_unlink(index);
                if let Some(head) = inner.meta[index as usize].published {
                    inner.remove_page(head);
                }
                let meta = &mut inner.meta[index as usize];
                meta.state = UnitState::Reserved;
                meta.ref_cnt = 1;
                units.push(UnitId(index));
            }
            pages.push(PageRef {
                units: units.into_boxed_slice(),
                pool: Arc::downgrade(&self.inner),
            });
        }
        inner.stats.allocations += 1;
        inner.stats.free = inner.free_count;
        Some(pages)
    }

    /// Marks this pool's `Reserved` units of `pages` as `Active`.
    ///
    /// Units in any other state, or pages from another pool, are unchanged.
    pub(crate) fn activate<'a>(&self, pages: impl IntoIterator<Item = &'a PageRef>) {
        let mut inner = lock(&self.inner);
        for page in pages {
            // Only this pool's units are activated; the page's pool is
            // compared by address, which needs no upgrade.
            if !std::ptr::eq(page.pool.as_ptr(), Arc::as_ptr(&self.inner)) {
                continue;
            }
            for unit in page.units.iter() {
                let meta = &mut inner.meta[unit.0 as usize];
                if meta.state == UnitState::Reserved {
                    meta.state = UnitState::Active;
                }
            }
        }
    }

    /// Returns a unit's lifecycle state.
    #[cfg(test)]
    fn unit_state(&self, unit: UnitId) -> UnitState {
        lock(&self.inner).meta[unit.0 as usize].state
    }

    /// Publishes one physical prefix page of `group` under its loaded Worker
    /// identity.
    ///
    /// Callers publish only pages whose contents the worker has already
    /// computed. The call does nothing when `source` already publishes these
    /// exact tokens under `hash` in `group` on some page. Otherwise any
    /// earlier identity of the page is removed first, so a page carries at
    /// most one hash.
    pub(crate) fn publish(
        &self,
        page: &PageRef,
        group: u32,
        hash: u64,
        tokens: &[u32],
        source: &Arc<WorkerEndpoint>,
    ) {
        let mut inner = lock(&self.inner);
        if inner.find_page(hash, group, tokens, source).is_some() {
            return;
        }

        // A unit belongs to at most one page, so the page's first unit keys
        // its identity; any earlier identity of these units is replaced.
        for unit in page.units.iter() {
            if let Some(head) = inner.meta[unit.0 as usize].published {
                inner.remove_page(head);
            }
        }
        let head = page.units[0].0;
        for unit in page.units.iter() {
            inner.meta[unit.0 as usize].published = Some(head);
        }
        inner.pages.insert(
            head,
            PublishedPage {
                hash,
                group,
                tokens: tokens.to_vec(),
                source: Arc::clone(source),
                units: page.units.clone(),
            },
        );
        inner.hash_to_pages.entry(hash).or_default().push(head);
        inner.stats.pages_stored += 1;
    }

    /// Finds a prefix page of `group` retained by this exact loaded Worker.
    ///
    /// Returns the page's first unit and whether the page is currently
    /// unreferenced, in which case its units count as free capacity until
    /// acquired. The lookup acquires nothing: the page can be evicted before
    /// the caller acts on it, so reuse must go through [`BlockPool::acquire`],
    /// which checks the identity again.
    pub(crate) fn lookup(
        &self,
        hash: u64,
        group: u32,
        tokens: &[u32],
        source: &WorkerEndpoint,
    ) -> Option<(UnitId, bool)> {
        let inner = lock(&self.inner);
        let head = inner.find_page(hash, group, tokens, source)?;
        Some((
            UnitId(head),
            inner.meta[head as usize].state == UnitState::Cached,
        ))
    }

    /// Acquires an active reference to a published page that still matches
    /// its identity.
    ///
    /// Unreferenced units of the page leave the free queue and stop counting
    /// as free capacity. Returns `None` when `head` keys no page with this
    /// identity or a reference count would overflow.
    pub(crate) fn acquire(
        &self,
        head: UnitId,
        hash: u64,
        group: u32,
        tokens: &[u32],
        source: &WorkerEndpoint,
    ) -> Option<PageRef> {
        let mut inner = lock(&self.inner);
        let page = inner.pages.get(&head.0)?;
        if page.hash != hash
            || page.group != group
            || page.tokens != tokens
            || page.source.as_ref() != source
        {
            return None;
        }
        let units = page.units.clone();
        if units
            .iter()
            .any(|unit| inner.meta[unit.0 as usize].ref_cnt == u32::MAX)
        {
            return None;
        }
        for unit in units.iter() {
            inner.fq_unlink(unit.0);
            let meta = &mut inner.meta[unit.0 as usize];
            meta.ref_cnt += 1;
            meta.state = UnitState::Active;
        }
        inner.stats.free = inner.free_count;
        Some(PageRef {
            units,
            pool: Arc::downgrade(&self.inner),
        })
    }

    /// Returns the number of published prefix pages, referenced or not.
    pub(crate) fn cached_pages(&self) -> usize {
        lock(&self.inner).pages.len()
    }

    /// Revokes cache lookup for one lost incarnation without freeing live
    /// request units.
    ///
    /// Referenced units keep their owners and reference counts; only their
    /// prefix identity is removed. Unreferenced units turn from `Cached` into
    /// `Free` without changing the free capacity.
    pub(crate) fn invalidate_source(&self, source: &WorkerEndpoint) {
        let mut inner = lock(&self.inner);
        let heads = inner
            .pages
            .iter()
            .filter(|(_, page)| page.source.as_ref() == source)
            .map(|(head, _)| *head)
            .collect::<Vec<_>>();
        for head in heads {
            inner.remove_page(head);
        }
    }
}

impl fmt::Debug for BlockPool {
    /// Formats the value for diagnostic output.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.stats();
        formatter
            .debug_struct("BlockPool")
            .field("usable_units", &stats.total)
            .field("free_units", &stats.free)
            .finish()
    }
}

/// The page shape and history policy of one cache group, in scheduler terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GroupShape {
    /// Tokens per logical page.
    pub(crate) page_tokens: usize,
    /// Units one logical page occupies.
    pub(crate) units_per_page: usize,
    /// History tokens any reader needs, or `None` for full attention.
    pub(crate) window: Option<usize>,
}

impl GroupShape {
    /// Derives the scheduler shape of a worker-reported group.
    pub(crate) fn from_group(group: &KvCacheGroup) -> Self {
        Self {
            page_tokens: group.page_tokens as usize,
            units_per_page: group.units_per_page as usize,
            window: group.kind.window().map(|window| window as usize),
        }
    }

    /// Returns the first logical page a read starting at token `read_start`
    /// needs: page zero for full attention, otherwise the page holding the
    /// oldest history token of the window.
    fn first_needed_page(&self, read_start: usize) -> usize {
        self.window.map_or(0, |window| {
            read_start.saturating_sub(window) / self.page_tokens
        })
    }

    /// Returns the logical pages that cover `tokens` tokens.
    fn pages_covering(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.page_tokens)
    }
}

/// One logical page held by a table, with the allocation serial of its first
/// unit when the table allocated it fresh.
#[derive(Debug)]
struct TablePage {
    page: PageRef,
    /// `None` for a page acquired from the prefix cache, which already holds
    /// computed KV; otherwise the owning allocation's unit serial at the
    /// page's allocation.
    serial: Option<u64>,
}

/// One sequence's logical-page to physical-unit mapping in one cache group.
///
/// Logical pages `start_page..end_page` are held; earlier pages are retired
/// tombstones that no worker reads, which only a sliding-window group
/// produces.
#[derive(Debug)]
pub(crate) struct BlockTable {
    group_id: usize,
    shape: GroupShape,
    start_page: usize,
    pages: VecDeque<TablePage>,
    /// `(start_page, end_page)` most recently declared to workers, or `None`
    /// before the first declaration or after a rollback.
    sent: Option<(usize, usize)>,
}

impl BlockTable {
    /// Creates an empty table for one cache group.
    pub(crate) fn new(group_id: usize, shape: GroupShape) -> Self {
        assert!(
            shape.page_tokens > 0 && shape.units_per_page > 0,
            "KV page shape must be positive"
        );
        Self {
            group_id,
            shape,
            start_page: 0,
            pages: VecDeque::new(),
            sent: None,
        }
    }

    /// Returns the cache group identifier.
    pub(crate) fn group_id(&self) -> usize {
        self.group_id
    }

    /// Returns the first logical page still held.
    pub(crate) fn start_page(&self) -> usize {
        self.start_page
    }

    /// Returns the logical page after the last held one.
    pub(crate) fn end_page(&self) -> usize {
        self.start_page + self.pages.len()
    }

    /// Returns the absolute token extent the table covers.
    pub(crate) fn capacity_tokens(&self) -> usize {
        self.end_page() * self.shape.page_tokens
    }

    /// Returns the held pages' units, page-major from `start_page`.
    pub(crate) fn unit_ids(&self) -> Vec<UnitId> {
        self.pages
            .iter()
            .flat_map(|entry| entry.page.units().iter().copied())
            .collect()
    }

    /// Returns the held page at absolute logical `index`, if held.
    pub(crate) fn page(&self, index: usize) -> Option<&PageRef> {
        index
            .checked_sub(self.start_page)
            .and_then(|offset| self.pages.get(offset))
            .map(|entry| &entry.page)
    }

    /// Returns the units of fresh pages allocated at or after `serial`, which
    /// a worker must reset before any call uses them.
    pub(crate) fn fresh_units(&self, serial: u64) -> Vec<UnitId> {
        self.pages
            .iter()
            .filter(|entry| entry.serial.is_some_and(|value| value >= serial))
            .flat_map(|entry| entry.page.units().iter().copied())
            .collect()
    }

    /// Whether the held page interval differs from the one last declared.
    pub(crate) fn changed_since_sent(&self) -> bool {
        self.sent != Some((self.start_page, self.end_page()))
    }

    /// Marks the table's `Reserved` units as `Active`.
    pub(crate) fn activate(&self, pool: &BlockPool) {
        pool.activate(self.pages.iter().map(|entry| &entry.page));
    }

    /// Records the held page interval as declared to workers.
    pub(crate) fn mark_sent(&mut self) {
        self.sent = Some((self.start_page, self.end_page()));
    }

    /// Forgets the last declaration, so the next dispatch resends the table.
    pub(crate) fn clear_sent(&mut self) {
        self.sent = None;
    }

    /// Returns the pages a call reading from token `read_start` and holding
    /// `total_tokens` tokens still needs, as `(new start page, pages to
    /// append)`.
    ///
    /// Pages are only appended, so a non-empty table keeps its start. An
    /// empty table that never held a page starts at the first page the read
    /// needs; a full-attention table always starts at page zero.
    fn demand(&self, read_start: usize, total_tokens: usize) -> (usize, usize) {
        let start = if self.pages.is_empty() && self.start_page == 0 {
            self.shape.first_needed_page(read_start)
        } else {
            self.start_page
        };
        let end = self.shape.pages_covering(total_tokens);
        let current_end = if self.pages.is_empty() && self.start_page == 0 {
            start
        } else {
            self.end_page()
        };
        (start, end.saturating_sub(current_end))
    }

    /// Retires held pages before absolute logical page `page`.
    ///
    /// Dropping a page releases its units: a published page's units stay
    /// matchable as `Cached`, others become `Free`.
    fn retire_before(&mut self, page: usize) {
        while self.start_page < page {
            if self.pages.pop_front().is_none() {
                // An empty table moves its origin without holding pages.
                self.start_page = page;
                return;
            }
            self.start_page += 1;
        }
    }
}

/// One sequence's page tables, one per cache group in group order.
///
/// Every page is a counted reference, so dropping the allocation releases the
/// sequence's references, and a page shared through the prefix cache stays
/// resident while another table still holds it.
#[derive(Debug)]
pub(crate) struct KvAllocation {
    pub(crate) tables: Vec<BlockTable>,
    /// Units allocated fresh to the tables so far. Each fresh page records
    /// the serial of its first unit, so a dispatch can name the units
    /// allocated after a watermark.
    allocated_units: u64,
}

impl KvAllocation {
    /// Returns the serial of the next freshly allocated unit.
    pub(crate) fn allocated_units(&self) -> u64 {
        self.allocated_units
    }

    /// Forces every table to be declared again at the next dispatch.
    pub(crate) fn clear_sent(&mut self) {
        for table in &mut self.tables {
            table.clear_sent();
        }
    }
}

/// Prefix lookup result for one scheduler-owned sequence.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PrefixLookup {
    /// Chained hash of every complete prompt page, indexed by group and then
    /// by logical page. The scheduler keeps these to publish the prompt
    /// through [`KvCacheCoordinator::publish_prompt`] as prefill completes.
    /// Empty when prefix caching does not apply to the request.
    pub(crate) page_hashes: Vec<Vec<u64>>,
    /// Leading prompt tokens acquired from the prefix cache; a multiple of
    /// the largest page size.
    pub(crate) cached_tokens: usize,
}

/// Read-only prefix-cache probe result across every cache group.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PrefixHit {
    /// Leading prompt tokens every group can reuse; a multiple of the
    /// largest page size.
    pub(crate) cached_tokens: usize,
    /// Units of the pages a hit would acquire that are currently
    /// unreferenced (`Cached`).
    ///
    /// Those units count as free capacity until acquired, so admission
    /// subtracts them from the free units.
    pub(crate) cached_free_units: usize,
}

/// The pages a validated prefix of `tokens` tokens needs from each group.
#[derive(Debug, Default)]
struct PrefixSelection {
    tokens: usize,
    /// Per group, the group's first needed page and, for it and each later
    /// needed page in logical order, the page's first unit and whether the
    /// page is currently unreferenced.
    pages: Vec<(usize, Vec<(UnitId, bool)>)>,
}

/// Coordinates capacity, window release and prefix state across cache
/// groups.
///
/// Callers pass one [`KvAllocation`] whose tables match `groups` in order;
/// the methods return `None` or `false` for any other table set.
#[derive(Debug)]
pub(crate) struct KvCacheCoordinator {
    groups: Vec<GroupShape>,
    prefix_enabled: bool,
    hash_algo: HashAlgo,
    /// Root of every hash chain; an isolation key is mixed into it.
    hash_seed: u64,
}

impl KvCacheCoordinator {
    /// Creates a coordinator for the given cache groups with prefix caching
    /// enabled, FNV-1a hashing and a zero chain seed.
    pub(crate) fn new(groups: Vec<GroupShape>) -> Self {
        Self {
            groups,
            prefix_enabled: true,
            hash_algo: HashAlgo::Fnv1a,
            hash_seed: 0,
        }
    }

    /// Returns the cache groups in table order.
    pub(crate) fn groups(&self) -> &[GroupShape] {
        &self.groups
    }

    /// Enables or disables prefix-cache lookup and publication.
    pub(crate) fn set_prefix_enabled(&mut self, enabled: bool) {
        self.prefix_enabled = enabled;
    }

    /// Sets the prefix-cache hashing algorithm.
    pub(crate) fn set_hash_algo(&mut self, algorithm: HashAlgo) {
        self.hash_algo = algorithm;
    }

    /// Returns the largest page size of any group, in tokens.
    fn max_page_tokens(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.page_tokens)
            .max()
            .unwrap_or(1)
    }

    /// Creates empty tables for every group.
    pub(crate) fn empty(&self) -> KvAllocation {
        KvAllocation {
            tables: self
                .groups
                .iter()
                .enumerate()
                .map(|(group, shape)| BlockTable::new(group, *shape))
                .collect(),
            allocated_units: 0,
        }
    }

    /// Whether `allocation` holds one table per group in group order.
    fn matches(&self, allocation: &KvAllocation) -> bool {
        allocation.tables.len() == self.groups.len()
            && allocation
                .tables
                .iter()
                .enumerate()
                .all(|(group, table)| table.group_id == group && table.shape == self.groups[group])
    }

    /// Returns the units that growing `allocation` for a call reading from
    /// token `read_start` and holding `total_tokens` tokens would allocate.
    ///
    /// A full-attention group needs every page before `total_tokens`; a
    /// sliding-window group needs only the pages intersecting
    /// `[read_start - window, total_tokens)`. Returns `None` when the tables
    /// do not match the groups.
    pub(crate) fn units_needed(
        &self,
        allocation: &KvAllocation,
        read_start: usize,
        total_tokens: usize,
    ) -> Option<usize> {
        if !self.matches(allocation) {
            return None;
        }
        Some(
            allocation
                .tables
                .iter()
                .map(|table| {
                    let (_, pages) = table.demand(read_start, total_tokens);
                    pages * table.shape.units_per_page
                })
                .sum(),
        )
    }

    /// Returns the units a request that will reuse a `prefix_tokens`-long
    /// cached prefix allocates to hold `total_tokens` tokens.
    ///
    /// The prefix is a multiple of every page size, and acquisition takes its
    /// pages (a full group's whole prefix, a sliding group's window) from the
    /// cache, so each group allocates only the pages after it.
    pub(crate) fn units_after_prefix(&self, prefix_tokens: usize, total_tokens: usize) -> usize {
        self.groups
            .iter()
            .map(|group| {
                let pages = group
                    .pages_covering(total_tokens)
                    .saturating_sub(prefix_tokens / group.page_tokens);
                pages * group.units_per_page
            })
            .sum()
    }

    /// Returns the units that tables holding every page of `tokens` tokens
    /// in every group occupy, as a worst-case reservation made up front.
    pub(crate) fn units_for_tokens(&self, tokens: usize) -> usize {
        self.groups
            .iter()
            .map(|group| group.pages_covering(tokens) * group.units_per_page)
            .sum()
    }

    /// Returns the fewest units that must coexist to compute a `tokens`-long
    /// prompt in chunks of at most `chunk` tokens.
    ///
    /// Full-attention groups hold every page; a sliding-window group holds
    /// only the pages one chunk's window and query can intersect.
    pub(crate) fn min_units_for_prompt(&self, tokens: usize, chunk: usize) -> usize {
        self.groups
            .iter()
            .map(|group| {
                let pages = group.pages_covering(tokens);
                let pages = group.window.map_or(pages, |window| {
                    pages.min((window + chunk.max(1)).div_ceil(group.page_tokens) + 1)
                });
                pages * group.units_per_page
            })
            .sum()
    }

    /// Grows every group table atomically for a call that reads from token
    /// `read_start` and holds `total_tokens` tokens.
    ///
    /// The sum of every group's unit demand must fit the free units; the call
    /// then allocates them all or nothing. Returns `None`, with every table
    /// unchanged, when the tables do not match the groups or the pool lacks
    /// free units.
    pub(crate) fn ensure_capacity(
        &self,
        pool: &BlockPool,
        allocation: &mut KvAllocation,
        read_start: usize,
        total_tokens: usize,
    ) -> Option<()> {
        let demand = self.units_needed(allocation, read_start, total_tokens)?;
        if demand == 0 {
            return Some(());
        }
        if demand > pool.free_units() {
            return None;
        }

        // Allocate every group's pages before appending any, so a shortage
        // (which the free-unit check above excludes) cannot leave a partial
        // growth. Dropping unappended pages releases them without holding
        // the pool's lock.
        let mut grown = Vec::with_capacity(allocation.tables.len());
        for table in &allocation.tables {
            let (start, pages) = table.demand(read_start, total_tokens);
            let allocated = if pages == 0 {
                Vec::new()
            } else {
                pool.allocate(pages, table.shape.units_per_page)?
            };
            grown.push((start, allocated));
        }
        for (table, (start, pages)) in allocation.tables.iter_mut().zip(grown) {
            if table.pages.is_empty() && table.start_page == 0 {
                table.start_page = start;
            }
            for page in pages {
                let serial = allocation.allocated_units;
                allocation.allocated_units += page.units().len() as u64;
                table.pages.push_back(TablePage {
                    page,
                    serial: Some(serial),
                });
            }
        }
        Some(())
    }

    /// Retires sliding-window pages no later reader needs.
    ///
    /// `visible` is the request's accepted KV extent after a completed call.
    /// Calls apply in submission order, so every earlier reader has
    /// completed, and every later call reads from at least `visible -
    /// window`. Pages that lie entirely before that bound are retired; a
    /// published page stays matchable as `Cached`. Full-attention tables are
    /// unchanged. Returns `false` when the tables do not match the groups.
    pub(crate) fn release_window(&self, allocation: &mut KvAllocation, visible: usize) -> bool {
        if !self.matches(allocation) {
            return false;
        }
        for table in &mut allocation.tables {
            if let Some(window) = table.shape.window {
                let bound = visible.saturating_sub(window) / table.shape.page_tokens;
                table.retire_before(bound);
            }
        }
        true
    }

    /// Computes group-specific chained hashes for each complete prompt page.
    ///
    /// A trailing partial page is not hashed, so only complete pages are ever
    /// published. An isolation key is rotated and XORed into the chain seed,
    /// so requests with different keys compute different hash chains and,
    /// barring a hash collision, reuse none of each other's pages. A key of
    /// zero leaves the seed unchanged and therefore shares pages with
    /// requests that carry no key.
    fn prefix_hashes(&self, prompt: &[u32], isolation_key: Option<u64>) -> Vec<Vec<u64>> {
        let seed = isolation_key.map_or(self.hash_seed, |key| self.hash_seed ^ key.rotate_left(17));
        self.groups
            .iter()
            .enumerate()
            .map(|(group, shape)| {
                let mut parent = seed;
                prompt
                    .chunks_exact(shape.page_tokens)
                    .map(|tokens| {
                        parent = block_hash(
                            parent,
                            group as u32,
                            modality_tag(Modality::Und),
                            tokens,
                            self.hash_algo,
                        );
                        parent
                    })
                    .collect()
            })
            .collect()
    }

    /// Finds the longest prompt prefix every group can reuse.
    ///
    /// The prefix length `M` is a multiple of the largest page size and
    /// leaves at least one prompt token to compute, so prefill still produces
    /// the first sampled token's logits. A full-attention group needs every
    /// page before `M`; a sliding-window group needs only the pages
    /// intersecting `[M - window, M)`. Starting from the longest eligible
    /// length, each group lowers `M` to the longest length it supports, and
    /// the groups repeat until no group lowers it further, as the SGLang
    /// `SWAComponent` validator does. A full group's constraint is a leading
    /// run of hits, so any shorter length stays valid for it; a sliding
    /// group's window moves with `M`, so it is re-checked after every change.
    fn select_prefix(
        &self,
        pool: &BlockPool,
        prompt: &[u32],
        hashes: &[Vec<u64>],
        source: &WorkerEndpoint,
    ) -> PrefixSelection {
        let step = self.max_page_tokens();
        let mut tokens = prompt.len().saturating_sub(1) / step * step;

        // One lookup per complete prompt page and group; each hit records
        // the page's first unit and whether it is unreferenced.
        let hits = self
            .groups
            .iter()
            .enumerate()
            .map(|(group, shape)| {
                let pages = (tokens / shape.page_tokens).min(hashes[group].len());
                (0..pages)
                    .map(|page| {
                        let span =
                            &prompt[page * shape.page_tokens..(page + 1) * shape.page_tokens];
                        pool.lookup(hashes[group][page], group as u32, span, source)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Misses before each page, so a window check is one subtraction.
        let misses = hits
            .iter()
            .map(|pages| {
                let mut total = 0;
                let mut prefix = Vec::with_capacity(pages.len() + 1);
                prefix.push(0);
                for page in pages {
                    total += usize::from(page.is_none());
                    prefix.push(total);
                }
                prefix
            })
            .collect::<Vec<_>>();

        loop {
            let previous = tokens;
            for (group, shape) in self.groups.iter().enumerate() {
                let supported = |length: usize| {
                    let end = length / shape.page_tokens;
                    let start = shape.first_needed_page(length).min(end);
                    misses[group][end] == misses[group][start]
                };
                while tokens > 0 && !supported(tokens) {
                    tokens -= step;
                }
            }
            if tokens == previous {
                break;
            }
        }

        // Validation leaves no miss in any selected range; a range that still
        // holds one selects no prefix rather than a table with a hole.
        let pages = self
            .groups
            .iter()
            .enumerate()
            .map(|(group, shape)| {
                let end = tokens / shape.page_tokens;
                let start = shape.first_needed_page(tokens).min(end);
                let selected = hits[group][start..end]
                    .iter()
                    .copied()
                    .collect::<Option<Vec<_>>>()?;
                Some((start, selected))
            })
            .collect::<Option<Vec<_>>>();
        pages.map_or_else(PrefixSelection::default, |pages| PrefixSelection {
            tokens,
            pages,
        })
    }

    /// Measures the longest reusable prompt prefix without acquiring page
    /// references.
    ///
    /// Admission uses the result to estimate the units a request needs
    /// before committing to it. A page found here can be evicted before
    /// [`KvCacheCoordinator::acquire_prefix`] runs, so the hit is an
    /// estimate. Returns an empty hit when prefix caching is disabled,
    /// `cache_read` is false, or the request carries images.
    pub(crate) fn probe_prefix(
        &self,
        pool: &BlockPool,
        prompt: &[u32],
        cache_read: bool,
        has_images: bool,
        isolation_key: Option<u64>,
        source: &WorkerEndpoint,
    ) -> PrefixHit {
        // Prefix hashes cover token ids only, not image content, so requests
        // with images are excluded from prefix reuse.
        if !self.prefix_enabled || !cache_read || has_images {
            return PrefixHit::default();
        }
        let hashes = self.prefix_hashes(prompt, isolation_key);
        let selection = self.select_prefix(pool, prompt, &hashes, source);
        let cached_free_units = selection
            .pages
            .iter()
            .zip(&self.groups)
            .map(|((_, pages), shape)| {
                pages.iter().filter(|(_, cached)| *cached).count() * shape.units_per_page
            })
            .sum();
        PrefixHit {
            cached_tokens: selection.tokens,
            cached_free_units,
        }
    }

    /// Acquires the longest reusable prompt prefix into empty tables.
    ///
    /// Full-attention tables take every page before the prefix length;
    /// sliding-window tables take only their window's pages and start at the
    /// first of them. Acquisition is all or nothing: when a page no longer
    /// matches its identity, every reference taken is dropped and no prefix
    /// is reused. The returned hashes are computed even when `cache_read` is
    /// false, so the prompt can still be published as prefill completes.
    /// Returns `None` when the tables do not match the groups or are not
    /// empty.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn acquire_prefix(
        &self,
        pool: &BlockPool,
        allocation: &mut KvAllocation,
        prompt: &[u32],
        cache_read: bool,
        has_images: bool,
        isolation_key: Option<u64>,
        source: &WorkerEndpoint,
    ) -> Option<PrefixLookup> {
        if !self.matches(allocation)
            || allocation
                .tables
                .iter()
                .any(|table| !table.pages.is_empty() || table.start_page != 0)
        {
            return None;
        }

        // Empty hashes also keep image requests out of later publication; see
        // `probe_prefix` for why images are excluded.
        if !self.prefix_enabled || has_images {
            return Some(PrefixLookup::default());
        }
        let hashes = self.prefix_hashes(prompt, isolation_key);
        let mut result = PrefixLookup {
            page_hashes: hashes,
            cached_tokens: 0,
        };
        if !cache_read {
            return Some(result);
        }

        let selection = self.select_prefix(pool, prompt, &result.page_hashes, source);
        let acquired = selection
            .pages
            .iter()
            .enumerate()
            .map(|(group, (start, pages))| {
                let shape = self.groups[group];
                let references = pages
                    .iter()
                    .enumerate()
                    .map(|(offset, (head, _))| {
                        let page = start + offset;
                        let span =
                            &prompt[page * shape.page_tokens..(page + 1) * shape.page_tokens];
                        // `acquire` rechecks each page's identity.
                        pool.acquire(
                            *head,
                            result.page_hashes[group][page],
                            group as u32,
                            span,
                            source,
                        )
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some((*start, references))
            })
            .collect::<Option<Vec<_>>>();
        // A page that changed identity since selection leaves the request
        // without a reused prefix; the references taken so far drop here,
        // without the pool's lock held.
        let Some(acquired) = acquired else {
            return Some(result);
        };

        for (table, (start, references)) in allocation.tables.iter_mut().zip(acquired) {
            table.start_page = start;
            table.pages.extend(
                references
                    .into_iter()
                    .map(|page| TablePage { page, serial: None }),
            );
        }
        result.cached_tokens = selection.tokens;
        Some(result)
    }

    /// Publishes complete computed prompt pages of every group into the
    /// prefix cache.
    ///
    /// `hashes` is the `PrefixLookup::page_hashes` from
    /// [`KvCacheCoordinator::acquire_prefix`], and `published` records per
    /// group how many leading pages were already published; both advance
    /// together. A page is published once the worker has computed all of its
    /// prompt tokens (`computed_tokens`) and while the table still holds it,
    /// so a caller that publishes after every prefill chunk, before releasing
    /// any window page, publishes every page of a sliding-window group before
    /// it leaves the window. Returns `true` without publishing when prefix
    /// caching is disabled, `cache_write` is false, or `hashes` is empty (a
    /// request excluded from prefix caching), and `false` when the tables or
    /// hashes do not match the groups.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn publish_prompt(
        &self,
        pool: &BlockPool,
        allocation: &KvAllocation,
        prompt: &[u32],
        hashes: &[Vec<u64>],
        computed_tokens: usize,
        published: &mut Vec<usize>,
        cache_write: bool,
        source: &Arc<WorkerEndpoint>,
    ) -> bool {
        if !self.prefix_enabled || !cache_write || hashes.is_empty() {
            return true;
        }
        if !self.matches(allocation) || hashes.len() != self.groups.len() {
            return false;
        }
        published.resize(self.groups.len(), 0);

        let computed = computed_tokens.min(prompt.len());
        for (group, (table, group_hashes)) in allocation.tables.iter().zip(hashes).enumerate() {
            let page_tokens = table.shape.page_tokens;
            let complete = (computed / page_tokens).min(group_hashes.len());
            for page in published[group]..complete {
                // A page retired before publication (never, for a caller that
                // publishes before releasing) cannot be published anymore.
                let Some(reference) = table.page(page) else {
                    continue;
                };
                let tokens = &prompt[page * page_tokens..(page + 1) * page_tokens];
                pool.publish(reference, group as u32, group_hashes[page], tokens, source);
            }
            published[group] = published[group].max(complete);
        }
        true
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

    fn endpoint() -> Arc<WorkerEndpoint> {
        Arc::new(uniserve_worker_ipc::WorkerInfo::default().endpoint)
    }

    fn full(page_tokens: usize, units_per_page: usize) -> GroupShape {
        GroupShape {
            page_tokens,
            units_per_page,
            window: None,
        }
    }

    fn sliding(page_tokens: usize, units_per_page: usize, window: usize) -> GroupShape {
        GroupShape {
            page_tokens,
            units_per_page,
            window: Some(window),
        }
    }

    /// Prefills `prompt` into fresh tables and publishes every complete page.
    fn publish(
        coordinator: &KvCacheCoordinator,
        pool: &BlockPool,
        prompt: &[u32],
        source: &Arc<WorkerEndpoint>,
    ) -> KvAllocation {
        let mut allocation = coordinator.empty();
        let lookup = coordinator
            .acquire_prefix(pool, &mut allocation, prompt, false, false, None, source)
            .unwrap();
        coordinator
            .ensure_capacity(pool, &mut allocation, 0, prompt.len())
            .unwrap();
        let mut published = Vec::new();
        assert!(coordinator.publish_prompt(
            pool,
            &allocation,
            prompt,
            &lookup.page_hashes,
            prompt.len(),
            &mut published,
            true,
            source,
        ));
        allocation
    }

    /// Publishes only the chosen logical pages of one group of `prompt`, each
    /// on its own freshly allocated units, and returns their references.
    fn publish_pages(
        coordinator: &KvCacheCoordinator,
        pool: &BlockPool,
        prompt: &[u32],
        group: usize,
        pages: &[usize],
        source: &Arc<WorkerEndpoint>,
    ) -> Vec<PageRef> {
        let shape = coordinator.groups()[group];
        let hashes = coordinator.prefix_hashes(prompt, None);
        pages
            .iter()
            .map(|page| {
                let reference = pool.allocate(1, shape.units_per_page).unwrap().remove(0);
                let tokens = &prompt[page * shape.page_tokens..(page + 1) * shape.page_tokens];
                pool.publish(
                    &reference,
                    group as u32,
                    hashes[group][*page],
                    tokens,
                    source,
                );
                reference
            })
            .collect()
    }

    #[test]
    fn pages_of_several_units_draw_from_one_pool_and_return_together() {
        // Nine allocatable units: a sliding page takes three, a full page one.
        let pool = BlockPool::new(10);
        let coordinator = KvCacheCoordinator::new(vec![full(8, 1), sliding(4, 3, 100)]);
        let mut allocation = coordinator.empty();

        // Eight tokens need one full page and two sliding pages: 1 + 6 units.
        assert_eq!(coordinator.units_needed(&allocation, 0, 8), Some(7));
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 8)
            .unwrap();
        assert_eq!(pool.free_units(), 2);
        assert_eq!(allocation.tables[1].unit_ids().len(), 6);
        assert!(
            allocation
                .tables
                .iter()
                .flat_map(BlockTable::unit_ids)
                .all(|unit| unit.0 != 0)
        );

        // Growing to twelve tokens needs a full page and a sliding page,
        // four units, but only two are free: nothing is allocated.
        assert!(
            coordinator
                .ensure_capacity(&pool, &mut allocation, 8, 12)
                .is_none()
        );
        assert_eq!(pool.free_units(), 2);
        assert_eq!(allocation.tables[0].end_page(), 1);
        assert_eq!(allocation.tables[1].end_page(), 2);

        drop(allocation);
        assert_eq!(pool.free_units(), 9);
    }

    #[test]
    fn admission_compares_the_sum_of_group_demand_with_free_units() {
        // Eight tokens need two units in one group and three in the other.
        // With one of five units taken, the four free units cover either
        // group alone, which a per-group minimum would accept, but not both.
        let pool = BlockPool::new(6);
        let coordinator = KvCacheCoordinator::new(vec![full(4, 1), full(8, 3)]);
        let mut allocation = coordinator.empty();
        assert_eq!(coordinator.units_needed(&allocation, 0, 8), Some(5));
        let blocker = pool.allocate(1, 1).unwrap();
        assert!(
            coordinator
                .ensure_capacity(&pool, &mut allocation, 0, 8)
                .is_none()
        );
        drop(blocker);
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 8)
            .unwrap();
        assert_eq!(pool.free_units(), 0);
    }

    #[test]
    fn sliding_pages_retire_only_behind_the_window_and_keep_published_copies() {
        let pool = BlockPool::new(64);
        let source = endpoint();
        let coordinator = KvCacheCoordinator::new(vec![full(4, 1), sliding(4, 2, 6)]);
        let prompt = (1..=16).collect::<Vec<u32>>();
        let mut allocation = publish(&coordinator, &pool, &prompt, &source);
        let free = pool.free_units();
        let retired = (0..2)
            .flat_map(|page| allocation.tables[1].page(page).unwrap().units().to_vec())
            .collect::<Vec<_>>();

        // Visible 16 with window 6 keeps tokens [10, 16): pages 2 and 3.
        assert!(coordinator.release_window(&mut allocation, 16));
        let table = &allocation.tables[1];
        assert_eq!((table.start_page(), table.end_page()), (2, 4));
        assert_eq!(table.unit_ids().len(), 4);
        assert!(table.page(1).is_none());
        assert_eq!(allocation.tables[0].start_page(), 0);

        // The two retired pages (four units) return as published, still
        // matchable copies.
        assert_eq!(pool.free_units(), free + 4);
        assert!(
            retired
                .iter()
                .all(|unit| pool.unit_state(*unit) == UnitState::Cached)
        );
        let hit = coordinator.probe_prefix(&pool, &prompt[..13], true, false, None, &source);
        assert_eq!(hit.cached_tokens, 12);

        // Growth appends after the window and never moves the start back.
        coordinator
            .ensure_capacity(&pool, &mut allocation, 16, 20)
            .unwrap();
        assert_eq!(allocation.tables[1].start_page(), 2);
        assert_eq!(allocation.tables[1].end_page(), 5);
    }

    #[test]
    fn unpublished_sliding_pages_return_free() {
        let pool = BlockPool::new(16);
        let coordinator = KvCacheCoordinator::new(vec![sliding(4, 1, 4)]);
        let mut allocation = coordinator.empty();
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 12)
            .unwrap();
        let first = allocation.tables[0].page(0).unwrap().units()[0];
        assert!(coordinator.release_window(&mut allocation, 12));
        assert_eq!(allocation.tables[0].start_page(), 2);
        assert_eq!(pool.unit_state(first), UnitState::Free);
        assert_eq!(pool.free_units(), 14);
    }

    #[test]
    fn a_fresh_sliding_table_starts_at_the_first_page_its_read_needs() {
        let pool = BlockPool::new(32);
        let coordinator = KvCacheCoordinator::new(vec![full(4, 1), sliding(4, 1, 5)]);
        let mut allocation = coordinator.empty();

        // A read from token 16 needs history [11, 20): sliding pages 2..5,
        // while the full group needs pages 0..5.
        assert_eq!(coordinator.units_needed(&allocation, 16, 20), Some(5 + 3));
        coordinator
            .ensure_capacity(&pool, &mut allocation, 16, 20)
            .unwrap();
        assert_eq!(allocation.tables[1].start_page(), 2);
        assert_eq!(allocation.tables[1].end_page(), 5);
        assert_eq!(allocation.tables[0].start_page(), 0);
    }

    #[test]
    fn prefix_validation_iterates_until_every_sliding_window_holds() {
        let pool = BlockPool::new(64);
        let source = endpoint();
        // A prefix of M tokens needs group 0 pages intersecting [M - 4, M)
        // (4-token pages) and group 1 pages intersecting [M - 8, M) (8-token
        // pages). Candidates are multiples of 8.
        let coordinator = KvCacheCoordinator::new(vec![sliding(4, 1, 4), sliding(8, 1, 8)]);
        let prompt = (1..=40).collect::<Vec<u32>>();
        // Group 0 misses pages 3 and 7; group 1 misses page 2. Group 0 lowers
        // 32 to 24, group 1 lowers 24 to 16, group 0 then lowers 16 to 8,
        // where both windows hold: group 0 page 1 and group 1 page 0.
        let _first = publish_pages(&coordinator, &pool, &prompt, 0, &[1, 5], &source);
        let _second = publish_pages(&coordinator, &pool, &prompt, 1, &[0, 1, 3], &source);

        let hit = coordinator.probe_prefix(&pool, &prompt, true, false, None, &source);
        assert_eq!(hit.cached_tokens, 8);

        let mut target = coordinator.empty();
        let lookup = coordinator
            .acquire_prefix(&pool, &mut target, &prompt, true, false, None, &source)
            .unwrap();
        assert_eq!(lookup.cached_tokens, 8);
        assert_eq!(
            (target.tables[0].start_page(), target.tables[0].end_page()),
            (1, 2)
        );
        assert_eq!(
            (target.tables[1].start_page(), target.tables[1].end_page()),
            (0, 1)
        );
    }

    #[test]
    fn full_groups_lock_the_whole_prefix_and_sliding_groups_only_their_window() {
        let pool = BlockPool::new(128);
        let source = endpoint();
        // Full pages of 8 tokens and sliding pages of 4 with a 6-token window.
        let coordinator = KvCacheCoordinator::new(vec![full(8, 1), sliding(4, 2, 6)]);
        let prompt = (1..=40).collect::<Vec<u32>>();
        // Every full page is cached, but the sliding group holds only pages
        // 2 and 3: the window of M = 16. M = 32 and M = 24 need sliding pages
        // 6 and 7, and 4 and 5.
        let full_pages = publish_pages(&coordinator, &pool, &prompt, 0, &[0, 1, 2, 3, 4], &source);
        let sliding_pages = publish_pages(&coordinator, &pool, &prompt, 1, &[2, 3], &source);
        drop(full_pages);
        drop(sliding_pages);

        // Every hit page is unreferenced: two full units and two sliding
        // pages of two units each count as free until acquired.
        let hit = coordinator.probe_prefix(&pool, &prompt, true, false, None, &source);
        assert_eq!(hit.cached_tokens, 16);
        assert_eq!(hit.cached_free_units, 2 + 4);

        let mut target = coordinator.empty();
        let lookup = coordinator
            .acquire_prefix(&pool, &mut target, &prompt, true, false, None, &source)
            .unwrap();
        assert_eq!(lookup.cached_tokens, 16);
        assert_eq!(
            (target.tables[0].start_page(), target.tables[0].end_page()),
            (0, 2)
        );
        assert_eq!(
            (target.tables[1].start_page(), target.tables[1].end_page()),
            (2, 4)
        );
        assert!(
            target
                .tables
                .iter()
                .flat_map(BlockTable::unit_ids)
                .all(|unit| pool.unit_state(unit) == UnitState::Active)
        );
        // Acquired pages hold computed KV and are never declared fresh.
        assert!(
            target
                .tables
                .iter()
                .all(|table| table.fresh_units(0).is_empty())
        );

        // The first prefill chunk appends after the prefix in every group.
        coordinator
            .ensure_capacity(&pool, &mut target, 16, 24)
            .unwrap();
        assert_eq!(target.tables[0].end_page(), 3);
        assert_eq!(target.tables[1].end_page(), 6);
    }

    #[test]
    fn a_full_group_miss_limits_the_sliding_prefix() {
        let pool = BlockPool::new(64);
        let source = endpoint();
        let coordinator = KvCacheCoordinator::new(vec![sliding(4, 1, 4), full(8, 1)]);
        let prompt = (1..=33).collect::<Vec<u32>>();
        let owner = publish(&coordinator, &pool, &prompt[..16], &source);

        // Only 16 prompt tokens were published: the full group limits every
        // group to 16 even though a longer prompt is probed.
        let hit = coordinator.probe_prefix(&pool, &prompt, true, false, None, &source);
        assert_eq!(hit.cached_tokens, 16);
        // Both groups' hit pages are referenced by the owner.
        assert_eq!(hit.cached_free_units, 0);
        drop(owner);
        let hit = coordinator.probe_prefix(&pool, &prompt, true, false, None, &source);
        // Unreferenced hits count as free: sliding page 3 and full pages 0
        // and 1.
        assert_eq!(hit.cached_free_units, 3);
    }

    #[test]
    fn prefix_copies_follow_worker_incarnations_without_releasing_active_pages() {
        let pool = BlockPool::new(5);
        let coordinator = KvCacheCoordinator::new(vec![full(4, 1)]);
        let prompt = [1, 2, 3, 4, 5];
        let first = endpoint();
        let second = Arc::new(WorkerEndpoint {
            worker_id: "replica".into(),
            incarnation: "replica-loaded".into(),
            ..first.as_ref().clone()
        });
        let owners = [&first, &second].map(|source| {
            let allocation = publish(&coordinator, &pool, &prompt, source);
            assert_eq!(
                coordinator
                    .probe_prefix(&pool, &prompt, true, false, None, source)
                    .cached_tokens,
                4
            );
            allocation
        });
        let replacement = WorkerEndpoint {
            incarnation: "reloaded".into(),
            ..first.as_ref().clone()
        };
        assert_eq!(
            coordinator
                .probe_prefix(&pool, &prompt, true, false, None, &replacement)
                .cached_tokens,
            0
        );

        let free_before = pool.free_units();
        pool.invalidate_source(&first);
        assert_eq!(pool.free_units(), free_before);
        assert_eq!(
            coordinator
                .probe_prefix(&pool, &prompt, true, false, None, &first)
                .cached_tokens,
            0
        );
        let mut consumer = coordinator.empty();
        let reused = coordinator
            .acquire_prefix(&pool, &mut consumer, &prompt, true, false, None, &second)
            .unwrap();
        assert_eq!(reused.cached_tokens, 4);
        assert_eq!(
            consumer.tables[0].unit_ids(),
            owners[1].tables[0].unit_ids()[..1]
        );
        let [first_owner, _] = owners;
        drop(first_owner);
        assert_eq!(pool.free_units(), free_before + 2);
    }

    #[test]
    fn fresh_units_follow_the_allocation_serial_in_every_group() {
        let pool = BlockPool::new(32);
        let coordinator = KvCacheCoordinator::new(vec![full(4, 1), sliding(2, 2, 4)]);
        let mut allocation = coordinator.empty();
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 4)
            .unwrap();
        let watermark = allocation.allocated_units();
        assert_eq!(watermark, 1 + 4);
        allocation.tables.iter_mut().for_each(BlockTable::mark_sent);
        assert!(!allocation.tables.iter().any(BlockTable::changed_since_sent));

        // Growth to six tokens adds one full page and one sliding page; only
        // those units are fresh past the watermark, and both tables changed.
        coordinator
            .ensure_capacity(&pool, &mut allocation, 4, 6)
            .unwrap();
        assert_eq!(allocation.tables[0].fresh_units(watermark).len(), 1);
        assert_eq!(allocation.tables[1].fresh_units(watermark).len(), 2);
        assert!(allocation.tables.iter().all(BlockTable::changed_since_sent));
    }
}
