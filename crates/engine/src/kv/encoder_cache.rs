//! Encoder-output cache. A hashed, reference-counted,
//! LRU-evicted cache keyed by image content hash, holding the exact immutable
//! product reference for a worker-side encoder output. On a hit the encode op is
//! skipped; evicted products are reported so the worker can reclaim their
//! physical storage. No embeddings live in the scheduler.

use std::collections::{BTreeMap, HashMap};
use uniserve_worker_ipc::ProductRef;

struct Entry {
    product: ProductRef,
    ref_cnt: u32,
    lru: u64, // monotonic tick, larger == more recent
}

#[derive(Default, Debug, Clone, Copy)]
pub struct EncoderCacheStats {
    pub queries: u64,
    pub hits: u64,
    pub evictions: u64,
    /// inserts that landed while at budget with *every* entry pinned, so
    /// no victim could be evicted and the live set grew past `budget`. This is
    /// bounded by the count of concurrently-pinned entries (itself bounded by
    /// the scheduler's multimodal admission), but a nonzero value means the
    /// encoder budget is under-provisioned for the admitted concurrency.
    pub over_budget_inserts: u64,
}

pub struct EncoderCacheManager {
    budget: usize,                // max cached entries
    entries: HashMap<u64, Entry>, // content hash -> entry
    // Reset removes entries from lookup immediately, but a worker-side tensor
    // cannot be reclaimed while an in-flight request still holds a reference.
    // Retired entries remain here until their final pin is released. A later
    // computation may reactivate an equivalent handle under the same hash.
    retired: HashMap<u64, Vec<Entry>>, // content hash -> pinned retired generations
    // Index of evictable (ref_cnt == 0) entries ordered by LRU tick, so the
    // victim search and `can_insert` are O(log n) / O(1) instead of full scans
    // of `entries`. Invariant: a hash is in `evictable` keyed by its current
    // `Entry::lru` iff its `ref_cnt == 0`. Ticks are globally unique (monotonic
    // `next_tick`), so each entry occupies at most one key here.
    evictable: BTreeMap<u64, u64>, // lru tick -> content hash
    tick: u64,
    pub stats: EncoderCacheStats,
}

impl EncoderCacheManager {
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            entries: HashMap::new(),
            retired: HashMap::new(),
            evictable: BTreeMap::new(),
            tick: 0,
            stats: EncoderCacheStats::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len() + self.retired.values().map(Vec::len).sum::<usize>()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.retired.is_empty()
    }
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Inspect a resident output without changing LRU order or cache metrics.
    pub fn peek_product(&self, hash: u64) -> Option<ProductRef> {
        self.entries.get(&hash).map(|entry| entry.product.clone())
    }

    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Look up the exact product and measured KV effect needed to skip encode.
    pub fn lookup_product(&mut self, hash: u64) -> Option<ProductRef> {
        self.stats.queries += 1;
        let tick = self.next_tick();
        if let Some(e) = self.entries.get_mut(&hash) {
            let old_lru = e.lru;
            let evictable = e.ref_cnt == 0;
            e.lru = tick;
            self.stats.hits += 1;
            // Keep the evictable index keyed by the entry's current LRU tick.
            if evictable {
                self.evictable.remove(&old_lru);
                self.evictable.insert(tick, hash);
            }
            Some(e.product.clone())
        } else {
            None
        }
    }

    /// Whether there's room (under budget, counting only unreferenced evictables)
    /// to insert another entry without exceeding the budget by referenced ones.
    pub fn can_insert(&self) -> bool {
        // room exists if under budget, or some entry is unreferenced (evictable).
        self.len() < self.budget || !self.evictable.is_empty()
    }

    /// Insert a freshly-computed encoder product, evicting the LRU unreferenced
    /// entry if at budget. Returns any freed product.
    /// `budget` bounds the *evictable* working set, not the live set. The
    /// encoder output already exists on the worker by the time it reaches here
    /// and the caller pins it immediately for the in-flight request, so a
    /// freshly-computed handle can never be dropped on the floor. When the cache
    /// is at budget and every entry is pinned (`ref_cnt > 0`), there is no victim
    /// to evict and the live set grows by one; this is bounded by the number of
    /// concurrently-pinned entries (the scheduler admits at most
    /// `max_num_seqs` requests) and is counted in `stats.over_budget_inserts` so
    /// the over-subscription is observable rather than silent. The hard cap is
    /// enforced upstream via [`can_insert`](Self::can_insert) at admission.
    pub fn insert(&mut self, hash: u64, product: ProductRef) -> Option<ProductRef> {
        if self.entries.contains_key(&hash) {
            let tick = self.next_tick();
            if let Some(existing) = self.entries.get_mut(&hash) {
                let existing_product = existing.product.clone();
                let old_lru = existing.lru;
                let evictable = existing.ref_cnt == 0;
                existing.lru = tick;
                if evictable {
                    self.evictable.remove(&old_lru);
                    self.evictable.insert(tick, hash);
                }
                // Concurrent misses compute equivalent output. Preserve the
                // existing entry and its references; a distinct redundant
                // product can be reclaimed by the caller.
                return (existing_product != product).then_some(product);
            }
        }

        let retired_match = self
            .retired
            .get(&hash)
            .and_then(|entries| entries.iter().position(|entry| entry.product == product));
        if let Some(index) = retired_match {
            let mut entry = match self.retired.get_mut(&hash) {
                Some(entries) if index < entries.len() => entries.swap_remove(index),
                _ => unreachable!("retired encoder entry disappeared without mutation"),
            };
            if self.retired.get(&hash).is_some_and(Vec::is_empty) {
                self.retired.remove(&hash);
            }
            entry.lru = self.next_tick();
            self.entries.insert(hash, entry);
            return None;
        }
        let mut freed = None;
        if self.len() >= self.budget {
            // At budget and inserting a new key: evict the LRU unreferenced entry
            // (smallest tick in the evictable index), if any exists.
            if let Some((_, victim)) = self.evictable.pop_first() {
                if let Some(e) = self.entries.remove(&victim) {
                    self.stats.evictions += 1;
                    freed = Some(e.product);
                }
            } else {
                // Every entry is pinned: no victim, so the live set will exceed
                // `budget` until a pinned entry is released. Record it.
                self.stats.over_budget_inserts += 1;
            }
        }
        let tick = self.next_tick();
        self.entries.insert(
            hash,
            Entry {
                product,
                ref_cnt: 0,
                lru: tick,
            },
        );
        // Freshly inserted entries are unreferenced, hence evictable.
        self.evictable.insert(tick, hash);
        freed
    }

    /// Reference a cached entry (pins it against eviction while a request uses it).
    pub fn acquire(&mut self, hash: u64) -> Option<ProductRef> {
        let tick = self.next_tick();
        let e = self.entries.get_mut(&hash)?;
        let old_lru = e.lru;
        let was_unreferenced = e.ref_cnt == 0;
        e.ref_cnt += 1;
        e.lru = tick;
        // A pinned entry is no longer evictable.
        if was_unreferenced {
            self.evictable.remove(&old_lru);
        }
        Some(e.product.clone())
    }

    /// Release a reference. Active entries become evictable at zero references;
    /// retired entries are removed and return their worker handle for reclaim.
    pub fn release(&mut self, hash: u64, product: &ProductRef) -> Option<ProductRef> {
        if let Some(e) = self.entries.get_mut(&hash)
            && &e.product == product
            && e.ref_cnt > 0
        {
            e.ref_cnt -= 1;
            // Dropping the last reference makes the entry evictable again.
            if e.ref_cnt == 0 {
                let lru = e.lru;
                self.evictable.insert(lru, hash);
            }
            return None;
        }

        let mut freed = None;
        let mut remove_hash = false;
        if let Some(entries) = self.retired.get_mut(&hash)
            && let Some(index) = entries
                .iter()
                .position(|entry| &entry.product == product && entry.ref_cnt > 0)
        {
            entries[index].ref_cnt -= 1;
            if entries[index].ref_cnt == 0 {
                freed = Some(entries.swap_remove(index).product);
            }
            remove_hash = entries.is_empty();
        }
        if remove_hash {
            self.retired.remove(&hash);
        }
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::RequestId;
    use uniserve_worker_ipc::{
        DType, OpId, PointRange, ProductKind, RequestKey, ShapeBound, StorageClass,
    };

    fn product(generation: u32) -> ProductRef {
        ProductRef {
            request_key: RequestKey::new(7, RequestId(u64::from(generation)), 3),
            producer_op_id: OpId(u64::from(generation)),
            output_index: 0,
            generation,
            kind: ProductKind::VisionFeature,
            storage_class: StorageClass::LatentArena,
            dtype: DType::BF16,
            shape_bound: ShapeBound::default(),
            point_range: PointRange::default(),
        }
    }

    #[test]
    fn lookup_returns_the_exact_producer_reference() {
        let mut cache = EncoderCacheManager::new(4);
        let expected = product(100);
        cache.insert(1, expected.clone());
        assert_eq!(cache.lookup_product(1), Some(expected));
        assert_eq!(cache.stats.queries, 1);
        assert_eq!(cache.stats.hits, 1);
    }

    #[test]
    fn lru_eviction_reports_the_exact_reclaimable_product() {
        let mut cache = EncoderCacheManager::new(2);
        let first = product(100);
        let second = product(200);
        let third = product(300);
        cache.insert(1, first.clone());
        cache.insert(2, second.clone());
        cache.lookup_product(1);
        assert_eq!(cache.insert(3, third), Some(second));
        assert_eq!(cache.lookup_product(1), Some(first));
    }

    #[test]
    fn pinned_products_remain_exact_under_capacity_pressure() {
        let mut cache = EncoderCacheManager::new(1);
        let pinned = product(100);
        cache.insert(1, pinned.clone());
        assert_eq!(cache.acquire(1), Some(pinned.clone()));
        cache.insert(2, product(200));
        assert_eq!(cache.lookup_product(1), Some(pinned));
        assert_eq!(cache.stats.over_budget_inserts, 1);
    }
}
