//! Reference-counted cache of worker-resident encoder products.
//!
//! Content-derived keys map to immutable product references. Unpinned entries
//! participate in LRU eviction, and eviction returns the product identity so the
//! worker can reclaim its storage.

use std::collections::{BTreeMap, HashMap};
use uniserve_worker_ipc::TensorRef;

struct Entry {
    product: TensorRef,
    ref_cnt: u32,
    /// Monotonic access tick; larger values are more recent.
    lru: u64,
}

/// Interval counters for encoder-cache lookups and hits.
#[derive(Default, Debug, Clone, Copy)]
pub(crate) struct EncoderCacheStats {
    pub(crate) queries: u64,
    pub(crate) hits: u64,
    pub(crate) evictions: u64,
    /// inserts that landed while at budget with *every* entry pinned, so
    /// no victim could be evicted and the live set grew past `budget`. This is
    /// bounded by the count of concurrently-pinned entries (itself bounded by
    /// the scheduler's multimodal admission), but a nonzero value means the
    /// encoder budget is under-provisioned for the admitted concurrency.
    pub(crate) over_budget_inserts: u64,
}

/// Scheduler-owned reference and eviction state for encoder products.
pub(crate) struct EncoderCacheManager {
    budget: usize,                // max cached entries
    entries: HashMap<u64, Entry>, // content hash -> entry
    // Reset removes entries from lookup immediately, but a worker-side tensor
    // cannot be reclaimed while an in-flight request still holds a reference.
    // Retired entries remain here until their final pin is released. A later
    // computation may reactivate an equivalent handle under the same hash.
    retired: HashMap<u64, Vec<Entry>>, // content hash -> pinned retired generations
    // Index of evictable (ref_cnt == 0) entries ordered by LRU tick, so the
    // victim search is O(log n) instead of full scans
    // of `entries`. Invariant: a hash is in `evictable` keyed by its current
    // `Entry::lru` iff its `ref_cnt == 0`. Ticks are globally unique (monotonic
    // `next_tick`), so each entry occupies at most one key here.
    evictable: BTreeMap<u64, u64>, // lru tick -> content hash
    tick: u64,
    pub(crate) stats: EncoderCacheStats,
}

impl EncoderCacheManager {
    /// Creates an initialized instance.
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            budget,
            entries: HashMap::new(),
            retired: HashMap::new(),
            evictable: BTreeMap::new(),
            tick: 0,
            stats: EncoderCacheStats::default(),
        }
    }

    /// Returns the number of entries.
    pub(crate) fn len(&self) -> usize {
        self.entries.len() + self.retired.values().map(Vec::len).sum::<usize>()
    }
    /// Returns the cache byte budget.
    pub(crate) fn budget(&self) -> usize {
        self.budget
    }

    /// Returns a resident output without changing LRU order or cache metrics.
    pub(crate) fn peek_product(&self, hash: u64) -> Option<TensorRef> {
        self.entries.get(&hash).map(|entry| entry.product.clone())
    }

    /// Revokes unavailable products from lookup while preserving active consumer pins.
    pub(crate) fn invalidate_buffers(
        &mut self,
        buffers: &std::collections::HashSet<uniserve_worker_ipc::BufferId>,
    ) -> Vec<TensorRef> {
        let revoked = self
            .entries
            .extract_if(|_, entry| buffers.contains(&entry.product.buffer_id()))
            .collect::<Vec<_>>();
        let mut reclaimable = Vec::new();
        for (hash, entry) in revoked {
            self.evictable.remove(&entry.lru);
            self.stats.evictions += 1;
            if entry.ref_cnt == 0 {
                reclaimable.push(entry.product);
            } else {
                self.retired.entry(hash).or_default().push(entry);
            }
        }
        reclaimable
    }

    /// Returns and advances the cache recency counter.
    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Looks up the exact product and measured KV effect needed to skip encoding.
    pub(crate) fn lookup_product(&mut self, hash: u64) -> Option<TensorRef> {
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

    /// Removes the least-recently-used unpinned entry so its worker buffer can
    /// be freed before a replacement encoder call is admitted.
    pub(crate) fn evict_one(&mut self) -> Option<TensorRef> {
        let (_, victim) = self.evictable.pop_first()?;
        // An index key without a resident entry evicts nothing, as in `insert`.
        let entry = self.entries.remove(&victim)?;
        debug_assert_eq!(entry.ref_cnt, 0);
        self.stats.evictions += 1;
        Some(entry.product)
    }

    /// Inserts a freshly-computed encoder product, evicting the LRU unreferenced
    /// entry if at budget. Returns any freed product.
    /// `budget` bounds the *evictable* working set, not the live set. The
    /// encoder output already exists on the worker by the time it reaches here
    /// and the caller pins it immediately for the in-flight request, so a
    /// freshly-computed handle can never be dropped on the floor. When the cache
    /// is at budget and every entry is pinned (`ref_cnt > 0`), there is no victim
    /// to evict and the live set grows by one; this is bounded by the number of
    /// concurrently-pinned entries (the scheduler admits at most
    /// `max_num_seqs` requests) and is counted in `stats.over_budget_inserts` so
    /// the over-subscription is observable. Admission reserves encoder-product
    /// capacity before submitting the computation.
    pub(crate) fn insert(&mut self, hash: u64, product: TensorRef) -> Option<TensorRef> {
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

    /// Acquires a cached entry, pinning it against eviction while the request uses it.
    pub(crate) fn acquire(&mut self, hash: u64) -> Option<TensorRef> {
        let tick = self.next_tick();
        let e = self.entries.get_mut(&hash)?;
        let old_lru = e.lru;
        let was_unreferenced = e.ref_cnt == 0;
        e.ref_cnt += 1;
        e.lru = tick;
        // Pinned entries are excluded from the eviction index.
        if was_unreferenced {
            self.evictable.remove(&old_lru);
        }
        Some(e.product.clone())
    }

    /// Releases a reference. Active entries become evictable at zero references;
    /// retired entries are removed and return their worker handle for reclaim.
    pub(crate) fn release(&mut self, hash: u64, product: &TensorRef) -> Option<TensorRef> {
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
    use uniserve_worker_ipc::{CallId, DType, RequestKey, ShapeBound};

    fn product(generation: u32) -> TensorRef {
        TensorRef {
            request_key: RequestKey::new(7, RequestId(u64::from(generation)), 3),
            producer_call_id: CallId::new(u64::from(generation), 0),
            output_index: 0,
            generation,
            dtype: DType::BF16,
            shape_bound: ShapeBound::default(),
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

    #[test]
    fn invalidation_preserves_pins_and_independent_replacements() {
        let mut cache = EncoderCacheManager::new(4);
        let pinned = product(100);
        let unpinned = product(200);
        let replacement = product(300);
        cache.insert(1, pinned.clone());
        cache.insert(2, unpinned.clone());
        assert_eq!(cache.acquire(1), Some(pinned.clone()));
        assert_eq!(cache.acquire(1), Some(pinned.clone()));
        let lost = [pinned.buffer_id(), unpinned.buffer_id()]
            .into_iter()
            .collect();
        assert_eq!(cache.invalidate_buffers(&lost), vec![unpinned]);
        assert_eq!(cache.lookup_product(1), None);
        assert_eq!(cache.insert(1, replacement.clone()), None);
        assert_eq!(cache.release(1, &pinned), None);
        assert_eq!(cache.release(1, &pinned), Some(pinned.clone()));
        assert_eq!(cache.release(1, &pinned), None);
        assert_eq!(cache.lookup_product(1), Some(replacement));
    }
}
