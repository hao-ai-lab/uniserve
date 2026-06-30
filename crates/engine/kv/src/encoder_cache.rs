//! Encoder-output cache. A hashed, reference-counted,
//! LRU-evicted cache keyed by image content hash, holding the *logical* handle to
//! a worker-side encoder output, bounded by a budget. On a hit the encode op is
//! skipped; evicted/freed handles are reported so the worker can reclaim the
//! physical encoder memory. Only handles and hashes live here — no embeddings.

use std::collections::{BTreeMap, HashMap};

struct Entry {
    handle: u64,
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
            budget: budget.max(1),
            entries: HashMap::new(),
            evictable: BTreeMap::new(),
            tick: 0,
            stats: EncoderCacheStats::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn budget(&self) -> usize {
        self.budget
    }

    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Look up a cached encoder handle by content hash (counts a query/hit).
    pub fn lookup(&mut self, hash: u64) -> Option<u64> {
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
            Some(e.handle)
        } else {
            None
        }
    }

    /// Whether there's room (under budget, counting only unreferenced evictables)
    /// to insert another entry without exceeding the budget by referenced ones.
    pub fn can_insert(&self) -> bool {
        // room exists if under budget, or some entry is unreferenced (evictable).
        self.entries.len() < self.budget || !self.evictable.is_empty()
    }

    /// Insert a freshly-computed encoder handle, evicting the LRU unreferenced
    /// entry if at budget. Returns any freed handle (to report to the worker).

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
    pub fn insert(&mut self, hash: u64, handle: u64) -> Option<u64> {
        let mut freed = None;
        // Replacing an existing entry resets its ref_cnt to 0; drop its stale
        // evictable index slot first so the invariant holds.
        let existing_evictable_tick = self
            .entries
            .get(&hash)
            .filter(|old| old.ref_cnt == 0)
            .map(|old| old.lru);
        if let Some(old_lru) = existing_evictable_tick {
            self.evictable.remove(&old_lru);
        } else if !self.entries.contains_key(&hash) && self.entries.len() >= self.budget {
            // At budget and inserting a new key: evict the LRU unreferenced entry
            // (smallest tick in the evictable index), if any exists.
            if let Some((_, victim)) = self.evictable.pop_first() {
                if let Some(e) = self.entries.remove(&victim) {
                    self.stats.evictions += 1;
                    freed = Some(e.handle);
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
                handle,
                ref_cnt: 0,
                lru: tick,
            },
        );
        // Freshly inserted entries are unreferenced, hence evictable.
        self.evictable.insert(tick, hash);
        freed
    }

    /// Reference a cached entry (pins it against eviction while a request uses it).
    pub fn acquire(&mut self, hash: u64) -> Option<u64> {
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
        Some(e.handle)
    }

    /// Release a reference (entry becomes evictable at ref_cnt 0).
    pub fn release(&mut self, hash: u64) {
        if let Some(e) = self.entries.get_mut(&hash)
            && e.ref_cnt > 0
        {
            e.ref_cnt -= 1;
            // Dropping the last reference makes the entry evictable again.
            if e.ref_cnt == 0 {
                let lru = e.lru;
                self.evictable.insert(lru, hash);
            }
        }
    }

    /// Clear the whole cache (the `/reset_encoder_cache` action). Returns all
    /// freed handles to report to the worker.
    pub fn clear(&mut self) -> Vec<u64> {
        let freed: Vec<u64> = self.entries.values().map(|e| e.handle).collect();
        self.entries.clear();
        self.evictable.clear();
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_and_miss() {
        let mut c = EncoderCacheManager::new(4);
        assert_eq!(c.lookup(1), None);
        c.insert(1, 100);
        assert_eq!(c.lookup(1), Some(100));
        assert_eq!(c.stats.queries, 2);
        assert_eq!(c.stats.hits, 1);
    }

    #[test]
    fn lru_eviction_reports_freed_handle() {
        let mut c = EncoderCacheManager::new(2);
        c.insert(1, 100);
        c.insert(2, 200);
        c.lookup(1); // touch 1 -> 2 is now LRU
        let freed = c.insert(3, 300);
        assert_eq!(freed, Some(200), "LRU entry 2 should be evicted");
        assert_eq!(c.lookup(2), None);
        assert!(c.lookup(1).is_some() && c.lookup(3).is_some());
    }

    #[test]
    fn referenced_entry_not_evicted() {
        let mut c = EncoderCacheManager::new(1);
        c.insert(1, 100);
        c.acquire(1); // pin
        assert!(!c.can_insert(), "no room: the only slot is referenced");
        // inserting anyway shouldn't evict the referenced entry
        let freed = c.insert(2, 200);
        assert_eq!(freed, None);
        assert!(c.lookup(1).is_some());
    }

    // at budget with every entry pinned, a (necessarily pinned-on-arrival)
    // insert grows the live set past `budget` instead of dropping the handle,
    // and the over-subscription is counted rather than silent.
    #[test]
    fn all_pinned_insert_grows_live_set_and_counts_over_budget() {
        let mut c = EncoderCacheManager::new(2);
        c.insert(1, 100);
        c.acquire(1);
        c.insert(2, 200);
        c.acquire(2);
        assert!(!c.can_insert(), "both slots pinned: no admission room");
        assert_eq!(c.stats.over_budget_inserts, 0);

        // A third encoder output arrives while both are in use: it must be kept
        // (the worker already computed it) even though it exceeds budget.
        let freed = c.insert(3, 300);
        assert_eq!(freed, None, "no unpinned victim to evict");
        assert_eq!(c.len(), 3, "live set transiently exceeds budget");
        assert_eq!(c.stats.over_budget_inserts, 1);
        assert!(c.lookup(1).is_some() && c.lookup(2).is_some() && c.lookup(3).is_some());

        // Once a pinned entry is released, the budget is reclaimed on next insert.
        c.release(1);
        let freed = c.insert(4, 400);
        assert_eq!(
            freed,
            Some(100),
            "released entry 1 is now the evictable victim"
        );
    }

    #[test]
    fn release_makes_entry_evictable_again() {
        // Exercises the evictable index across acquire/release so the LRU victim
        // search stays correct after pinning churn.
        let mut c = EncoderCacheManager::new(2);
        c.insert(1, 100);
        c.insert(2, 200);
        c.acquire(2); // pin 2 -> only 1 is evictable, but acquire makes 2 newest
        assert!(c.can_insert(), "1 is unreferenced, so there is room");
        c.release(2); // 2 becomes evictable again; its LRU tick is newer than 1
        // At budget; inserting a new key must evict the LRU unreferenced entry,
        // which is 1 (touched at insert) — 2's tick advanced on acquire.
        let freed = c.insert(3, 300);
        assert_eq!(freed, Some(100), "1 is the LRU unreferenced entry");
        assert_eq!(c.lookup(1), None);
        assert!(c.lookup(2).is_some() && c.lookup(3).is_some());
    }

    #[test]
    fn reinsert_same_hash_keeps_index_consistent() {
        // Re-inserting an existing hash resets ref_cnt; the evictable index must
        // not retain a stale slot for the old tick.
        let mut c = EncoderCacheManager::new(2);
        c.insert(1, 100);
        c.acquire(1); // pin -> not evictable
        c.insert(1, 101); // overwrite resets ref_cnt to 0 -> evictable again
        assert_eq!(c.len(), 1);
        c.insert(2, 200); // fills budget, no eviction needed
        let freed = c.insert(3, 300); // at budget -> evict LRU unreferenced (1)
        assert_eq!(freed, Some(101), "re-inserted entry 1 is the LRU evictable");
        assert_eq!(c.lookup(1), None);
    }

    #[test]
    fn clear_reports_all() {
        let mut c = EncoderCacheManager::new(4);
        c.insert(1, 10);
        c.insert(2, 20);
        let mut freed = c.clear();
        freed.sort();
        assert_eq!(freed, vec![10, 20]);
        assert!(c.is_empty());
    }
}
