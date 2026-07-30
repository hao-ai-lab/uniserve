//! Logical KV block management for the engine layer. Owns
//! block identity, ownership, the reference-counted state machine, an intrusive
//! LRU free/eviction queue, one-or-more KV-cache groups (hybrid attention),
//! sliding-window/sink trimming, the per-request segment table, the scratch
//! budget, the prefix-cache hash→block map, and a cache-event stream.
//! Holds NO GPU memory and speaks only integer block ids — the worker's physical
//! pool turns ids into tensors.
//!
//! **substrate.** Blocks are reference-counted; a block becomes eligible
//! for reuse only at `ref_cnt == 0`, when it is appended to its group's free
//! queue (most-recently-used end). Allocation pops the queue's LRU end; if the
//! popped block was *cached* (retains a prefix hash) its hash is dropped from the
//! map and a `BlockRemoved` event is emitted. This is exactly the reference's
//! `FreeKVCacheBlockQueue` + `BlockPool` mechanism (`block_pool.py`), and it is
//! the substrate prefix caching and preemption-recompute need.
//!
//! The single-group, no-prefix-cache path (BAGEL today) is byte-for-byte the
//! previous all-or-nothing allocator: every released block has no hash, so it
//! returns straight to the free queue as `Free`.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::collections::{HashMap, VecDeque};
use uniserve_core::{BlockId, HashAlgo, KvGroupKind, Modality, RequestId};

mod encoder_cache;
mod freeq;
pub use encoder_cache::{EncoderCacheManager, EncoderCacheStats};
use freeq::BlockMeta;

/// Incremental per-block prefix hash, chained as
/// `H(parent_hash, group_id, modality, token_count, token_ids)`. Modality is
/// folded in so a text (Und) prefix and an image-latent (Gen) prefix never
/// collide (the `BlockHashWithGroupId` analog). Block *position* is captured
/// implicitly by the `parent` chain (block i's hash depends on every preceding
/// block), and the block's `token_count` is folded in explicitly so two blocks
/// with identical leading tokens but different lengths (e.g. a short final
/// block) cannot alias. Pluggable: a fast FNV-1a mixer by default (the role
/// xxhash plays) or SHA-256 for a cryptographic digest.
pub fn block_hash(
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
                    *h ^= *b as u64;
                    *h = h.wrapping_mul(PRIME);
                }
            };
            mix(&parent.to_le_bytes(), &mut h);
            mix(&group_id.to_le_bytes(), &mut h);
            mix(&[modality_tag], &mut h);
            mix(&token_count.to_le_bytes(), &mut h);
            for t in tokens {
                mix(&t.to_le_bytes(), &mut h);
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
            for t in tokens {
                hasher.update(t.to_le_bytes());
            }
            let d = hasher.finalize();
            u64::from_le_bytes(d[0..8].try_into().unwrap())
        }
    }
}

/// Modality tag folded into block hashes.
pub fn modality_tag(m: Modality) -> u8 {
    match m {
        Modality::Und => 0,
        Modality::Gen => 1,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockState {
    Free,
    Reserved {
        owner: RequestId,
    },
    Active {
        owner: RequestId,
    },
    /// Reference count reached 0 but the block retains a prefix hash and is
    /// reusable until evicted. Lives in the free queue, LRU-ordered.
    Cached,
    /// Block 0: the CUDA-graph padding slot — never handed to a request.
    Padding,
}

/// One contiguous single-modality span of a request.
#[derive(Debug, Clone)]
pub struct Segment {
    pub modality: Modality,
    pub pos_start: u32,
    pub pos_end: u32,
    pub block_ids: Vec<BlockId>,
    pub phase: String,
}

#[derive(Debug, Default)]
pub struct SegmentTable {
    pub segments: Vec<Segment>,
}

/// Cache-observability events, surfaced via `/stats` and Prometheus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheEvent {
    BlockStored { hash: u64, block: BlockId },
    BlockRemoved { hash: u64, block: BlockId },
    AllBlocksCleared,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BlockManagerStats {
    pub total: usize,
    pub free: usize,
    pub allocations: u64,
    pub evictions: u64,
    pub blocks_stored: u64,
}

/// One KV-cache group: a block subspace with its own attention kind + free queue
/// head/tail (the links live in the shared `meta` arena).
struct Group {
    kind: KvGroupKind,
    fq_head: Option<u32>, // LRU end (allocate/evict from here); None == empty
    fq_tail: Option<u32>, // MRU end (freed blocks appended here)
    free_count: usize,
    total: usize,
}

/// All per-request KV state, consolidated under a single `RequestId` key so a
/// request's blocks, segment table, scratch reservation, and trim offsets can
/// never drift out of sync across parallel maps.
#[derive(Debug, Default)]
struct RequestKvState {
    blocks: Vec<BlockId>,
    // `None` => no segment table has been materialized for this request (the
    // previous "absent from the `segments` map" state); `Some` => present, even
    // when empty.
    segments: Option<SegmentTable>,
    scratch_res: u64,
    // true token-start offset of each *currently-retained* block, parallel
    // to `blocks`. Sliding-window trimming removes interior blocks, so after the
    // first trim the surviving blocks are no longer token-contiguous-from-0 and
    // positional `idx*block_size` would mis-map. This table preserves each
    // survivor's real offset so repeated trims stay correct. `None` => the block
    // list is still contiguous from 0 (the post-allocate / prefix-cache-hit
    // baseline), so positional offsets are valid.
    block_tok_starts: Option<Vec<usize>>,
}

impl RequestKvState {
    fn is_empty(&self) -> bool {
        self.blocks.is_empty()
            && self.segments.is_none()
            && self.scratch_res == 0
            && self.block_tok_starts.is_none()
    }
}

pub struct BlockManager {
    block_size: usize,
    num_blocks: usize,
    meta: Vec<BlockMeta>,
    groups: Vec<Group>,
    block_group: Vec<u32>, // group id per block
    requests: HashMap<RequestId, RequestKvState>,
    // prefix cache: hash -> the cached block holding that prefix.
    hash_to_block: HashMap<u64, BlockId>,
    // scratch budget
    scratch_capacity: u64,
    scratch_used: u64,
    // cache events (bounded ring)
    events: VecDeque<CacheEvent>,
    events_cap: usize,
    pub stats: BlockManagerStats,
}

impl BlockManager {
    /// Single full-attention group spanning `[1, num_blocks)` (BAGEL default).
    pub fn new(num_blocks: usize, block_size: usize, scratch_capacity: u64) -> Self {
        // group spans [1, num_blocks): block 0 is the padding slot.
        Self::with_groups(
            num_blocks,
            block_size,
            scratch_capacity,
            &[(KvGroupKind::Full, 1, num_blocks.saturating_sub(1) as u32)],
        )
    }

    /// Validate that `(kind, first_block, count)` groups partition `[1, num_blocks)`
    /// without gaps, overlaps, or out-of-range/overflowing ranges. Returns
    /// the offending group on failure so a malformed worker handshake can be
    /// rejected with a clear message instead of an opaque out-of-bounds panic or
    /// silent free-queue corruption. `block 0` is always reserved as padding, so
    /// every group must start at `>= 1`.
    pub fn validate_group_specs(
        num_blocks: usize,
        group_specs: &[(KvGroupKind, u32, u32)],
    ) -> Result<(), String> {
        if num_blocks == 0 {
            return Err("num_blocks must be >= 1 (block 0 is padding)".to_string());
        }
        // covered[b] tracks which group (if any) claims block b; index 0 is padding.
        let mut covered = vec![false; num_blocks];
        covered[0] = true; // padding slot, never claimable by a group
        for (gid, (_, first, count)) in group_specs.iter().enumerate() {
            if *first < 1 {
                return Err(format!(
                    "group {gid}: first_block {first} must be >= 1 (block 0 is padding)"
                ));
            }
            let end = (*first as u64)
                .checked_add(*count as u64)
                .ok_or_else(|| format!("group {gid}: first+count overflows"))?;
            if end > num_blocks as u64 {
                return Err(format!(
                    "group {gid}: range [{first}, {end}) exceeds num_blocks {num_blocks}"
                ));
            }
            for b in *first..(*first + *count) {
                let slot = &mut covered[b as usize];
                if *slot {
                    return Err(format!("group {gid}: block {b} overlaps another group"));
                }
                *slot = true;
            }
        }
        if let Some(b) = covered.iter().position(|c| !c) {
            return Err(format!(
                "block {b} is not covered by any group ([1, num_blocks) must be fully partitioned)"
            ));
        }
        Ok(())
    }

    /// Build with explicit `(kind, first_block, count)` groups. `first_block`
    /// must be >= 1 (block 0 is padding) and groups must partition `[1, num_blocks)`.
    ///
    /// Panics with a descriptive message (via [`Self::validate_group_specs`]) if
    /// the specs are malformed — a buggy/version-skewed worker handshake that
    /// reports overlapping, gapped, or out-of-range ranges fails loudly here
    /// rather than corrupting `block_group`/free-queue accounting. Callers
    /// that can recover should pre-validate with `validate_group_specs` and reject
    /// the handshake instead of constructing.
    pub fn with_groups(
        num_blocks: usize,
        block_size: usize,
        scratch_capacity: u64,
        group_specs: &[(KvGroupKind, u32, u32)],
    ) -> Self {
        if let Err(e) = Self::validate_group_specs(num_blocks, group_specs) {
            panic!("BlockManager::with_groups: invalid KV-cache group specs: {e}");
        }
        let mut meta: Vec<BlockMeta> = (0..num_blocks)
            .map(|_| BlockMeta::new(BlockState::Free))
            .collect();
        meta[0].state = BlockState::Padding;
        let mut block_group = vec![0u32; num_blocks];
        let mut groups = Vec::new();
        for (gid, (kind, first, count)) in group_specs.iter().enumerate() {
            for b in *first..(*first + *count) {
                block_group[b as usize] = gid as u32;
            }
            groups.push(Group {
                kind: *kind,
                fq_head: None,
                fq_tail: None,
                free_count: 0,
                total: *count as usize,
            });
        }
        let mut mgr = Self {
            block_size,
            num_blocks,
            meta,
            groups,
            block_group,
            requests: HashMap::new(),
            hash_to_block: HashMap::new(),
            scratch_capacity,
            scratch_used: 0,
            events: VecDeque::new(),
            events_cap: 4096,
            stats: BlockManagerStats {
                total: num_blocks,
                free: 0,
                allocations: 0,
                evictions: 0,
                blocks_stored: 0,
            },
        };
        // Seed each group's free queue with its blocks, ascending id (so block 1
        // is reclaimed first — matches the previous `(1..n).rev` pop order via
        // pop_front of an ascending push).
        for (gid, (_, first, count)) in group_specs.iter().enumerate() {
            for b in *first..(*first + *count) {
                mgr.fq_push_back(gid, BlockId(b));
            }
        }
        mgr.stats.free = mgr.total_free();
        mgr
    }

    // ---- free-queue primitives (intrusive doubly-linked over `meta`) ----

    fn fq_push_back(&mut self, gid: usize, id: BlockId) {
        let i = id.0;
        let tail = self.groups[gid].fq_tail;
        self.meta[id.0 as usize].fq_prev = tail;
        self.meta[id.0 as usize].fq_next = None;
        self.meta[id.0 as usize].in_fq = true;
        if let Some(tail) = tail {
            self.meta[tail as usize].fq_next = Some(i);
        } else {
            self.groups[gid].fq_head = Some(i);
        }
        self.groups[gid].fq_tail = Some(i);
        self.groups[gid].free_count += 1;
    }

    fn fq_pop_front(&mut self, gid: usize) -> Option<BlockId> {
        let head = self.groups[gid].fq_head?;
        self.fq_unlink_idx(gid, head);
        Some(BlockId(head))
    }

    fn fq_unlink(&mut self, id: BlockId) {
        if !self.meta[id.0 as usize].in_fq {
            return;
        }
        let gid = self.block_group[id.0 as usize] as usize;
        self.fq_unlink_idx(gid, id.0);
    }

    fn fq_unlink_idx(&mut self, gid: usize, i: u32) {
        let prev = self.meta[i as usize].fq_prev;
        let next = self.meta[i as usize].fq_next;
        if let Some(prev) = prev {
            self.meta[prev as usize].fq_next = next;
        } else {
            self.groups[gid].fq_head = next;
        }
        if let Some(next) = next {
            self.meta[next as usize].fq_prev = prev;
        } else {
            self.groups[gid].fq_tail = prev;
        }
        self.meta[i as usize].fq_prev = None;
        self.meta[i as usize].fq_next = None;
        self.meta[i as usize].in_fq = false;
        self.groups[gid].free_count -= 1;
    }

    fn total_free(&self) -> usize {
        self.groups.iter().map(|g| g.free_count).sum()
    }

    fn push_event(&mut self, ev: CacheEvent) {
        if self.events.len() >= self.events_cap {
            self.events.pop_front();
        }
        self.events.push_back(ev);
    }

    // ---- public API (unchanged single-group semantics) ----

    pub fn free_blocks(&self) -> usize {
        self.total_free()
    }
    pub fn free_blocks_in_group(&self, gid: usize) -> usize {
        self.groups.get(gid).map(|g| g.free_count).unwrap_or(0)
    }
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }
    pub fn group_kind(&self, gid: usize) -> Option<KvGroupKind> {
        self.groups.get(gid).map(|g| g.kind)
    }
    /// Total blocks in a group (the `get_kv_cache_specs` capacity analog).
    pub fn group_capacity(&self, gid: usize) -> usize {
        self.groups.get(gid).map(|g| g.total).unwrap_or(0)
    }

    pub fn blocks_needed(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }
    pub fn padding_block(&self) -> BlockId {
        BlockId(0)
    }

    /// All-or-nothing allocation from group 0 (the BAGEL default).
    pub fn allocate(&mut self, req: RequestId, n: usize) -> Option<Vec<BlockId>> {
        self.allocate_in(req, 0, n)
    }

    /// All-or-nothing allocation from a specific group; pops the LRU end, evicting
    /// cached blocks (dropping their hash) when no truly-free block remains.
    pub fn allocate_in(&mut self, req: RequestId, gid: usize, n: usize) -> Option<Vec<BlockId>> {
        if gid >= self.groups.len() || n > self.groups[gid].free_count {
            return None;
        }
        let mut got = Vec::with_capacity(n);
        for _ in 0..n {
            let b = self.fq_pop_front(gid).expect("free_count invariant");
            // Block 0 is the padding slot and is never seeded into any group's
            // free queue, so it can never be popped here; assert the invariant so
            // a future group-seeding bug surfaces loudly instead of corrupting the
            // worker's padding row (host usable capacity == num_blocks-1).
            debug_assert!(b.0 != 0, "padding block 0 must never be allocated");
            // evict a cached block: drop its hash from the map.
            if let Some(h) = self.meta[b.0 as usize].hash.take() {
                // drop the stored tokens too so a re-cached block can't
                // ever be content-matched against a stale prefix.
                self.meta[b.0 as usize].tokens = Vec::new();
                self.hash_to_block.remove(&h);
                self.stats.evictions += 1;
                self.push_event(CacheEvent::BlockRemoved { hash: h, block: b });
            }
            self.meta[b.0 as usize].state = BlockState::Reserved { owner: req };
            self.meta[b.0 as usize].ref_cnt = 1;
            got.push(b);
        }
        let rec = self.requests.entry(req).or_default();
        rec.blocks.extend(got.iter().copied());
        // Growing the block list re-establishes positional layout reasoning;
        // discard any stale per-block trim offsets.
        rec.block_tok_starts = None;
        rec.segments.get_or_insert_with(SegmentTable::default);
        self.stats.allocations += 1;
        self.stats.free = self.total_free();
        Some(got)
    }

    /// Grow the request's block list (group 0) to cover `total_tokens`.
    pub fn ensure_capacity(&mut self, req: RequestId, total_tokens: usize) -> bool {
        let have = self.requests.get(&req).map(|r| r.blocks.len()).unwrap_or(0);
        let need = self.blocks_needed(total_tokens);
        if need <= have {
            return true;
        }
        self.allocate(req, need - have).is_some()
    }

    pub fn grow(&mut self, req: RequestId) -> Option<BlockId> {
        self.allocate(req, 1).map(|v| v[0])
    }

    /// First write moves Reserved -> Active.
    pub fn activate(&mut self, req: RequestId) {
        if let Some(rec) = self.requests.get(&req) {
            let blocks = rec.blocks.clone();
            for b in blocks {
                if let BlockState::Reserved { .. } = self.meta[b.0 as usize].state {
                    self.meta[b.0 as usize].state = BlockState::Active { owner: req };
                }
            }
        }
    }

    pub fn blocks_for(&self, req: RequestId) -> &[BlockId] {
        self.requests
            .get(&req)
            .map(|r| r.blocks.as_slice())
            .unwrap_or(&[])
    }

    pub fn add_segment(&mut self, req: RequestId, seg: Segment) {
        self.requests
            .entry(req)
            .or_default()
            .segments
            .get_or_insert_with(SegmentTable::default)
            .segments
            .push(seg);
    }
    pub fn segment_table(&self, req: RequestId) -> Option<&SegmentTable> {
        self.requests.get(&req).and_then(|r| r.segments.as_ref())
    }

    // ---- reference counting ----

    pub fn ref_count(&self, b: BlockId) -> u32 {
        self.meta[b.0 as usize].ref_cnt
    }
    pub fn block_state(&self, b: BlockId) -> BlockState {
        self.meta[b.0 as usize].state
    }

    /// Decrement a block's ref_cnt; at 0 it joins the free queue (Cached if it
    /// retains a hash, else Free) — reusable/evictable, not destroyed.
    fn deref_block(&mut self, b: BlockId) {
        let m = &mut self.meta[b.0 as usize];
        if m.ref_cnt > 0 {
            m.ref_cnt -= 1;
        }
        if m.ref_cnt == 0 && !m.in_fq && m.state != BlockState::Padding {
            let gid = self.block_group[b.0 as usize] as usize;
            self.meta[b.0 as usize].state = if self.meta[b.0 as usize].hash.is_some() {
                BlockState::Cached
            } else {
                BlockState::Free
            };
            self.fq_push_back(gid, b);
        }
    }

    // ---- prefix-cache substrate ----

    // Collision safety: a cached block retains the exact token ids
    // whose content produced its `hash` (`BlockMeta::tokens`). `lookup_cached`
    // and `acquire_cached` both re-verify the candidate prefix's tokens against
    // the stored tokens on a hash hit and treat a mismatch as a miss, so a
    // 64-bit (or 64-bit-truncated `Sha256`) `block_hash` collision can no longer
    // cause one request to silently reuse another's KV for a *different* prefix.
    // This mirrors the reference `BlockPool`, which stores each block's tokens
    // and compares them on a hit.

    /// Associate a filled block with a prefix `hash` and the `tokens` whose
    /// content produced it, so future requests can reuse it. Emits `BlockStored`.
    /// Idempotent on an already-mapped hash.
    ///
    /// `tokens` are retained for content verification on later hits.
    pub fn cache_block(&mut self, b: BlockId, hash: u64, tokens: &[u32]) {
        if self.hash_to_block.contains_key(&hash) {
            return;
        }
        let m = &mut self.meta[b.0 as usize];
        m.hash = Some(hash);
        m.tokens.clear();
        m.tokens.extend_from_slice(tokens);
        self.hash_to_block.insert(hash, b);
        self.stats.blocks_stored += 1;
        self.push_event(CacheEvent::BlockStored { hash, block: b });
    }

    /// Look up a cached block by prefix `hash`, verifying the candidate `tokens`
    /// match the block's stored tokens (no ref change). a hash hit whose
    /// tokens differ (a digest collision) is treated as a miss (`None`) rather
    /// than silently reusing a foreign prefix's KV.
    pub fn lookup_cached(&self, hash: u64, tokens: &[u32]) -> Option<BlockId> {
        let b = *self.hash_to_block.get(&hash)?;
        if self.meta[b.0 as usize].tokens == tokens {
            Some(b)
        } else {
            None
        }
    }

    /// Reuse a cached block for `req`: bump ref_cnt, remove it from the free queue
    /// if it is sitting there, and (re-)own it. A block that is still `Active`
    /// (held by a concurrently-running request whose prompt was already published
    /// to the prefix cache) is shared by bumping its ref_cnt — that cross-request
    /// sharing is the whole point of prefix caching.
    ///
    /// Returns false (treated as a cache miss) when the block no longer carries
    /// `hash`, when the block's stored tokens do not match the candidate
    /// `tokens` (content verification — a digest collision must not reuse a
    /// foreign prefix's KV), or when `req` already owns this block, since
    /// re-listing the same block id twice in one request would inflate its
    /// ref_cnt without a matching deref on release and silently leak it.
    /// `ref_cnt` is bumped saturatingly so a pathological share count can never
    /// wrap past `u32::MAX`.
    pub fn acquire_cached(
        &mut self,
        req: RequestId,
        b: BlockId,
        hash: u64,
        tokens: &[u32],
    ) -> bool {
        let m = &self.meta[b.0 as usize];
        if m.hash != Some(hash) {
            return false;
        }
        // re-verify the actual block content before reusing it. A 64-bit
        // hash hit on a different prefix (collision) is a miss, not silent reuse.
        if m.tokens != tokens {
            return false;
        }
        // Guard against double-listing the same block within one request.
        if self
            .requests
            .get(&req)
            .is_some_and(|r| r.blocks.contains(&b))
        {
            return false;
        }
        self.fq_unlink(b);
        let m = &mut self.meta[b.0 as usize];
        m.ref_cnt = m.ref_cnt.saturating_add(1);
        m.state = BlockState::Active { owner: req };
        let rec = self.requests.entry(req).or_default();
        rec.blocks.push(b);
        // Growing the block list re-establishes positional layout reasoning;
        // discard any stale per-block trim offsets.
        rec.block_tok_starts = None;
        rec.segments.get_or_insert_with(SegmentTable::default);
        self.stats.free = self.total_free();
        true
    }

    /// Clear the entire prefix cache (the `/reset_prefix_cache` action).
    /// Cached-but-unreferenced blocks lose their hash and become plain free blocks;
    /// still-referenced blocks (Active/Reserved) merely drop their hash so they
    /// return as `Free` rather than `Cached` once dereffed — they are NOT stranded,
    /// because they remain in their owner's block list and free-queue links are
    /// untouched. Each cleared `Cached` block is counted as an eviction so the
    /// eviction counter stays consistent with the allocation-driven eviction path
    /// (which also bumps `evictions` when it drops a cached block's hash).
    pub fn reset_prefix_cache(&mut self) {
        let hashes: Vec<(u64, BlockId)> = self.hash_to_block.drain().collect();
        for (_h, b) in hashes {
            let m = &mut self.meta[b.0 as usize];
            m.hash = None;
            // clear stored content alongside the hash.
            m.tokens = Vec::new();
            if m.state == BlockState::Cached {
                m.state = BlockState::Free;
                self.stats.evictions += 1;
            }
        }
        self.push_event(CacheEvent::AllBlocksCleared);
    }

    // ---- sliding window / sink trimming ----

    /// For each sliding-window group a request uses, release blocks that fall
    /// outside `[0, sink) ∪ [pos-window, pos)` (analogous to the reference's
    /// `remove_skipped_blocks`). No-op for full-attention groups (BAGEL).
    ///
    /// each block's token range is resolved from its *true* start offset,
    /// not its positional index. For a request that is still token-contiguous
    /// from offset 0 (the post-allocate baseline, including one seeded by a
    /// leading prefix-cache hit whose cached blocks are the contiguous leading
    /// prompt blocks) block index `idx` does map to `[idx*block_size,
    /// (idx+1)*block_size)`, so the positional offset is used. But once a trim
    /// has removed interior blocks the surviving blocks are no longer contiguous,
    /// so `idx*block_size` would mis-map; to make repeated (monotonic) trims
    /// correct, the surviving blocks' real offsets are recorded in
    /// `block_tok_starts` and consulted on the next call instead of the index.
    /// Growing the block list afterwards (`allocate`/`grow`/`acquire_cached`)
    /// clears that table, returning the request to the contiguous baseline.
    /// (Today there is no live caller; sliding-window groups are not yet wired
    /// into the scheduler step loop.)
    pub fn trim_sliding_window(&mut self, req: RequestId, pos_tokens: usize) {
        let (blocks, prior_starts) = match self.requests.get(&req) {
            Some(r) => (r.blocks.clone(), r.block_tok_starts.clone()),
            None => return,
        };
        let bs = self.block_size;
        // True token-start of each block: the recorded offsets if this request
        // has already been trimmed (so its block list is no longer contiguous),
        // else positional `idx*block_size` for the contiguous baseline.
        let prior_starts = prior_starts.as_ref();
        let mut keep: Vec<BlockId> = Vec::with_capacity(blocks.len());
        let mut keep_starts: Vec<usize> = Vec::with_capacity(blocks.len());
        let mut trimmed: Vec<BlockId> = Vec::new();
        for (idx, b) in blocks.iter().enumerate() {
            let gid = self.block_group[b.0 as usize] as usize;
            let kind = self.groups[gid].kind;
            let block_tok_start = prior_starts
                .and_then(|s| s.get(idx).copied())
                .unwrap_or(idx * bs);
            let block_tok_end = block_tok_start + bs;
            let keep_it = match kind {
                KvGroupKind::Full => true,
                KvGroupKind::SlidingWindow { window, sink } => {
                    let sink = sink as usize;
                    let window = window as usize;
                    let in_sink = block_tok_start < sink;
                    let window_start = pos_tokens.saturating_sub(window);
                    let in_window = block_tok_end > window_start;
                    in_sink || in_window
                }
            };
            if keep_it {
                keep.push(*b);
                keep_starts.push(block_tok_start);
            } else {
                trimmed.push(*b);
            }
        }
        if trimmed.is_empty() {
            return;
        }
        for b in &trimmed {
            self.deref_block(*b);
        }
        let rec = self.requests.entry(req).or_default();
        rec.blocks = keep;
        // Record the survivors' real offsets so a subsequent trim does not
        // mis-map them via positional index.
        rec.block_tok_starts = Some(keep_starts);
        self.stats.free = self.total_free();
    }

    // ---- scratch budget ----
    pub fn can_reserve_scratch(&self, tokens: u64) -> bool {
        self.scratch_used + tokens <= self.scratch_capacity
    }
    pub fn reserve_scratch(&mut self, req: RequestId, tokens: u64) -> bool {
        if !self.can_reserve_scratch(tokens) {
            return false;
        }
        self.requests.entry(req).or_default().scratch_res += tokens;
        self.scratch_used += tokens;
        true
    }
    pub fn release_scratch(&mut self, req: RequestId) {
        if let Some(rec) = self.requests.get_mut(&req) {
            self.scratch_used -= rec.scratch_res;
            rec.scratch_res = 0;
            if rec.is_empty() {
                self.requests.remove(&req);
            }
        }
    }

    /// Release all of a request's blocks (deref → free queue) and its scratch.
    pub fn release(&mut self, req: RequestId) {
        if let Some(rec) = self.requests.remove(&req) {
            for b in rec.blocks {
                // a request holds exactly one ref per block; deref returns it to
                // the free queue (Cached if it has a hash, else Free).
                self.deref_block(b);
            }
            self.scratch_used -= rec.scratch_res;
        }
        self.stats.free = self.total_free();
    }

    // ---- cache events / stats drain ----

    /// Drain and return the buffered cache events.
    pub fn drain_events(&mut self) -> Vec<CacheEvent> {
        self.events.drain(..).collect()
    }
    pub fn cached_blocks(&self) -> usize {
        self.hash_to_block.len()
    }

    /// Total physical rows the worker declared (`EngineCaps.num_blocks`). NOTE
    /// the host reserves block id 0 as the CUDA-graph padding slot and only ever
    /// hands out ids in `[1, num_blocks)`, so the host-*usable* capacity is
    /// `num_blocks - 1`. The worker's `PagedKVPool` treats all
    /// `num_blocks` rows as usable storage and simply leaves row 0 unwritten; the
    /// two sides therefore agree on the wire scalar but differ on its meaning, so
    /// do not assume this many blocks are allocatable — see `free_blocks`.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::{KvGroupKind, RequestId};

    fn rid(n: u64) -> RequestId {
        RequestId(n)
    }

    #[test]
    fn single_group_allocate_release_roundtrip() {
        let mut bm = BlockManager::new(8, 4, 0);
        assert_eq!(bm.free_blocks(), 7); // block 0 is padding
        let blocks = bm.allocate(rid(1), 3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(bm.free_blocks(), 4);
        for b in &blocks {
            assert_eq!(bm.ref_count(*b), 1);
        }
        bm.release(rid(1));
        assert_eq!(bm.free_blocks(), 7);
    }

    #[test]
    fn lru_reclaim_order() {
        // Freed blocks return to the MRU end; allocation pops the LRU end, so the
        // earliest-freed block is reclaimed first.
        let mut bm = BlockManager::new(5, 4, 0); // blocks 1..5 free
        let a = bm.allocate(rid(1), 1).unwrap()[0];
        let b = bm.allocate(rid(2), 1).unwrap()[0];
        let c = bm.allocate(rid(3), 1).unwrap()[0];
        // free in order a, b, c
        bm.release(rid(1));
        bm.release(rid(2));
        bm.release(rid(3));
        // next allocations should hand back the remaining free first, then a,b,c LRU.
        // After releasing, free queue (MRU end) = [remaining_free..., a, b, c]
        // pop_front order reclaims oldest first.
        let n = bm.free_blocks();
        let reclaimed: Vec<BlockId> = (0..n).map(|_| bm.allocate(rid(9), 1).unwrap()[0]).collect();
        // a was freed before b before c, so among them a comes before b before c.
        let pos = |x: BlockId| reclaimed.iter().position(|y| *y == x);
        assert!(pos(a) < pos(b));
        assert!(pos(b) < pos(c));
    }

    #[test]
    fn ref_counted_block_not_evicted_until_zero() {
        let mut bm = BlockManager::new(4, 4, 0); // blocks 1,2,3
        let blk = bm.allocate(rid(1), 1).unwrap()[0];
        let toks = &[1u32, 2, 3, 4];
        bm.cache_block(blk, 0xABCD, toks);
        // request still holds it (ref_cnt=1): cannot be reused even under pressure.
        // Exhaust the other free blocks.
        let _ = bm.allocate(rid(2), 2).unwrap();
        assert_eq!(bm.free_blocks(), 0);
        assert!(
            bm.allocate(rid(3), 1).is_none(),
            "referenced cached block must not be evicted"
        );
        // release the referenced block -> ref_cnt 0 -> cached & evictable, still mapped.
        bm.release(rid(1));
        assert_eq!(bm.lookup_cached(0xABCD, toks), Some(blk));
        // now allocation reuses it, evicting the hash.
        let got = bm.allocate(rid(3), 1).unwrap()[0];
        assert_eq!(got, blk);
        assert_eq!(bm.lookup_cached(0xABCD, toks), None);
        assert_eq!(bm.stats.evictions, 1);
    }

    #[test]
    fn prefix_reuse_acquire_and_release() {
        let mut bm = BlockManager::new(6, 4, 0);
        let blk = bm.allocate(rid(1), 1).unwrap()[0];
        let toks = &[5u32, 6, 7, 8];
        bm.cache_block(blk, 42, toks);
        bm.release(rid(1)); // ref_cnt 0, cached, in free queue
        // a new request reuses the cached block
        assert_eq!(bm.lookup_cached(42, toks), Some(blk));
        assert!(bm.acquire_cached(rid(2), blk, 42, toks));
        assert_eq!(bm.ref_count(blk), 1);
        // it left the free queue
        assert!(bm.allocate(rid(3), bm.free_blocks()).is_some());
    }

    /// a hash *collision* (same 64-bit hash, different token content) must
    /// NOT reuse the cached block — both `lookup_cached` and `acquire_cached`
    /// re-verify the candidate tokens against the block's stored tokens and
    /// report a miss on mismatch, so a digest collision can never silently serve
    /// one request another's KV for a different prefix.
    #[test]
    fn prefix_lookup_rejects_hash_collision_with_different_tokens() {
        let mut bm = BlockManager::new(6, 4, 0);
        let blk = bm.allocate(rid(1), 1).unwrap()[0];
        let real = &[10u32, 11, 12, 13];
        bm.cache_block(blk, 0xC0FFEE, real);
        bm.release(rid(1)); // cached, evictable

        // Same hash, *different* tokens (the collision): treated as a miss.
        let collider = &[99u32, 98, 97, 96];
        assert_eq!(bm.lookup_cached(0xC0FFEE, collider), None);
        assert!(!bm.acquire_cached(rid(2), blk, 0xC0FFEE, collider));
        assert_eq!(bm.ref_count(blk), 0, "colliding prefix must not re-own it");

        // The genuine prefix (matching tokens) still hits and reuses the block.
        assert_eq!(bm.lookup_cached(0xC0FFEE, real), Some(blk));
        assert!(bm.acquire_cached(rid(3), blk, 0xC0FFEE, real));
        assert_eq!(bm.ref_count(blk), 1);
    }

    #[test]
    fn cache_events_emitted() {
        let mut bm = BlockManager::new(4, 4, 0);
        let blk = bm.allocate(rid(1), 1).unwrap()[0];
        bm.cache_block(blk, 7, &[1, 2, 3, 4]);
        bm.release(rid(1));
        let _ = bm.allocate(rid(2), bm.free_blocks()).unwrap(); // evicts blk's hash
        let events = bm.drain_events();
        assert!(events.contains(&CacheEvent::BlockStored {
            hash: 7,
            block: blk
        }));
        assert!(events.contains(&CacheEvent::BlockRemoved {
            hash: 7,
            block: blk
        }));
    }

    #[test]
    fn reset_prefix_cache_clears_map() {
        let mut bm = BlockManager::new(6, 4, 0);
        let blk = bm.allocate(rid(1), 1).unwrap()[0];
        let toks = &[1u32, 2, 3, 4];
        bm.cache_block(blk, 100, toks);
        bm.release(rid(1));
        assert_eq!(bm.cached_blocks(), 1);
        bm.reset_prefix_cache();
        assert_eq!(bm.cached_blocks(), 0);
        assert_eq!(bm.lookup_cached(100, toks), None);
        assert!(bm.drain_events().contains(&CacheEvent::AllBlocksCleared));
    }

    #[test]
    fn sliding_window_trims_out_of_window_blocks() {
        // group: sliding window of 8 tokens, sink 4 tokens, block_size 4.
        let mut bm = BlockManager::with_groups(
            16,
            4,
            0,
            &[(KvGroupKind::SlidingWindow { window: 8, sink: 4 }, 1, 15)],
        );
        // allocate 5 blocks covering tokens [0,20)
        let _ = bm.allocate(rid(1), 5).unwrap();
        let before = bm.blocks_for(rid(1)).len();
        assert_eq!(before, 5);
        // at pos=20: keep sink block (tokens 0..4) + window blocks covering [12,20)
        bm.trim_sliding_window(rid(1), 20);
        let after = bm.blocks_for(rid(1)).len();
        assert!(after < before, "expected trimming, {after} !< {before}");
        // sink block (idx 0) retained; some middle blocks freed back.
        assert!(bm.free_blocks() > 15 - 5);
    }

    #[test]
    fn block_hash_is_deterministic_and_modality_aware() {
        use uniserve_core::HashAlgo;
        let a = block_hash(0, 0, 0, &[1, 2, 3, 4], HashAlgo::Fnv1a);
        let b = block_hash(0, 0, 0, &[1, 2, 3, 4], HashAlgo::Fnv1a);
        assert_eq!(a, b, "same inputs must hash identically");
        // different tokens -> different hash
        assert_ne!(a, block_hash(0, 0, 0, &[1, 2, 3, 5], HashAlgo::Fnv1a));
        // different modality -> different hash (no text/image collision)
        assert_ne!(a, block_hash(0, 0, 1, &[1, 2, 3, 4], HashAlgo::Fnv1a));
        // chaining: parent affects the result
        assert_ne!(a, block_hash(a, 0, 0, &[1, 2, 3, 4], HashAlgo::Fnv1a));
        // token count is folded in, so a prefix of the same tokens with a
        // different length cannot alias the full block.
        assert_ne!(
            block_hash(0, 0, 0, &[1, 2, 3], HashAlgo::Fnv1a),
            block_hash(0, 0, 0, &[1, 2, 3, 4], HashAlgo::Fnv1a)
        );
        assert_ne!(
            block_hash(0, 0, 0, &[1, 2, 3], HashAlgo::Sha256),
            block_hash(0, 0, 0, &[1, 2, 3, 4], HashAlgo::Sha256)
        );
        // sha256 path is also deterministic
        let s = block_hash(0, 0, 0, &[1, 2, 3, 4], HashAlgo::Sha256);
        assert_eq!(s, block_hash(0, 0, 0, &[1, 2, 3, 4], HashAlgo::Sha256));
        assert_ne!(s, a);
    }

    #[test]
    fn acquire_cached_shares_across_requests_but_refuses_self_double_list() {
        // a still-Active cached block may be SHARED by another request
        // (ref_cnt bumps — the point of prefix caching), but re-acquiring it for a
        // request that already holds it is refused so its ref_cnt cannot drift.
        let mut bm = BlockManager::new(6, 4, 0);
        let blk = bm.allocate(rid(1), 1).unwrap()[0];
        let toks = &[9u32, 8, 7, 6];
        bm.cache_block(blk, 77, toks);
        // request 1 still owns it (Active, ref_cnt 1); request 2 shares it.
        assert!(bm.acquire_cached(rid(2), blk, 77, toks));
        assert_eq!(bm.ref_count(blk), 2);
        assert_eq!(bm.blocks_for(rid(2)), [blk].as_slice());
        // a second acquire by request 2 is refused (no self double-list).
        assert!(!bm.acquire_cached(rid(2), blk, 77, toks));
        assert_eq!(bm.ref_count(blk), 2);
        assert_eq!(bm.blocks_for(rid(2)).len(), 1);
        // wrong hash is a miss.
        assert!(!bm.acquire_cached(rid(3), blk, 0xDEAD, toks));
        // a matching hash but mismatched tokens (a digest collision) is
        // also a miss — content verification refuses the foreign prefix.
        assert!(!bm.acquire_cached(rid(3), blk, 77, &[0, 0, 0, 0]));
        // releasing both owners returns it to the cache (still mapped).
        bm.release(rid(1));
        bm.release(rid(2));
        assert_eq!(bm.ref_count(blk), 0);
        assert_eq!(bm.lookup_cached(77, toks), Some(blk));
    }

    #[test]
    fn reset_prefix_cache_counts_evictions() {
        // clearing the cache counts each reclaimed cached block as an
        // eviction, matching the allocation-driven eviction path.
        let mut bm = BlockManager::new(8, 4, 0);
        let a = bm.allocate(rid(1), 1).unwrap()[0];
        let b = bm.allocate(rid(2), 1).unwrap()[0];
        bm.cache_block(a, 1, &[1, 1, 1, 1]);
        bm.cache_block(b, 2, &[2, 2, 2, 2]);
        bm.release(rid(1));
        bm.release(rid(2)); // both now Cached in the free queue
        assert_eq!(bm.stats.evictions, 0);
        bm.reset_prefix_cache();
        assert_eq!(bm.stats.evictions, 2);
        assert_eq!(bm.cached_blocks(), 0);
        // referenced block's hash is dropped but it is not stranded.
        let c = bm.allocate(rid(3), 1).unwrap()[0];
        bm.cache_block(c, 3, &[3, 3, 3, 3]);
        bm.reset_prefix_cache(); // c is Active, not counted, still owned
        assert_eq!(bm.stats.evictions, 2);
        assert_eq!(bm.ref_count(c), 1);
    }

    #[test]
    fn validate_group_specs_accepts_full_partition_and_rejects_malformed() {
        use uniserve_core::KvGroupKind::Full;
        // BAGEL default: one full group covering [1, num_blocks).
        assert!(BlockManager::validate_group_specs(8, &[(Full, 1, 7)]).is_ok());
        // two disjoint groups exactly partitioning [1, 8).
        assert!(BlockManager::validate_group_specs(8, &[(Full, 1, 3), (Full, 4, 4)]).is_ok());
        // first_block 0 collides with the padding slot.
        assert!(BlockManager::validate_group_specs(8, &[(Full, 0, 7)]).is_err());
        // range runs past num_blocks (the OOB-panic case, now rejected).
        assert!(BlockManager::validate_group_specs(8, &[(Full, 1, 20)]).is_err());
        // overlapping ranges (would double-list / corrupt free_count).
        assert!(BlockManager::validate_group_specs(8, &[(Full, 1, 4), (Full, 3, 4)]).is_err());
        // a gap leaves block 7 orphaned.
        assert!(BlockManager::validate_group_specs(8, &[(Full, 1, 6)]).is_err());
        // first+count overflow is caught instead of wrapping.
        assert!(BlockManager::validate_group_specs(8, &[(Full, u32::MAX, 2)]).is_err());
    }

    #[test]
    #[should_panic(expected = "invalid KV-cache group specs")]
    fn with_groups_panics_on_out_of_range_spec() {
        let _ = BlockManager::with_groups(
            8,
            4,
            0,
            &[(KvGroupKind::Full, 1, 20)], // exceeds num_blocks
        );
    }

    #[test]
    fn padding_block_zero_never_allocated() {
        // block 0 is the padding slot; usable capacity is num_blocks-1 and
        // allocation never returns id 0 even when the pool is fully drained.
        let mut bm = BlockManager::new(4, 4, 0); // ids 1,2,3 usable
        assert_eq!(bm.free_blocks(), 3);
        let all = bm.allocate(rid(1), 3).unwrap();
        assert!(all.iter().all(|b| b.0 != 0));
        assert_eq!(bm.free_blocks(), 0);
        assert_eq!(bm.padding_block(), BlockId(0));
    }

    #[test]
    fn repeated_trim_uses_true_offsets_not_positional_index() {
        // after the first trim removes interior blocks, the survivors are no
        // longer token-contiguous from 0, so a positional `idx*block_size` mapping
        // would mis-identify which blocks fall inside the window on the next trim.
        // window 8, sink 4, block_size 4.
        let mut bm = BlockManager::with_groups(
            16,
            4,
            0,
            &[(KvGroupKind::SlidingWindow { window: 8, sink: 4 }, 1, 15)],
        );
        // 5 blocks covering tokens [0,4),[4,8),[8,12),[12,16),[16,20).
        let blks = bm.allocate(rid(1), 5).unwrap();
        // first trim at pos=20: keep sink block (idx0) + window blocks whose end
        // exceeds window_start=12 -> idx3 (tokens [12,16)) and idx4 (tokens [16,20)).
        bm.trim_sliding_window(rid(1), 20);
        assert_eq!(
            bm.blocks_for(rid(1)),
            [blks[0], blks[3], blks[4]].as_slice(),
            "first trim keeps sink + in-window blocks"
        );
        // second trim at pos=24: window_start=16. Using TRUE offsets, blk3 (tokens
        // [12,16)) leaves the window and is trimmed, while blk4 (tokens [16,20)) is
        // still inside it and must be kept. A buggy positional mapping (survivor
        // indices 0,1,2 -> starts 0,4,8) would instead drop blk4 too.
        bm.trim_sliding_window(rid(1), 24);
        assert_eq!(
            bm.blocks_for(rid(1)),
            [blks[0], blks[4]].as_slice(),
            "second trim must keep blk4 (in window) and drop blk3 (out of window)"
        );
        // blk3 was returned to the free queue; blk4 (idx4) is still owned.
        assert_eq!(bm.ref_count(blks[4]), 1);
        assert_eq!(bm.ref_count(blks[3]), 0);
    }

    #[test]
    fn full_group_never_trims() {
        let mut bm = BlockManager::new(16, 4, 0);
        let _ = bm.allocate(rid(1), 5).unwrap();
        bm.trim_sliding_window(rid(1), 20);
        assert_eq!(bm.blocks_for(rid(1)).len(), 5);
    }
}
