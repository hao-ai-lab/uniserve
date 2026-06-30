//! Per-block metadata including the intrusive free/eviction-queue links.
//!
//! The free queue is a doubly-linked list threaded through these `prev`/`next`
//! indices (block-id space), giving O(1) push-back, pop-front, and middle removal
//! — the mechanics of the reference `FreeKVCacheBlockQueue` (`block_pool.py`), needed so
//! a cached block can be pulled out of the LRU queue the moment a prefix lookup
//! re-references it.

use crate::BlockState;

pub(crate) struct BlockMeta {
    pub state: BlockState,
    pub ref_cnt: u32,
    /// Prefix-cache hash this block holds, if any. Cleared on eviction.
    pub hash: Option<u64>,
    /// Collision safety: the exact token ids whose content produced
    /// `hash`. Retained alongside the hash so `lookup_cached`/`acquire_cached`
    /// can re-verify a candidate prefix's tokens against the block on a hash
    /// hit — a 64-bit (or 64-bit-truncated) digest collision would otherwise let
    /// one request silently reuse another's KV for a *different* prefix. Mirrors
    /// the reference `BlockPool`, which stores the block's tokens and compares
    /// them on a hit. Empty when the block is not cached; cleared on eviction.
    pub tokens: Vec<u32>,
    /// Free-queue links (valid only while `in_fq`); `None` == none.
    pub fq_prev: Option<u32>,
    pub fq_next: Option<u32>,
    pub in_fq: bool,
}

impl BlockMeta {
    pub fn new(state: BlockState) -> Self {
        Self {
            state,
            ref_cnt: 0,
            hash: None,
            tokens: Vec::new(),
            fq_prev: None,
            fq_next: None,
            in_fq: false,
        }
    }
}
