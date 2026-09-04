//! Per-page cache metadata and intrusive free-queue links.

use crate::kv::BlockState;

/// Mutable ownership and cache metadata for one physical KV page.
pub(crate) struct BlockMeta {
    pub state: BlockState,
    pub ref_cnt: u32,
    /// Prefix-cache hash this block holds, if any. Cleared on eviction.
    pub hash: Option<u64>,
    /// Exact token content used to disambiguate equal prefix hashes.
    ///
    /// The value is empty outside the cached state and is cleared on eviction.
    pub tokens: Vec<u32>,
    /// Intrusive queue links, valid only while `in_fq` is set.
    pub fq_prev: Option<u32>,
    pub fq_next: Option<u32>,
    pub in_fq: bool,
}

impl BlockMeta {
    /// Creates an empty free-block queue.
    pub(super) fn new(state: BlockState) -> Self {
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
