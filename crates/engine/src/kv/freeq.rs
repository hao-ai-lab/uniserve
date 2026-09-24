//! Per-page cache metadata and intrusive free-queue links.
//!
//! One [`BlockMeta`] exists for every physical KV page, indexed by page id in
//! the block pool's metadata vector. The free queues themselves (head, tail,
//! and free count per cache group) live in the parent module's pool state;
//! this module only holds the per-page links, which store page indices rather
//! than pointers.

use crate::kv::BlockState;
use std::sync::Arc;
use uniserve_worker_ipc::WorkerEndpoint;

/// Mutable ownership and cache metadata for one physical KV page.
pub(crate) struct BlockMeta {
    pub state: BlockState,
    /// Number of live page references (`CacheBlockRef` handles).
    ///
    /// A page enters its group's free queue when this count drops to zero.
    /// Page zero is the worker's padding sentinel: it keeps a zero count but
    /// is never enqueued or allocated.
    pub ref_cnt: u32,
    /// Prefix-cache hash this block holds, if any. Cleared on eviction.
    pub hash: Option<u64>,
    /// Loaded Worker whose rank group retains this cached page.
    ///
    /// Set and cleared together with `hash`.
    pub source: Option<Arc<WorkerEndpoint>>,
    /// Exact token content used to disambiguate equal prefix hashes.
    ///
    /// The value is empty unless the page is published under `hash`, and it
    /// is cleared on eviction.
    pub tokens: Vec<u32>,
    /// Intrusive queue links, valid only while `in_fq` is set.
    pub fq_prev: Option<u32>,
    pub fq_next: Option<u32>,
    pub in_fq: bool,
}

impl BlockMeta {
    /// Creates metadata for an unreferenced, unpublished page outside any
    /// free queue.
    pub(super) fn new(state: BlockState) -> Self {
        Self {
            state,
            ref_cnt: 0,
            hash: None,
            source: None,
            tokens: Vec::new(),
            fq_prev: None,
            fq_next: None,
            in_fq: false,
        }
    }
}
