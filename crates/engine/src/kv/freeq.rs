//! Per-unit ownership metadata and intrusive free-queue links.
//!
//! One [`UnitMeta`] exists for every physical KV unit, indexed by unit id in
//! the block pool's metadata vector. The free queue itself (head, tail, and
//! free count) lives in the parent module's pool state; this module only
//! holds the per-unit links, which store unit indices rather than pointers.

use crate::kv::UnitState;

/// Mutable ownership metadata for one physical KV unit.
pub(crate) struct UnitMeta {
    pub state: UnitState,
    /// Number of live page references (`PageRef` handles) holding the unit.
    ///
    /// Every unit of one logical page shares its page's references, so the
    /// units of a page reach zero together and enter the free queue as one
    /// consecutive run. Unit zero is the worker's padding sentinel: it keeps
    /// a zero count but is never enqueued or allocated.
    pub ref_cnt: u32,
    /// First unit of the published prefix page this unit belongs to, which
    /// keys the page's identity in the pool; `None` for an unpublished unit.
    pub published: Option<u32>,
    /// Intrusive queue links, valid only while `in_fq` is set.
    pub fq_prev: Option<u32>,
    pub fq_next: Option<u32>,
    pub in_fq: bool,
}

impl UnitMeta {
    /// Creates metadata for an unreferenced, unpublished unit outside the
    /// free queue.
    pub(super) fn new() -> Self {
        Self {
            state: UnitState::Free,
            ref_cnt: 0,
            published: None,
            fq_prev: None,
            fq_next: None,
            in_fq: false,
        }
    }
}
