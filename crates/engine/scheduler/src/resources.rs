//! Host-side resource accounting.
//!
//! A parallel, **observe-only** ledger: it issues a typed [`ResourceLease`] for
//! every per-request resource the scheduler acquires (KV residency, denoise
//! image latents, denoise scratch) and proves every lease is released after the
//! owning request completes / is dropped / is preempted. It does NOT allocate or
//! change policy — `BlockManager` / `EncoderCacheManager` / the scratch budget
//! stay authoritative; the ledger asserts their bookkeeping is consistent so a
//! resource leak is detected immediately rather than as slow GPU-memory growth.

use std::collections::{HashMap, VecDeque};

use uniserve_core::RequestId;
use uniserve_worker_wire::{
    LeasePolicy, ResourceClass, ResourceEvent, ResourceEventKind, ResourceHandle, ResourceLease,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct LedgerStats {
    pub issued: u64,
    pub released: u64,
    /// Leases still active when a request left `running` (released defensively).
    pub leaked: u64,
    pub invariant_violations: u64,
}

/// Per-request resource leases + lifecycle invariants.
pub struct ResourceLedger {
    active: HashMap<RequestId, Vec<ResourceLease>>,
    next_id: u64,
    pub stats: LedgerStats,
    events: VecDeque<ResourceEvent>,
    max_events: usize,
}

impl Default for ResourceLedger {
    fn default() -> Self {
        Self {
            active: HashMap::new(),
            next_id: 1,
            stats: LedgerStats::default(),
            events: VecDeque::new(),
            max_events: 4096,
        }
    }
}

impl ResourceLedger {
    pub fn new() -> Self {
        Self::default()
    }

    fn record(
        &mut self,
        kind: ResourceEventKind,
        class: ResourceClass,
        owner: RequestId,
        cap: u64,
    ) {
        if self.events.len() >= self.max_events {
            self.events.pop_front();
        }
        self.events.push_back(ResourceEvent {
            kind,
            class,
            owner_request: owner,
            capacity: cap,
        });
    }

    /// Issue a lease for `capacity` units of `class` to `req`; returns the handle.
    pub fn issue(
        &mut self,
        req: RequestId,
        class: ResourceClass,
        capacity: u64,
        policy: LeasePolicy,
    ) -> ResourceHandle {
        let handle = ResourceHandle {
            class,
            id: self.next_id,
        };
        self.next_id += 1;
        self.active.entry(req).or_default().push(ResourceLease {
            handle,
            owner_request: req,
            capacity,
            policy,
        });
        self.stats.issued += 1;
        self.record(ResourceEventKind::Issued, class, req, capacity);
        handle
    }

    /// Release every lease of one class held by `req` (e.g. image latents +
    /// scratch when a generated image commits). Returns the count released.
    pub fn release_class(&mut self, req: RequestId, class: ResourceClass) -> usize {
        let Some(leases) = self.active.get_mut(&req) else {
            return 0;
        };
        let mut released = 0;
        leases.retain(|l| {
            if l.handle.class == class {
                released += 1;
                false
            } else {
                true
            }
        });
        if leases.is_empty() {
            self.active.remove(&req);
        }
        self.stats.released += released as u64;
        for _ in 0..released {
            self.record(ResourceEventKind::Released, class, req, 0);
        }
        released
    }

    /// Release every lease held by `req` (completion / drop / preemption).
    /// Returns the count released; counts any still-held lease as a soft leak so
    /// missing intermediate releases are visible without masking the invariant.
    pub fn release_request(&mut self, req: RequestId) -> usize {
        let Some(leases) = self.active.remove(&req) else {
            return 0;
        };
        let n = leases.len();
        for l in &leases {
            self.record(ResourceEventKind::Released, l.handle.class, req, l.capacity);
        }
        self.stats.released += n as u64;
        n
    }

    /// Active lease count for a request (0 == fully released).
    pub fn active_for(&self, req: RequestId) -> usize {
        self.active.get(&req).map(|v| v.len()).unwrap_or(0)
    }

    /// Total active leases across all requests (0 when the engine is idle).
    pub fn total_active(&self) -> usize {
        self.active.values().map(|v| v.len()).sum()
    }

    /// Assert `req` holds no leases (call AFTER `release_request`). A violation is
    /// counted + logged, never panicked, so the live serving loop is unaffected.
    pub fn assert_released(&mut self, req: RequestId) -> bool {
        let held = self.active_for(req);
        if held != 0 {
            self.stats.invariant_violations += 1;
            self.stats.leaked += held as u64;
            self.record(
                ResourceEventKind::InvariantViolation,
                ResourceClass::KvBlock,
                req,
                held as u64,
            );
            tracing::warn!(
                request = req.0,
                held,
                "resource invariant: request retained leases after release"
            );
            false
        } else {
            true
        }
    }

    pub fn drain_events(&mut self) -> Vec<ResourceEvent> {
        self.events.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_release_roundtrip_no_leak() {
        let mut l = ResourceLedger::new();
        let r = RequestId(1);
        l.issue(r, ResourceClass::KvBlock, 256, LeasePolicy::PerRequest);
        l.issue(r, ResourceClass::ImageLatent, 64, LeasePolicy::Pinned);
        l.issue(r, ResourceClass::Scratch, 64, LeasePolicy::PerRequest);
        assert_eq!(l.active_for(r), 3);
        assert_eq!(l.total_active(), 3);
        // commit releases image leases; KV persists.
        assert_eq!(l.release_class(r, ResourceClass::ImageLatent), 1);
        assert_eq!(l.release_class(r, ResourceClass::Scratch), 1);
        assert_eq!(l.active_for(r), 1);
        // finish releases the rest.
        assert_eq!(l.release_request(r), 1);
        assert_eq!(l.active_for(r), 0);
        assert!(l.assert_released(r));
        assert_eq!(l.stats.issued, 3);
        assert_eq!(l.stats.released, 3);
        assert_eq!(l.stats.invariant_violations, 0);
    }

    #[test]
    fn release_request_clears_everything_idempotently() {
        let mut l = ResourceLedger::new();
        let r = RequestId(2);
        l.issue(r, ResourceClass::KvBlock, 256, LeasePolicy::PerRequest);
        l.issue(r, ResourceClass::ImageLatent, 64, LeasePolicy::Pinned);
        assert_eq!(l.release_request(r), 2); // drop/preempt releases all
        assert!(l.assert_released(r));
        assert_eq!(l.release_request(r), 0); // idempotent
        assert_eq!(l.total_active(), 0);
    }

    #[test]
    fn many_requests_idle_to_zero() {
        let mut l = ResourceLedger::new();
        for i in 1..=10 {
            let r = RequestId(i);
            l.issue(r, ResourceClass::KvBlock, 256, LeasePolicy::PerRequest);
        }
        assert_eq!(l.total_active(), 10);
        for i in 1..=10 {
            l.release_request(RequestId(i));
        }
        assert_eq!(l.total_active(), 0, "ledger must drain to zero when idle");
    }
}
