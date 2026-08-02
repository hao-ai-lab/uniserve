//! Authoritative scheduler credit accounting.
//!
//! One request owns one vector. Admission and operation registration extend
//! that vector atomically, and reclamation subtracts the exact reservation.
//! Physical managers consume only work for which this ledger already holds
//! credit.

use std::collections::HashMap;

use uniserve_core::RequestId;
use uniserve_worker_wire::{CreditDimension, CreditVector};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LedgerStats {
    pub acquisitions: u64,
    pub releases: u64,
    pub would_block: u64,
    pub invariant_violations: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreditExhausted {
    pub dimension: CreditDimension,
    pub requested: u64,
    pub available: u64,
}

/// Atomic finite-credit owner table.
pub struct CreditLedger {
    capacity: CreditVector,
    used: CreditVector,
    active: HashMap<RequestId, CreditVector>,
    pub stats: LedgerStats,
}

impl CreditLedger {
    pub fn new(capacity: CreditVector) -> Self {
        Self {
            capacity,
            used: CreditVector::ZERO,
            active: HashMap::new(),
            stats: LedgerStats::default(),
        }
    }

    pub fn capacity(&self) -> CreditVector {
        self.capacity
    }

    pub fn used(&self) -> CreditVector {
        self.used
    }

    pub fn available(&self) -> CreditVector {
        self.capacity
            .checked_sub(self.used)
            .expect("credit usage is bounded by capacity")
    }

    pub fn active_for(&self, request: RequestId) -> CreditVector {
        self.active
            .get(&request)
            .copied()
            .unwrap_or(CreditVector::ZERO)
    }

    pub fn active_requests(&self) -> usize {
        self.active.len()
    }

    pub fn can_acquire(
        &self,
        request: RequestId,
        delta: CreditVector,
        request_limit: CreditVector,
    ) -> Result<(), CreditExhausted> {
        let current = self.active_for(request);
        if let Some(dimension) = request_limit.first_exhausted(current, delta) {
            return Err(CreditExhausted {
                dimension,
                requested: delta.get(dimension),
                available: request_limit
                    .get(dimension)
                    .saturating_sub(current.get(dimension)),
            });
        }
        if let Some(dimension) = self.capacity.first_exhausted(self.used, delta) {
            return Err(CreditExhausted {
                dimension,
                requested: delta.get(dimension),
                available: self
                    .capacity
                    .get(dimension)
                    .saturating_sub(self.used.get(dimension)),
            });
        }
        Ok(())
    }

    /// Atomically extend one request reservation within both its route maximum
    /// and the worker-wide capacity.
    pub fn try_acquire(
        &mut self,
        request: RequestId,
        delta: CreditVector,
        request_limit: CreditVector,
    ) -> Result<(), CreditExhausted> {
        let current = self.active_for(request);
        let Some(projected_request) = current.checked_add(delta) else {
            self.stats.invariant_violations += 1;
            return Err(CreditExhausted {
                dimension: CreditDimension::RegisteredOperations,
                requested: u64::MAX,
                available: 0,
            });
        };
        if let Err(error) = self.can_acquire(request, delta, request_limit) {
            self.stats.would_block += 1;
            return Err(error);
        }
        let projected_used = self
            .used
            .checked_add(delta)
            .expect("capacity preflight proves credit addition cannot overflow");
        if projected_request.is_zero() {
            self.active.remove(&request);
        } else {
            self.active.insert(request, projected_request);
        }
        self.used = projected_used;
        self.stats.acquisitions += 1;
        Ok(())
    }

    /// Release an exact sub-vector. A mismatched release is an accounting
    /// invariant violation and leaves state unchanged.
    pub fn release(&mut self, request: RequestId, delta: CreditVector) -> bool {
        let current = self.active_for(request);
        let Some(projected_request) = current.checked_sub(delta) else {
            self.stats.invariant_violations += 1;
            return false;
        };
        let Some(projected_used) = self.used.checked_sub(delta) else {
            self.stats.invariant_violations += 1;
            return false;
        };
        if projected_request.is_zero() {
            self.active.remove(&request);
        } else {
            self.active.insert(request, projected_request);
        }
        self.used = projected_used;
        self.stats.releases += 1;
        true
    }

    /// Reclaim the complete request reservation after every logical successor
    /// and physical release control has been issued.
    pub fn release_request(&mut self, request: RequestId) -> CreditVector {
        let Some(reservation) = self.active.remove(&request) else {
            return CreditVector::ZERO;
        };
        match self.used.checked_sub(reservation) {
            Some(used) => {
                self.used = used;
                self.stats.releases += 1;
                reservation
            }
            None => {
                self.stats.invariant_violations += 1;
                CreditVector::ZERO
            }
        }
    }

    pub fn assert_released(&mut self, request: RequestId) -> bool {
        if self.active_for(request).is_zero() {
            true
        } else {
            self.stats.invariant_violations += 1;
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(operations: u64, bytes: u64) -> CreditVector {
        CreditVector {
            registered_operations: operations,
            execution_slots: operations,
            completion_slots: operations,
            output_journal_bytes: bytes,
            ..CreditVector::ZERO
        }
    }

    #[test]
    fn capacity_failure_has_no_partial_side_effect() {
        let mut ledger = CreditLedger::new(vector(2, 1024));
        let request = RequestId(1);
        ledger
            .try_acquire(request, vector(1, 768), vector(2, 1024))
            .unwrap();
        let before_request = ledger.active_for(request);
        let before_used = ledger.used();
        let error = ledger
            .try_acquire(request, vector(1, 512), vector(2, 1024))
            .unwrap_err();
        assert_eq!(error.dimension, CreditDimension::OutputJournalBytes);
        assert_eq!(ledger.active_for(request), before_request);
        assert_eq!(ledger.used(), before_used);
    }

    #[test]
    fn request_limit_is_enforced_before_worker_capacity() {
        let mut ledger = CreditLedger::new(vector(8, 8192));
        let request = RequestId(7);
        ledger
            .try_acquire(request, vector(1, 512), vector(1, 1024))
            .unwrap();
        let error = ledger
            .try_acquire(request, vector(1, 1), vector(1, 1024))
            .unwrap_err();
        assert_eq!(error.dimension, CreditDimension::RegisteredOperations);
        assert_eq!(ledger.used(), vector(1, 512));
    }

    #[test]
    fn exact_release_and_request_reclamation_drain_to_zero() {
        let mut ledger = CreditLedger::new(vector(8, 8192));
        let first = RequestId(1);
        let second = RequestId(2);
        ledger
            .try_acquire(first, vector(2, 1024), vector(4, 4096))
            .unwrap();
        ledger
            .try_acquire(second, vector(1, 512), vector(4, 4096))
            .unwrap();
        assert!(ledger.release(first, vector(1, 256)));
        assert_eq!(ledger.release_request(first), vector(1, 768));
        assert_eq!(ledger.release_request(second), vector(1, 512));
        assert!(ledger.used().is_zero());
        assert_eq!(ledger.active_requests(), 0);
    }
}
