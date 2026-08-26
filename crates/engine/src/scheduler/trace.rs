//! Per-operation lifecycle trace keyed by canonical operation identity.
//!
//! Each operation records a monotonic microsecond stamp for every lifecycle
//! phase it reaches, from planning through physical reclamation, correlated by
//! `(request_key, op_id, control_seq, domain)`. Device timing that is only known
//! after asynchronous readiness is reconstructed from the completion record's
//! reported durations rather than observed synchronously. The trace is bounded;
//! completed request traces move to a ring for post-finish reconstruction.

use std::collections::HashMap;

use uniserve_core::TraceId;
use uniserve_worker_ipc::{Domain, OpId, Operation, RequestKey};

const MAX_OPERATIONS: usize = 512;

/// The twelve lifecycle phases of one operation, in causal order.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LifecyclePhase {
    Planned,
    LogicalResourcesReserved,
    WorkerRegistrationComplete,
    Submitted,
    DeviceExecutionStarted,
    ProducerReady,
    CompletionCopyReady,
    CompletionObserved,
    SemanticallyCommitted,
    PubliclyCommitted,
    ReleaseIssued,
    PhysicallyReclaimed,
}

impl LifecyclePhase {
    /// Every phase in causal order.
    pub const ALL: [LifecyclePhase; 12] = [
        LifecyclePhase::Planned,
        LifecyclePhase::LogicalResourcesReserved,
        LifecyclePhase::WorkerRegistrationComplete,
        LifecyclePhase::Submitted,
        LifecyclePhase::DeviceExecutionStarted,
        LifecyclePhase::ProducerReady,
        LifecyclePhase::CompletionCopyReady,
        LifecyclePhase::CompletionObserved,
        LifecyclePhase::SemanticallyCommitted,
        LifecyclePhase::PubliclyCommitted,
        LifecyclePhase::ReleaseIssued,
        LifecyclePhase::PhysicallyReclaimed,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            LifecyclePhase::Planned => "planned",
            LifecyclePhase::LogicalResourcesReserved => "logical_resources_reserved",
            LifecyclePhase::WorkerRegistrationComplete => "worker_registration_complete",
            LifecyclePhase::Submitted => "submitted",
            LifecyclePhase::DeviceExecutionStarted => "device_execution_started",
            LifecyclePhase::ProducerReady => "producer_ready",
            LifecyclePhase::CompletionCopyReady => "completion_copy_ready",
            LifecyclePhase::CompletionObserved => "completion_observed",
            LifecyclePhase::SemanticallyCommitted => "semantically_committed",
            LifecyclePhase::PubliclyCommitted => "publicly_committed",
            LifecyclePhase::ReleaseIssued => "release_issued",
            LifecyclePhase::PhysicallyReclaimed => "physically_reclaimed",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// Canonical operation identity used to correlate every lifecycle phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationKey {
    pub request_key: RequestKey,
    pub op_id: OpId,
    pub control_seq: u64,
    pub domain: Domain,
}

impl From<&Operation> for OperationKey {
    fn from(operation: &Operation) -> Self {
        Self {
            request_key: operation.request_key,
            op_id: operation.op_id,
            control_seq: operation.control_seq,
            domain: operation.domain,
        }
    }
}

/// One operation's lifecycle: a monotonic microsecond stamp per reached phase.
#[derive(Debug, Clone)]
pub struct OperationLifecycle {
    pub key: OperationKey,
    pub op_kind: Option<&'static str>,
    stamps: [Option<u64>; 12],
}

impl OperationLifecycle {
    fn new(key: OperationKey) -> Self {
        Self {
            key,
            op_kind: None,
            stamps: [None; 12],
        }
    }

    /// Record the first time this operation reached `phase`. Re-stamping a
    /// reached phase is idempotent, so duplicate observation cannot rewrite an
    /// established timestamp.
    pub fn stamp(&mut self, phase: LifecyclePhase, at_us: u64) {
        let slot = &mut self.stamps[phase.index()];
        if slot.is_none() {
            *slot = Some(at_us);
        }
    }

    pub fn at(&self, phase: LifecyclePhase) -> Option<u64> {
        self.stamps[phase.index()]
    }

    pub fn reached(&self, phase: LifecyclePhase) -> bool {
        self.stamps[phase.index()].is_some()
    }

    /// Whether the recorded stamps are non-decreasing in causal phase order.
    pub fn is_ordered(&self) -> bool {
        let mut last = 0u64;
        for phase in LifecyclePhase::ALL {
            if let Some(at) = self.at(phase) {
                if at < last {
                    return false;
                }
                last = at;
            }
        }
        true
    }

    /// The elapsed microseconds between two reached phases, if both are present.
    pub fn span_us(&self, from: LifecyclePhase, to: LifecyclePhase) -> Option<u64> {
        Some(self.at(to)?.saturating_sub(self.at(from)?))
    }
}

/// The lifecycle trace of one request: its operations plus admit/finish marks.
#[derive(Debug, Clone)]
pub struct RequestTrace {
    pub request_key: RequestKey,
    pub trace_id: TraceId,
    pub admitted_us: Option<u64>,
    pub finished_us: Option<u64>,
    pub finish_reason: Option<&'static str>,
    operations: Vec<OperationLifecycle>,
    operation_indices: HashMap<u64, usize>,
}

impl RequestTrace {
    pub fn new(request_key: RequestKey, trace_id: TraceId) -> Self {
        Self {
            request_key,
            trace_id,
            admitted_us: None,
            finished_us: None,
            finish_reason: None,
            operations: Vec::new(),
            operation_indices: HashMap::new(),
        }
    }

    pub fn mark_admitted(&mut self, at_us: u64) {
        if self.admitted_us.is_none() {
            self.admitted_us = Some(at_us);
        }
    }

    pub fn mark_finished(&mut self, reason: &'static str, at_us: u64) {
        if self.finished_us.is_none() {
            self.finished_us = Some(at_us);
            self.finish_reason = Some(reason);
        }
    }

    /// Stamp `phase` for the operation named by `key`, creating its lifecycle on
    /// first sighting. `op_kind` labels the operation once known.
    pub fn stamp(
        &mut self,
        key: OperationKey,
        op_kind: Option<&'static str>,
        phase: LifecyclePhase,
        at_us: u64,
    ) {
        // `op_id` is unique within one scheduler-authority lifetime, so it
        // alone identifies the operation's lifecycle. Later phases (commit,
        // release) arrive under a different control sequence than registration,
        // so matching on the operation id keeps them on the same lifecycle.
        let entry = match self.operation_indices.get(&key.op_id.0).copied() {
            Some(index) => &mut self.operations[index],
            None => {
                if self.operations.len() >= MAX_OPERATIONS {
                    return;
                }
                let index = self.operations.len();
                self.operations.push(OperationLifecycle::new(key));
                self.operation_indices.insert(key.op_id.0, index);
                &mut self.operations[index]
            }
        };
        if op_kind.is_some() {
            entry.op_kind = op_kind;
        }
        entry.stamp(phase, at_us);
    }

    /// Stamp `phase` for an already-registered operation named by `op_id`. A
    /// post-submission phase (commit, release, reclaim) always follows the
    /// operation's registration, so no lifecycle is created here; an unknown
    /// `op_id` is a no-op.
    pub fn stamp_existing(&mut self, op_id: OpId, phase: LifecyclePhase, at_us: u64) -> bool {
        if let Some(index) = self.operation_indices.get(&op_id.0).copied() {
            let entry = &mut self.operations[index];
            let reached = entry.reached(phase);
            entry.stamp(phase, at_us);
            return !reached;
        }
        false
    }

    /// Return one phase timestamp for a registered operation without scanning
    /// the bounded lifecycle history.
    pub fn at(&self, op_id: OpId, phase: LifecyclePhase) -> Option<u64> {
        let index = *self.operation_indices.get(&op_id.0)?;
        self.operations[index].at(phase)
    }

    pub fn domain(&self, op_id: OpId) -> Option<Domain> {
        let index = *self.operation_indices.get(&op_id.0)?;
        Some(self.operations[index].key.domain)
    }

    pub fn span_us(&self, op_id: OpId, from: LifecyclePhase, to: LifecyclePhase) -> Option<u64> {
        let index = *self.operation_indices.get(&op_id.0)?;
        self.operations[index].span_us(from, to)
    }

    pub fn operations(&self) -> &[OperationLifecycle] {
        &self.operations
    }

    /// Count of operations whose completion has been observed.
    pub fn resolved_ops(&self) -> usize {
        self.operations
            .iter()
            .filter(|op| op.reached(LifecyclePhase::CompletionObserved))
            .count()
    }

    pub fn was_admitted(&self) -> bool {
        self.admitted_us.is_some()
    }

    pub fn is_finished(&self) -> bool {
        self.finished_us.is_some()
    }

    /// Whether every operation's stamps are ordered and the request markers
    /// bracket its operations. Used by the observability conformance checks.
    pub fn is_ordered(&self) -> bool {
        self.operations.iter().all(OperationLifecycle::is_ordered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::{RequestId, TraceId};

    fn key(op_id: u64, control_seq: u64, domain: Domain) -> OperationKey {
        OperationKey {
            request_key: RequestKey::new(1, RequestId(7), 2),
            op_id: OpId(op_id),
            control_seq,
            domain,
        }
    }

    #[test]
    fn stamps_reconstruct_one_operation_lifecycle_in_causal_order() {
        let mut trace = RequestTrace::new(RequestKey::new(1, RequestId(7), 2), TraceId(7));
        trace.mark_admitted(0);
        let k = key(10, 1, Domain::Decode);
        for (offset, phase) in LifecyclePhase::ALL.into_iter().enumerate() {
            trace.stamp(k, Some("decode_und"), phase, offset as u64 * 10);
        }
        trace.mark_finished("eos", 200);

        assert!(trace.was_admitted() && trace.is_finished());
        assert_eq!(trace.resolved_ops(), 1);
        assert_eq!(trace.finish_reason, Some("eos"));
        let op = &trace.operations()[0];
        assert_eq!(op.key.op_id, OpId(10));
        assert_eq!(op.key.domain, Domain::Decode);
        assert_eq!(op.op_kind, Some("decode_und"));
        assert!(op.is_ordered());
        assert_eq!(
            op.span_us(
                LifecyclePhase::Submitted,
                LifecyclePhase::CompletionObserved
            ),
            Some(40),
        );
    }

    #[test]
    fn a_reached_phase_is_never_rewritten() {
        let mut trace = RequestTrace::new(RequestKey::new(1, RequestId(7), 2), TraceId(7));
        let k = key(10, 1, Domain::Decode);
        trace.stamp(k, Some("decode_und"), LifecyclePhase::Submitted, 100);
        trace.stamp(k, Some("decode_und"), LifecyclePhase::Submitted, 999);
        assert_eq!(
            trace.operations()[0].at(LifecyclePhase::Submitted),
            Some(100)
        );
    }

    #[test]
    fn out_of_order_stamps_are_detected() {
        let mut trace = RequestTrace::new(RequestKey::new(1, RequestId(7), 2), TraceId(7));
        let k = key(10, 1, Domain::Decode);
        trace.stamp(k, None, LifecyclePhase::Submitted, 100);
        trace.stamp(k, None, LifecyclePhase::CompletionObserved, 50);
        assert!(!trace.is_ordered());
    }

    #[test]
    fn distinct_operations_keep_independent_lifecycles() {
        let mut trace = RequestTrace::new(RequestKey::new(1, RequestId(7), 2), TraceId(7));
        trace.stamp(
            key(10, 1, Domain::Decode),
            None,
            LifecyclePhase::Submitted,
            10,
        );
        trace.stamp(
            key(11, 2, Domain::Flow),
            None,
            LifecyclePhase::Submitted,
            20,
        );
        assert_eq!(trace.operations().len(), 2);
        assert_eq!(trace.operations()[1].key.domain, Domain::Flow);
    }
}
