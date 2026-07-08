//! Per-request lifecycle trace.
//!
//! A compact, reconstructable event log per request: admitted → op submitted →
//! op resolved (with host round-trip + worker compute time) → finished. Carries
//! the trace/request/op ids the runtime requires so a full lifecycle can be reconstructed after the fact. Bounded; completed traces move to a ring.

use uniserve_core::{RequestId, TraceId};

const MAX_EVENTS: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceEventKind {
    Admitted,
    OpSubmitted,
    OpResolved,
    Preempted,
    Finished,
}

#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub kind: TraceEventKind,
    pub op_id: Option<u64>,
    pub op_kind: Option<&'static str>,
    pub step_id: u64,
    /// Host-observed submit→resolve round-trip (ring + worker), microseconds.
    pub roundtrip_us: u64,
    /// Worker-reported compute time for the op's batch, microseconds.
    pub worker_us: u64,
    pub finish_reason: Option<&'static str>,
}

impl TraceEvent {
    pub fn at(kind: TraceEventKind) -> Self {
        Self {
            kind,
            op_id: None,
            op_kind: None,
            step_id: 0,
            roundtrip_us: 0,
            worker_us: 0,
            finish_reason: None,
        }
    }
}

/// The lifecycle trace of one request.
#[derive(Debug, Clone)]
pub struct RequestTrace {
    pub request_id: RequestId,
    pub trace_id: TraceId,
    pub events: Vec<TraceEvent>,
}

impl RequestTrace {
    pub fn new(request_id: RequestId, trace_id: TraceId) -> Self {
        Self {
            request_id,
            trace_id,
            events: Vec::new(),
        }
    }

    pub fn push(&mut self, ev: TraceEvent) {
        if self.events.len() < MAX_EVENTS {
            self.events.push(ev);
        }
    }

    /// Count of resolved ops (one lifecycle "step" completed).
    pub fn resolved_ops(&self) -> usize {
        self.events
            .iter()
            .filter(|e| e.kind == TraceEventKind::OpResolved)
            .count()
    }

    pub fn is_finished(&self) -> bool {
        self.events
            .iter()
            .any(|e| e.kind == TraceEventKind::Finished)
    }

    pub fn was_admitted(&self) -> bool {
        self.events
            .iter()
            .any(|e| e.kind == TraceEventKind::Admitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_records_a_reconstructable_lifecycle() {
        let mut t = RequestTrace::new(RequestId(1), TraceId(1));
        t.push(TraceEvent::at(TraceEventKind::Admitted));
        let mut sub = TraceEvent::at(TraceEventKind::OpSubmitted);
        sub.op_id = Some(10);
        sub.op_kind = Some("prefill_und");
        sub.step_id = 1;
        t.push(sub);
        let mut res = TraceEvent::at(TraceEventKind::OpResolved);
        res.op_id = Some(10);
        res.roundtrip_us = 1234;
        res.worker_us = 1000;
        t.push(res);
        let mut fin = TraceEvent::at(TraceEventKind::Finished);
        fin.finish_reason = Some("eos");
        t.push(fin);

        assert!(t.was_admitted() && t.is_finished());
        assert_eq!(t.resolved_ops(), 1);
        // ids are present for correlation.
        assert_eq!(t.trace_id, TraceId(1));
        let submitted = t
            .events
            .iter()
            .find(|e| e.kind == TraceEventKind::OpSubmitted)
            .unwrap();
        assert_eq!(submitted.op_id, Some(10));
        assert_eq!(submitted.op_kind, Some("prefill_und"));
    }
}
