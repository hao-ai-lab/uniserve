//! Structured scheduling facts, explainable decisions, and latency history.
//!
//! This module provides read-only telemetry around the scheduler's admission
//! and admission policy while scheduling authority remains in [`crate::Scheduler`].

use std::collections::HashMap;
use std::collections::VecDeque;

use uniserve_core::RequestId;

/// The facts the scheduler weighs each admission pass.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolicySnapshot {
    // queue depths
    pub waiting: usize,
    pub running: usize,
    pub skipped_waiting: usize,
    pub in_flight: usize,
    // resource pressure
    pub free_blocks: usize,
    pub total_blocks: usize,
    pub reserved_blocks: usize,
    pub active_credit_requests: usize,
    // cache facts
    pub cached_blocks: usize,
    pub prefix_hit_rate: f32,
    pub mm_cache_hit_rate: f32,
    // latency history: per-op-kind round-trip EWMA, microseconds.
    pub op_latency_us: Vec<(String, u64)>,
}

/// Why an admission or rejection decision was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyReason {
    Admitted,
    DelayedNoBlocks,
    DelayedScratch,
    DelayedMaxSeqs,
    RejectedTooLarge,
}

impl PolicyReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            PolicyReason::Admitted => "admitted",
            PolicyReason::DelayedNoBlocks => "delayed_no_blocks",
            PolicyReason::DelayedScratch => "delayed_scratch",
            PolicyReason::DelayedMaxSeqs => "delayed_max_seqs",
            PolicyReason::RejectedTooLarge => "rejected_too_large",
        }
    }
}

/// One explainable policy decision (compact; drained into traces/stats).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyDecision {
    pub request: RequestId,
    pub reason: PolicyReason,
    pub free_blocks: usize,
    pub needed_blocks: usize,
}

/// Per-op-kind EWMA latency history.
#[derive(Debug, Clone)]
pub struct LatencyHistory {
    ewma_us: HashMap<&'static str, f64>,
    alpha: f64,
}

impl Default for LatencyHistory {
    fn default() -> Self {
        Self {
            ewma_us: HashMap::new(),
            alpha: 0.2,
        }
    }
}

impl LatencyHistory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one observed round-trip latency for `kind` into its EWMA.
    pub fn observe(&mut self, kind: &'static str, us: u64) {
        let e = self.ewma_us.entry(kind).or_insert(us as f64);
        *e = self.alpha * us as f64 + (1.0 - self.alpha) * *e;
    }

    pub fn get(&self, kind: &str) -> Option<u64> {
        self.ewma_us.get(kind).map(|v| *v as u64)
    }

    pub fn as_pairs(&self) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = self
            .ewma_us
            .iter()
            .map(|(k, x)| (k.to_string(), *x as u64))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }
}

/// A bounded ring of recent policy decisions for explainability.
#[derive(Debug, Clone)]
pub struct DecisionLog {
    ring: VecDeque<PolicyDecision>,
    cap: usize,
    pub admitted: u64,
    pub delayed: u64,
    pub rejected: u64,
}

impl Default for DecisionLog {
    fn default() -> Self {
        Self {
            ring: VecDeque::new(),
            cap: 256,
            admitted: 0,
            delayed: 0,
            rejected: 0,
        }
    }
}

impl DecisionLog {
    pub fn record(&mut self, d: PolicyDecision) {
        match d.reason {
            PolicyReason::Admitted => self.admitted += 1,
            PolicyReason::RejectedTooLarge => self.rejected += 1,
            PolicyReason::DelayedNoBlocks
            | PolicyReason::DelayedScratch
            | PolicyReason::DelayedMaxSeqs => self.delayed += 1,
        }
        if self.ring.len() >= self.cap {
            self.ring.pop_front();
        }
        self.ring.push_back(d);
    }

    pub fn recent(&self) -> impl Iterator<Item = &PolicyDecision> {
        self.ring.iter()
    }

    pub fn drain(&mut self) -> Vec<PolicyDecision> {
        self.ring.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewma_converges_toward_observations() {
        let mut h = LatencyHistory::new();
        for _ in 0..50 {
            h.observe("decode_und", 1000);
        }
        let v = h.get("decode_und").unwrap();
        assert!(
            (900..=1100).contains(&v),
            "ewma should converge near 1000, got {v}"
        );
        assert!(h.get("prefill_und").is_none());
    }

    #[test]
    fn decision_log_counts_by_reason_and_bounds() {
        let mut log = DecisionLog::default();
        for i in 0..300 {
            log.record(PolicyDecision {
                request: RequestId(i),
                reason: PolicyReason::Admitted,
                free_blocks: 10,
                needed_blocks: 1,
            });
        }
        log.record(PolicyDecision {
            request: RequestId(999),
            reason: PolicyReason::RejectedTooLarge,
            free_blocks: 0,
            needed_blocks: 99999,
        });
        assert_eq!(log.admitted, 300);
        assert_eq!(log.rejected, 1);
        assert!(log.recent().count() <= 256, "ring is bounded");
    }
}
