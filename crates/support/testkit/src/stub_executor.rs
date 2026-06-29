//! A configurable [`Executor`] test double for integration tests: it declares
//! configurable [`EngineCaps`], serves a queue of canned [`ForwardResult`]s,
//! and records every submitted [`ForwardBatch`] for assertions.

use std::collections::VecDeque;

use uniserve_core::RequestId;
use uniserve_executor::{ControlAck, ControlOp, Executor};
use uniserve_worker_wire::{EngineCaps, ForwardBatch, ForwardResult, SeqResult};

/// Configurable [`Executor`] double for unit and integration tests.
///
/// Behavior is fully driven by its fields:
///
/// * [`caps`](Self::caps) — the [`EngineCaps`] reported to the scheduler.
/// * [`results`](Self::results) — canned [`ForwardResult`]s served in FIFO order
///   by [`poll`](Executor::poll) / [`next_result`](Executor::next_result). When
///   the queue is exhausted, [`echo_submitted`](Self::echo_submitted) decides
///   whether a submit synthesizes an echo result or leaves nothing to poll.
/// * [`submitted`](Self::submitted) — every [`ForwardBatch`] passed to
///   [`submit`](Executor::submit), in order, for post-hoc assertions.
/// * [`controls`](Self::controls) — every [`ControlOp`] passed to
///   [`control`](Executor::control) / [`control_wait`](Executor::control_wait).
pub struct StubExecutor {
    pub caps: EngineCaps,
    pub results: VecDeque<ForwardResult>,
    pub submitted: Vec<ForwardBatch>,
    pub controls: Vec<ControlOp>,
    /// When `results` is empty, synthesize an echo [`ForwardResult`] per submit
    /// (one [`SeqResult`] per op, `sampled_token_id` set to the request id). This
    /// reproduces the `PoolExec` pass-through behavior so a stub with no preloaded
    /// results still returns one result per batch.
    pub echo_submitted: bool,
    ready: VecDeque<ForwardResult>,
}

impl Default for StubExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl StubExecutor {
    /// A stub with default [`EngineCaps`], no canned results, and echo enabled.
    pub fn new() -> Self {
        Self {
            caps: EngineCaps::default(),
            results: VecDeque::new(),
            submitted: Vec::new(),
            controls: Vec::new(),
            echo_submitted: true,
            ready: VecDeque::new(),
        }
    }

    /// Set the reported [`EngineCaps`].
    pub fn with_caps(mut self, caps: EngineCaps) -> Self {
        self.caps = caps;
        self
    }

    /// Set the reported `pipeline_depth` capability.
    pub fn with_pipeline_depth(mut self, depth: u32) -> Self {
        self.caps.pipeline_depth = depth;
        self
    }

    /// Preload canned [`ForwardResult`]s, served in FIFO order, and stop
    /// synthesizing echo results.
    pub fn with_results(mut self, results: impl IntoIterator<Item = ForwardResult>) -> Self {
        self.results = results.into_iter().collect();
        self.echo_submitted = false;
        self
    }

    /// Whether to synthesize an echo [`ForwardResult`] per submit once the canned
    /// `results` queue is exhausted.
    pub fn echo_when_empty(mut self, echo: bool) -> Self {
        self.echo_submitted = echo;
        self
    }

    fn echo_result(batch: &ForwardBatch) -> ForwardResult {
        let per_seq = batch
            .ops
            .iter()
            .map(|op| SeqResult {
                req_id: op.req_id,
                sampled_token_id: Some(op.req_id.0 as u32),
                op_id: op.op_id,
                ..Default::default()
            })
            .collect();
        ForwardResult {
            step_id: batch.step_id,
            per_seq,
            worker_exec_us: Some(0),
            forward_stats: None,
        }
    }
}

impl Executor for StubExecutor {
    fn caps(&self) -> EngineCaps {
        self.caps.clone()
    }

    fn pipeline_depth(&self) -> usize {
        self.caps.pipeline_depth.max(1) as usize
    }

    fn in_flight(&self) -> usize {
        self.ready.len()
    }

    fn submit(&mut self, batch: ForwardBatch) -> anyhow::Result<()> {
        if let Some(result) = self.results.pop_front() {
            self.ready.push_back(result);
        } else if self.echo_submitted {
            self.ready.push_back(Self::echo_result(&batch));
        }
        self.submitted.push(batch);
        Ok(())
    }

    fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
        Ok(self.ready.pop_front())
    }

    fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
        self.ready
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("StubExecutor has no ready result"))
    }

    fn control(&mut self, op: ControlOp) -> anyhow::Result<u64> {
        self.controls.push(op);
        Ok(self.controls.len() as u64)
    }

    fn control_wait(
        &mut self,
        op: ControlOp,
        _targets: Option<&[u32]>,
    ) -> anyhow::Result<Vec<ControlAck>> {
        self.controls.push(op);
        Ok(vec![ControlAck {
            rank: 0,
            ok: true,
            message: None,
        }])
    }
}

impl StubExecutor {
    /// Drop the next ready result without polling it (e.g. to model a worker that
    /// discarded `req_id`'s in-flight op).
    pub fn forget_ready(&mut self, _req_id: RequestId) {
        self.ready.pop_front();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::Modality;
    use uniserve_worker_wire::{ForwardOp, OpKind};

    fn batch(step_id: u64, req_id: u64) -> ForwardBatch {
        ForwardBatch {
            step_id,
            new_reqs: Vec::new(),
            ops: vec![ForwardOp {
                req_id: RequestId(req_id),
                kind: OpKind::DecodeUnd,
                modality: Modality::Und,
                ..Default::default()
            }],
        }
    }

    #[test]
    fn echo_mode_records_batches_and_echoes_one_result_per_op() {
        let mut exec = StubExecutor::new().with_pipeline_depth(4);
        exec.submit(batch(5, 7)).unwrap();
        assert_eq!(exec.submitted.len(), 1);
        assert_eq!(exec.pipeline_depth(), 4);
        let out = exec.poll().unwrap().expect("echoed result");
        assert_eq!(out.step_id, 5);
        assert_eq!(out.per_seq[0].req_id, RequestId(7));
        assert_eq!(out.per_seq[0].sampled_token_id, Some(7));
        assert!(exec.poll().unwrap().is_none());
    }

    #[test]
    fn canned_results_serve_fifo_then_stop() {
        let canned = ForwardResult {
            step_id: 99,
            per_seq: vec![SeqResult {
                req_id: RequestId(1),
                sampled_token_id: Some(1234),
                ..Default::default()
            }],
            worker_exec_us: Some(1),
            forward_stats: None,
        };
        let mut exec = StubExecutor::new().with_results([canned]);
        exec.submit(batch(1, 1)).unwrap();
        let out = exec.next_result().unwrap();
        assert_eq!(out.step_id, 99);
        assert_eq!(out.per_seq[0].sampled_token_id, Some(1234));
        // No echo once canned results are exhausted.
        exec.submit(batch(2, 2)).unwrap();
        assert!(exec.poll().unwrap().is_none());
    }

    #[test]
    fn control_calls_are_recorded() {
        let mut exec = StubExecutor::new();
        exec.control(ControlOp::ResetPrefixCache).unwrap();
        let acks = exec.control_wait(ControlOp::Sleep, None).unwrap();
        assert_eq!(exec.controls.len(), 2);
        assert!(acks[0].ok);
    }
}
