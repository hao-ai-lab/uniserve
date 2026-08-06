//! A configurable [`Executor`] test double for integration tests: it declares
//! configurable [`EngineCaps`], serves a queue of canned [`CompletionReport`]s,
//! and records every submitted [`Batch`] for assertions.

use std::collections::VecDeque;

use uniserve_core::RequestId;
use uniserve_executor::{ControlAck, ControlOp, Executor};
use uniserve_worker_wire::{
    Batch, CompletionRecord, CompletionReport, EngineCaps, FinishFlags, LogicalLengths, OpStatus,
    Operation, PartitionCompletion, Point, RegistrationAck, TimingCounters, TokenSpan, Work,
};

/// Configurable [`Executor`] double for unit and integration tests.
///
/// Behavior is fully driven by its fields:
///
/// * [`caps`](Self::caps) — the [`EngineCaps`] reported to the scheduler.
/// * [`results`](Self::results) — canned [`CompletionReport`]s served in FIFO
///   order by [`poll`](Executor::poll) / [`next_result`](Executor::next_result).
///   When the queue is exhausted, [`echo_submitted`](Self::echo_submitted)
///   decides whether a submit synthesizes an echo report or leaves nothing to
///   poll.
/// * [`submitted`](Self::submitted) — every [`Batch`] passed to
///   [`submit`](Executor::submit), in order, for post-hoc assertions.
/// * [`controls`](Self::controls) — every [`ControlOp`] passed to
///   [`control`](Executor::control) / [`control_wait`](Executor::control_wait).
pub struct StubExecutor {
    pub caps: EngineCaps,
    pub results: VecDeque<CompletionReport>,
    pub submitted: Vec<Batch>,
    pub controls: Vec<ControlOp>,
    /// When `results` is empty, synthesize one completion per operation.
    pub echo_submitted: bool,
    ready: VecDeque<CompletionReport>,
}

impl Default for StubExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl StubExecutor {
    /// A stub with default [`EngineCaps`], no canned reports, and echo enabled.
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

    /// Preload canned [`CompletionReport`]s, served in FIFO order, and stop
    /// synthesizing echo reports.
    pub fn with_results(mut self, results: impl IntoIterator<Item = CompletionReport>) -> Self {
        self.results = results.into_iter().collect();
        self.echo_submitted = false;
        self
    }

    /// Whether to synthesize an echo [`CompletionReport`] per submit once the
    /// canned `results` queue is exhausted.
    pub fn echo_when_empty(mut self, echo: bool) -> Self {
        self.echo_submitted = echo;
        self
    }

    /// The terminal completion an echoing stub emits for one operation. A token
    /// operation commits the request's session id as its single token; every
    /// other work variant contributes no committed tokens.
    fn echo_record(operation: &Operation) -> CompletionRecord {
        let parent_semantic = match &operation.parent.point {
            Point::Fixed {
                semantic_digest, ..
            } => semantic_digest.clone(),
            Point::Device { .. } => "0".repeat(64),
        };
        let selected_point = u32::from(operation.advances_state);
        let committed_tokens = if matches!(operation.work, Work::Token(_)) {
            vec![operation.request_key.session_id.0 as u32]
        } else {
            Vec::new()
        };
        let mut record = CompletionRecord {
            request_key: operation.request_key,
            op_id: operation.op_id,
            completion_slot_generation: ((operation.op_id.0 - 1) % u64::from(u32::MAX) + 1) as u32,
            status: OpStatus::Ok,
            selected_point,
            logical_lengths: LogicalLengths {
                token_len: committed_tokens.len() as u32,
                kv_visible_len: 0,
                latent_len: 0,
                ..LogicalLengths::default()
            },
            token_span: TokenSpan {
                base: 0,
                len: committed_tokens.len() as u32,
            },
            committed_tokens,
            finish_flags: FinishFlags::default(),
            product_generations: operation.outputs.iter().map(|out| out.generation).collect(),
            semantic_digest: String::new(),
            error_code: None,
            timing_counters: TimingCounters::default(),
        };
        record.semantic_digest =
            record.compute_semantic_digest(&parent_semantic, &operation.plan_digest);
        record
    }

    fn echo_result(batch: &Batch) -> CompletionReport {
        CompletionReport {
            step_id: batch.step_id,
            partitions: batch
                .partitions
                .iter()
                .map(|partition| PartitionCompletion {
                    partition_id: partition.partition_id,
                    completions: partition.operations.iter().map(Self::echo_record).collect(),
                    products: Vec::new(),
                    registration: RegistrationAck { visible: true },
                    worker_exec_us: Some(0),
                    forward_stats: None,
                })
                .collect(),
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

    fn submit(&mut self, batch: Batch) -> anyhow::Result<()> {
        batch.validate()?;
        if let Some(result) = self.results.pop_front() {
            self.ready.push_back(result);
        } else if self.echo_submitted {
            self.ready.push_back(Self::echo_result(&batch));
        }
        self.submitted.push(batch);
        Ok(())
    }

    fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
        Ok(self.ready.pop_front())
    }

    fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
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
            snapshot: None,
        }])
    }
}

impl StubExecutor {
    /// Drop the next ready report without polling it (e.g. to model a worker that
    /// discarded `req_id`'s in-flight op).
    pub fn forget_ready(&mut self, _req_id: RequestId) {
        self.ready.pop_front();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::SamplingParams;
    use uniserve_worker_wire::{
        Admission, AttentionRegime, BatchPartition, Bounds, Domain, ExecutionCapability, OpId,
        RequestKey, RouteId, TokenMode, UndAdmission, VersionRef,
    };

    fn batch(step_id: u64, session_id: u64) -> Batch {
        let request_key = RequestKey::new(0, RequestId(session_id), 1);
        let admission = Admission::new(
            request_key,
            Some(UndAdmission {
                sampling: SamplingParams::default(),
                negative_token_ids: Vec::new(),
                finish_token_ids: Vec::new(),
                kv: Default::default(),
            }),
            None,
        )
        .expect("admission");
        let parent = VersionRef::admission_root(request_key, OpId(1), admission.digest.clone());
        let operation = Operation::registered(
            request_key,
            OpId(step_id.max(1)),
            parent,
            Work::Token(TokenMode::Decode),
            RouteId(0),
            Domain::Und,
            Bounds {
                max_points: 1,
                max_tokens: 1,
                ..Bounds::default()
            },
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            None,
            0,
        );
        Batch::new(
            step_id,
            vec![admission],
            vec![BatchPartition {
                partition_id: 1,
                submission_group: 1,
                collective_seq: step_id.max(1),
                domain: Domain::Und,
                route: RouteId(0),
                execution: ExecutionCapability::DomainHomogeneous,
                attention: AttentionRegime::Causal,
                shape_class: 0,
                operations: vec![operation],
            }],
        )
    }

    #[test]
    fn echo_mode_records_batches_and_echoes_one_completion_per_operation() {
        let mut executor = StubExecutor::new().with_pipeline_depth(4);
        executor.submit(batch(1, 7)).expect("submit");
        assert_eq!(executor.submitted.len(), 1);
        assert_eq!(executor.pipeline_depth(), 4);
        let report = executor.poll().expect("poll").expect("echoed report");
        assert_eq!(report.step_id, 1);
        let completion = report.completions().next().unwrap();
        assert_eq!(completion.request_key.session_id, RequestId(7));
        assert_eq!(completion.committed_tokens, vec![7]);
        assert!(executor.poll().expect("poll").is_none());
    }

    #[test]
    fn canned_results_serve_fifo_then_stop() {
        let canned = StubExecutor::echo_result(&batch(99, 1));
        let mut executor = StubExecutor::new().with_results([canned]);
        executor.submit(batch(1, 1)).expect("submit");
        assert_eq!(executor.next_result().expect("result").step_id, 99);
        executor.submit(batch(2, 2)).expect("submit");
        assert!(executor.poll().expect("poll").is_none());
    }

    #[test]
    fn control_calls_are_recorded() {
        let mut executor = StubExecutor::new();
        executor
            .control(ControlOp::ResetPrefixCache)
            .expect("control");
        let acknowledgements = executor
            .control_wait(ControlOp::ReleaseProducts(Vec::new()), None)
            .expect("control wait");
        assert_eq!(executor.controls.len(), 2);
        assert!(acknowledgements[0].ok);
    }
}
