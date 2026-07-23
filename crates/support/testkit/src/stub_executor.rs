//! A configurable [`Executor`] test double for integration tests: it declares
//! configurable [`EngineCaps`], serves a queue of canned [`ExecutionResult`]s,
//! and records every submitted [`Batch`] for assertions.

use std::collections::VecDeque;

use uniserve_core::RequestId;
use uniserve_executor::{ControlAck, ControlOp, Executor};
use uniserve_worker_wire::{
    Batch, EncodeDelta, EngineCaps, ExecutionResult, FlowDelta, MaterializeDelta,
    MaterializedProduct, Operation, OperationResult, ResultDelta, SequenceDelta, SequenceEffect,
    TransferDelta,
};

/// Configurable [`Executor`] double for unit and integration tests.
///
/// Behavior is fully driven by its fields:
///
/// * [`caps`](Self::caps) — the [`EngineCaps`] reported to the scheduler.
/// * [`results`](Self::results) — canned [`ExecutionResult`]s served in FIFO order
///   by [`poll`](Executor::poll) / [`next_result`](Executor::next_result). When
///   the queue is exhausted, [`echo_submitted`](Self::echo_submitted) decides
///   whether a submit synthesizes an echo result or leaves nothing to poll.
/// * [`submitted`](Self::submitted) — every [`Batch`] passed to
///   [`submit`](Executor::submit), in order, for post-hoc assertions.
/// * [`controls`](Self::controls) — every [`ControlOp`] passed to
///   [`control`](Executor::control) / [`control_wait`](Executor::control_wait).
pub struct StubExecutor {
    pub caps: EngineCaps,
    pub results: VecDeque<ExecutionResult>,
    pub submitted: Vec<Batch>,
    pub controls: Vec<ControlOp>,
    /// When `results` is empty, synthesize one typed result per operation.
    pub echo_submitted: bool,
    ready: VecDeque<ExecutionResult>,
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

    /// Preload canned [`ExecutionResult`]s, served in FIFO order, and stop
    /// synthesizing echo results.
    pub fn with_results(mut self, results: impl IntoIterator<Item = ExecutionResult>) -> Self {
        self.results = results.into_iter().collect();
        self.echo_submitted = false;
        self
    }

    /// Whether to synthesize an echo [`ExecutionResult`] per submit once the canned
    /// `results` queue is exhausted.
    pub fn echo_when_empty(mut self, echo: bool) -> Self {
        self.echo_submitted = echo;
        self
    }

    fn echo_result(batch: &Batch) -> ExecutionResult {
        let operations = batch
            .operations
            .iter()
            .map(|operation| OperationResult {
                session_id: operation.session_id,
                epoch: operation.epoch,
                op_id: operation.op_id,
                base_version: operation.base_version,
                result_version: operation.base_version.saturating_add(1),
                delta: match &operation.operation {
                    Operation::Sequence(_) => ResultDelta::Sequence(SequenceDelta {
                        effect: SequenceEffect {
                            sampled_token_ids: vec![operation.session_id.0 as u32],
                            ..SequenceEffect::default()
                        },
                    }),
                    Operation::Flow(flow) => ResultDelta::Flow(FlowDelta {
                        steps_completed: flow.start_step.saturating_add(flow.step_count),
                        done: false,
                    }),
                    Operation::Encode(_) => ResultDelta::Encode(EncodeDelta {
                        product_handle: operation.session_id.0.max(1),
                        kv_tokens: 1,
                        image_size: None,
                    }),
                    Operation::Materialize(_) => ResultDelta::Materialize(MaterializeDelta {
                        product: MaterializedProduct::Published(
                            uniserve_worker_wire::PublishedProduct {
                                handle: operation.session_id.0.max(1),
                                locator: format!("stub-product-{}", operation.session_id.0),
                            },
                        ),
                        kv_tokens: None,
                        sequence: None,
                    }),
                    Operation::Transfer(transfer) => ResultDelta::Transfer(TransferDelta {
                        product: Some(transfer.source.clone()),
                        kv_tokens: None,
                        sequence: None,
                    }),
                },
            })
            .collect();
        ExecutionResult {
            step_id: batch.step_id,
            operations,
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

    fn poll(&mut self) -> anyhow::Result<Option<ExecutionResult>> {
        Ok(self.ready.pop_front())
    }

    fn next_result(&mut self) -> anyhow::Result<ExecutionResult> {
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
    /// Drop the next ready result without polling it (e.g. to model a worker that
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
        Admission, KvAllocation, KvLeaseDelta, SequenceAdmission, SequenceInput, SequenceMode,
        SequenceOperation, TokenInput, TokenPolicy, TokenSource,
    };

    fn batch(step_id: u64, session_id: u64) -> Batch {
        let admission = Admission::new(
            RequestId(session_id),
            Some(SequenceAdmission {
                sampling: SamplingParams::default(),
                negative_token_ids: Vec::new(),
                kv: KvAllocation {
                    block_ids: Vec::new(),
                    prefix_len: 0,
                    group_id: 0,
                },
            }),
            None,
            None,
        )
        .expect("admission");
        let mut operation = uniserve_worker_wire::OperationEnvelope::unsealed(
            RequestId(session_id),
            Operation::Sequence(SequenceOperation {
                mode: SequenceMode::Decode,
                lease: KvLeaseDelta::default(),
                position: (0, 1),
                policy: TokenPolicy::default(),
                input: SequenceInput::Tokens(TokenInput {
                    token_ids: vec![1],
                    source: TokenSource::Wire,
                    draft_token_ids: Vec::new(),
                    burst_tokens: 1,
                    stop_token_ids: Vec::new(),
                    stop_terminal: false,
                    return_all_logits: false,
                }),
            }),
        );
        operation.admission_digest = admission.digest.clone();
        operation.model_spec_digest = "0".repeat(64);
        operation.weight_digest = "1".repeat(64);
        operation.seal(1, step_id, step_id.saturating_sub(1));
        Batch::new(step_id, vec![admission], vec![operation])
    }

    #[test]
    fn echo_mode_records_batches_and_echoes_one_result_per_operation() {
        let mut executor = StubExecutor::new().with_pipeline_depth(4);
        executor.submit(batch(1, 7)).expect("submit");
        assert_eq!(executor.submitted.len(), 1);
        assert_eq!(executor.pipeline_depth(), 4);
        let result = executor.poll().expect("poll").expect("echoed result");
        assert_eq!(result.step_id, 1);
        assert_eq!(result.operations[0].session_id, RequestId(7));
        let ResultDelta::Sequence(delta) = &result.operations[0].delta else {
            panic!("expected sequence delta");
        };
        assert_eq!(delta.effect.sampled_token_ids, vec![7]);
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
