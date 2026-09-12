//! Scheduler-owned pending batches and request-local completion ordering.

use super::*;
use uniserve_worker_ipc::PipelineStage;

/// Algorithm inputs retained until their computation completes.
pub(super) enum InflightInput {
    Generation {
        /// Physical KV start and exact token contribution, when known, from the
        /// submitted image-extension input. The operation carries its capacity.
        image_kv: Option<(u32, Option<u32>)>,
        /// Actual latent input step; cancellation can stop accepted progress
        /// while later submitted intervals still need result validation.
        start_step: Option<u32>,
    },
    Media {
        latent: Option<LatentParams>,
        decode: Option<DecodeRange>,
    },
}

/// Submitted operation and timing state awaiting completion.
pub(super) struct InflightOp {
    pub(super) operation: ScheduledRequest,
    pub(super) input: InflightInput,
    pub(super) started: Instant,
}

/// Completion held until earlier request operations are applied.
pub(super) struct PendingCompletion {
    pub(super) record: ModelOutput,
    /// Claimed media storage stays alive while its result waits or is discarded.
    pub(super) media: Option<Arc<SharedMedia>>,
    pub(super) arrival_seq: u64,
}

/// Terminal event held until outstanding operations are reconciled.
pub(super) struct PendingFinish {
    pub(super) reason: FinishReason,
    pub(super) stop_reason: Option<uniserve_core::StopReason>,
}

/// One submitted batch remains owned until all results and command receipts arrive.
/// Partial results consume identities without releasing the batch's queue credit.
pub(super) struct PendingBatch {
    pub(super) started: Instant,
    pub(super) operations: HashSet<(RequestKey, ComputationId)>,
    pub(super) commands: Vec<BatchCommand>,
    pub(super) worker_exec_us: u64,
    pub(super) prefill: bool,
}

impl Scheduler {
    /// Returns and advances the next batch identifier.
    pub(super) fn next_batch_id(&mut self) -> u64 {
        self.batch_id = self.batch_id.saturating_add(1);
        self.batch_id
    }

    /// Returns and advances the next arrival sequence.
    pub(super) fn next_arrival(&mut self) -> u64 {
        let arrival = self.next_arrival_seq;
        self.next_arrival_seq = self.next_arrival_seq.saturating_add(1);
        arrival
    }

    /// Whether the request still owns a submitted operation.
    pub(super) fn has_pending_operations(&self, id: RequestId) -> bool {
        self.pending_operations
            .get(&id)
            .is_some_and(|queue| !queue.is_empty())
    }

    /// Number of submitted operations still owned by the request.
    pub(super) fn num_pending_operations(&self, id: RequestId) -> usize {
        self.pending_operations.get(&id).map_or(0, VecDeque::len)
    }

    /// Returns whether any in-flight operation performs denoising.
    pub(super) fn has_pending_denoising(&self) -> bool {
        self.pending_operations
            .values()
            .flatten()
            .any(|op| op.operation.code == Computation::Pipeline(PipelineStage::Denoising))
    }

    /// Stateful operations retain request order. Pure media branches complete
    /// independently once their actual input producers have resolved.
    pub(super) fn take_ready_completions(&mut self) -> Vec<PendingCompletion> {
        let mut ready = Vec::new();
        for (id, pending) in &self.pending_completions {
            let Some(queue) = self.pending_operations.get(id) else {
                continue;
            };
            for (index, inflight) in queue.iter().enumerate() {
                let operation = &inflight.operation;
                let independent = matches!(inflight.input, InflightInput::Media { .. })
                    && !operation.advances_state();
                if index > 0 && !independent {
                    continue;
                }
                if independent
                    && operation.tensor_inputs().any(|input| {
                        queue
                            .iter()
                            .any(|producer| producer.operation.op_id == input.producer_op_id)
                    })
                {
                    continue;
                }
                if let Some(completion) = pending.get(&operation.op_id) {
                    ready.push((
                        completion_priority(operation.code),
                        completion.arrival_seq,
                        *id,
                        operation.op_id,
                    ));
                }
            }
        }
        ready.sort_unstable_by_key(|(priority, arrival, ..)| (*priority, *arrival));
        let mut completions = Vec::with_capacity(ready.len());
        for (_, _, id, op_id) in ready {
            let pending = self
                .pending_completions
                .get_mut(&id)
                .expect("selected completion exists");
            completions.push(pending.remove(&op_id).expect("selected operation exists"));
            if pending.is_empty() {
                self.pending_completions.remove(&id);
            }
        }
        completions
    }

    /// Removes a selected operation and releases its transfer reservation.
    pub(super) fn pop_pending_operation(
        &mut self,
        request_key: RequestKey,
        op_id: ComputationId,
    ) -> Option<InflightOp> {
        let id = request_key.request_id;
        let queue = self.pending_operations.get_mut(&id)?;
        let index = queue.iter().position(|inflight| {
            inflight.operation.request_key == request_key && inflight.operation.op_id == op_id
        })?;
        let selected = &queue[index];
        if op_id.batch_id == 0
            || (index > 0
                && !(matches!(selected.input, InflightInput::Media { .. })
                    && !selected.operation.advances_state()))
        {
            return None;
        }
        let inflight = queue.remove(index).expect("selected operation exists");
        if inflight.operation.bounds.max_transfer_bytes > 0 {
            self.num_pending_transfers = self
                .num_pending_transfers
                .checked_sub(1)
                .expect("completed transfer owns a reservation");
        }
        if queue.is_empty() {
            self.pending_operations.remove(&id);
        }
        Some(inflight)
    }

    /// Removes failed operations from the in-flight registry.
    pub(super) fn retire_operation(
        &mut self,
        batch_id: u64,
        request: RequestKey,
        op: ComputationId,
    ) -> Option<InflightOp> {
        if !self
            .pending_batches
            .get_mut(&batch_id)?
            .operations
            .remove(&(request, op))
        {
            return None;
        }
        let queue = self.pending_operations.get_mut(&request.request_id)?;
        let position = queue.iter().position(|inflight| {
            inflight.operation.request_key == request && inflight.operation.op_id == op
        })?;
        let inflight = queue.remove(position)?;
        if queue.is_empty() {
            self.pending_operations.remove(&request.request_id);
        }
        if inflight.operation.bounds.max_transfer_bytes > 0 {
            self.num_pending_transfers = self
                .num_pending_transfers
                .checked_sub(1)
                .expect("retired transfer owns a reservation");
        }
        Some(inflight)
    }

    /// Removes failed operations from the in-flight registry.
    pub(super) fn clear_failed_operations(&mut self) -> (Vec<RequestId>, Vec<BatchCommand>) {
        let ids = self.pending_operations.keys().copied().collect();
        let commands = self
            .pending_batches
            .drain()
            .flat_map(|(_, batch)| batch.commands)
            .collect();
        self.pending_operations.clear();
        self.num_pending_transfers = 0;
        self.pending_completions.clear();
        (ids, commands)
    }
}
