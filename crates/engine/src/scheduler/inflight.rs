//! Scheduler-owned pending batches and request-local completion ordering.

use super::*;
use uniserve_worker_ipc::PipelineStage;

/// Algorithm inputs retained until their computation completes.
pub(super) enum InflightInput {
    Generation {
        /// Physical KV start and exact token contribution, when known, from the
        /// submitted image-extension input. The call carries its capacity.
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

/// Submitted call and timing state awaiting completion.
pub(super) struct InflightOp {
    pub(super) call: Call,
    pub(super) input: InflightInput,
    pub(super) started: Instant,
}

/// Completion held until earlier request calls are applied.
pub(super) struct PendingCompletion {
    pub(super) record: uniserve_worker_ipc::RequestOutput,
    /// Claimed media storage stays alive while its result waits or is discarded.
    pub(super) media: Option<Arc<SharedMedia>>,
    pub(super) arrival_seq: u64,
}

/// Terminal event held until outstanding calls are reconciled.
pub(super) struct PendingFinish {
    pub(super) reason: FinishReason,
    pub(super) stop_reason: Option<uniserve_core::StopReason>,
}

/// One submitted batch remains owned until all results and command receipts arrive.
/// Partial results consume identities without releasing the batch's queue credit.
pub(super) struct PendingBatch {
    pub(super) started: Instant,
    pub(super) calls: HashSet<(RequestKey, CallId)>,
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

    /// Whether the request still owns a submitted call.
    pub(super) fn has_pending_calls(&self, id: RequestId) -> bool {
        self.pending_calls
            .get(&id)
            .is_some_and(|queue| !queue.is_empty())
    }

    /// Number of submitted calls still owned by the request.
    pub(super) fn num_pending_calls(&self, id: RequestId) -> usize {
        self.pending_calls.get(&id).map_or(0, VecDeque::len)
    }

    /// Returns whether any in-flight call performs denoising.
    pub(super) fn has_pending_denoising(&self) -> bool {
        self.pending_calls
            .values()
            .flatten()
            .any(|op| op.call.code == CallKind::Pipeline(PipelineStage::Denoising))
    }

    /// Stateful calls retain request order. Pure media branches complete
    /// independently once their actual input producers have resolved.
    pub(super) fn take_ready_completions(&mut self) -> Vec<PendingCompletion> {
        let mut ready = Vec::new();
        for (id, pending) in &self.pending_completions {
            let Some(queue) = self.pending_calls.get(id) else {
                continue;
            };
            for (index, inflight) in queue.iter().enumerate() {
                let call = &inflight.call;
                let independent =
                    matches!(inflight.input, InflightInput::Media { .. }) && !call.advances_state();
                if index > 0 && !independent {
                    continue;
                }
                if independent
                    && call.tensor_inputs().any(|input| {
                        queue
                            .iter()
                            .any(|producer| producer.call.call_id == input.producer_call_id)
                    })
                {
                    continue;
                }
                if let Some(completion) = pending.get(&call.call_id) {
                    ready.push((
                        completion_priority(call.code),
                        completion.arrival_seq,
                        *id,
                        call.call_id,
                    ));
                }
            }
        }
        ready.sort_unstable_by_key(|(priority, arrival, ..)| (*priority, *arrival));
        let mut completions = Vec::with_capacity(ready.len());
        for (_, _, id, call_id) in ready {
            let pending = self
                .pending_completions
                .get_mut(&id)
                .expect("selected completion exists");
            completions.push(pending.remove(&call_id).expect("selected call exists"));
            if pending.is_empty() {
                self.pending_completions.remove(&id);
            }
        }
        completions
    }

    /// Removes a selected call and releases its transfer reservation.
    pub(super) fn pop_pending_call(
        &mut self,
        request_key: RequestKey,
        call_id: CallId,
    ) -> Option<InflightOp> {
        let id = request_key.request_id;
        let queue = self.pending_calls.get_mut(&id)?;
        let index = queue.iter().position(|inflight| {
            inflight.call.request_key == request_key && inflight.call.call_id == call_id
        })?;
        let selected = &queue[index];
        if call_id.batch_id == 0
            || (index > 0
                && !(matches!(selected.input, InflightInput::Media { .. })
                    && !selected.call.advances_state()))
        {
            return None;
        }
        let inflight = queue.remove(index).expect("selected call exists");
        if inflight.call.bounds.max_transfer_bytes > 0 {
            self.num_pending_transfers = self
                .num_pending_transfers
                .checked_sub(1)
                .expect("completed transfer owns a reservation");
        }
        if queue.is_empty() {
            self.pending_calls.remove(&id);
        }
        Some(inflight)
    }

    /// Removes failed calls from the in-flight registry.
    pub(super) fn retire_call(
        &mut self,
        batch_id: u64,
        request: RequestKey,
        op: CallId,
    ) -> Option<InflightOp> {
        if !self
            .pending_batches
            .get_mut(&batch_id)?
            .calls
            .remove(&(request, op))
        {
            return None;
        }
        let queue = self.pending_calls.get_mut(&request.request_id)?;
        let position = queue.iter().position(|inflight| {
            inflight.call.request_key == request && inflight.call.call_id == op
        })?;
        let inflight = queue.remove(position)?;
        if queue.is_empty() {
            self.pending_calls.remove(&request.request_id);
        }
        if inflight.call.bounds.max_transfer_bytes > 0 {
            self.num_pending_transfers = self
                .num_pending_transfers
                .checked_sub(1)
                .expect("retired transfer owns a reservation");
        }
        Some(inflight)
    }

    /// Removes failed calls from the in-flight registry.
    pub(super) fn clear_failed_calls(&mut self) -> (Vec<RequestId>, Vec<BatchCommand>) {
        let ids = self.pending_calls.keys().copied().collect();
        let commands = self
            .pending_batches
            .drain()
            .flat_map(|(_, batch)| batch.commands)
            .collect();
        self.pending_calls.clear();
        self.num_pending_transfers = 0;
        self.pending_completions.clear();
        (ids, commands)
    }
}
