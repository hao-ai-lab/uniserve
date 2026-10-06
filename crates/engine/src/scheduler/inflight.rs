//! Scheduler-owned pending batches and request-local completion ordering.
//!
//! [`Inflight`] owns everything between planning a call and applying its
//! result: batches awaiting submission, submitted batches awaiting their
//! results and command receipts, each request's queue of submitted calls,
//! completions queued until their call may apply, deferred terminal events,
//! lifecycle commands awaiting a batch, and the transfer reservation count.
//!
//! Results arrive in executor order, but a request's calls apply in submission
//! order: a queued completion is released only when its call is at the front
//! of the request's queue. The exception is an independent call, one with
//! `InflightInput::Media` that does not advance state, which applies as soon
//! as the producers of its tensor inputs have left the queue.

use super::*;
use uniserve_worker_ipc::MediaCall;

/// Algorithm inputs retained until their computation completes.
pub(super) enum InflightInput {
    Generation {
        /// Physical KV start and exact token contribution, when known, from the
        /// submitted image-extension input. The call carries its capacity.
        image_kv: Option<(u32, Option<u32>)>,
        /// The submitted latent interval; cancellation can stop accepted
        /// progress while later submitted intervals still need validation.
        latent: Option<LatentParams>,
    },
    Media {
        latent: Option<LatentParams>,
        decode: Option<DecodeRange>,
    },
}

impl InflightInput {
    /// Returns the latent interval a submitted call covers.
    pub(super) const fn latent(&self) -> Option<&LatentParams> {
        match self {
            Self::Generation { latent, .. } | Self::Media { latent, .. } => latent.as_ref(),
        }
    }
}

/// Submitted call awaiting completion, queued per request in submission
/// order.
pub(super) struct InflightCall {
    pub(super) call: Call,
    pub(super) input: InflightInput,
}

/// Completion held until earlier request calls are applied.
pub(super) struct PendingCompletion {
    pub(super) record: uniserve_worker_ipc::RequestOutput,
    /// Claimed media storage stays alive while its result waits or is discarded.
    pub(super) media: Option<Arc<SharedMedia>>,
    /// Engine-wide arrival order (`Inflight::next_arrival`), used to order
    /// ready completions of equal priority.
    pub(super) arrival_seq: u64,
}

/// A request finish whose retirement waits until outstanding calls are
/// reconciled.
#[derive(Clone)]
pub(super) struct PendingFinish {
    pub(super) reason: FinishReason,
    pub(super) stop_reason: Option<uniserve_core::StopReason>,
    /// Whether the terminal `Finished` event is already published: a finish
    /// decided by the request's own output is final once no decoder decision
    /// is pending, since the calls still in flight are discarded
    /// (`Scheduler::finish_after_inflight`). Retirement then publishes no
    /// second terminal event, and the request publishes nothing more.
    pub(super) terminal_published: bool,
}

/// One submitted batch remains owned until all results and command receipts arrive.
/// Partial results consume identities without releasing the batch's queue credit.
pub(super) struct PendingBatch {
    /// Time the scheduling pass planned the batch; the batch round trip is
    /// measured from it.
    pub(super) started: Instant,
    /// Call identities whose results have not arrived. Result validation
    /// removes each identity once and treats a repeated or unknown one as
    /// invalid; the final result must leave this set empty.
    pub(super) calls: HashSet<(RequestKey, CallId)>,
    /// Lifecycle commands other than `Start`, in batch order, so a position
    /// here is the executor's `CommandResult::command_index`.
    pub(super) commands: Vec<BatchCommand>,
    /// Worker execution time accumulated across partial results, in
    /// microseconds.
    pub(super) worker_exec_us: u64,
    /// Whether the batch carries a prefill-lane call; such batches count
    /// against `PREFILL_WINDOW_CREDITS`.
    pub(super) prefill: bool,
}

/// Owns submitted identities and their lifetime through ordered reconciliation.
pub(super) struct Inflight {
    /// Planned batches the executor has not accepted yet, in dispatch order.
    pub(super) pending_submissions: VecDeque<ExecutionBatch>,
    /// Transfer calls (nonzero `max_transfer_bytes`) holding a reservation,
    /// bounded by `Scheduler::transfer_capacity`.
    pub(super) num_pending_transfers: usize,
    /// Last issued batch id; ids start at 1, so a call id with batch zero has
    /// not been assigned to a batch.
    pub(super) batch_id: u64,
    /// Next arrival sequence number, issued by `next_arrival`.
    pub(super) next_arrival_seq: u64,
    /// Submitted calls per request, in submission order.
    pub(super) pending_calls: HashMap<RequestId, VecDeque<InflightCall>>,
    /// Results queued until `take_ready_completions` releases them.
    pub(super) pending_completions: HashMap<RequestId, BTreeMap<CallId, PendingCompletion>>,
    /// Terminal events deferred until the request's calls have drained.
    pub(super) pending_finishes: HashMap<RequestId, PendingFinish>,
    /// Batches by id, from registration at planning until their final result;
    /// includes batches still waiting in `pending_submissions`.
    pub(super) pending_batches: HashMap<u64, PendingBatch>,
    /// `Finish` and `Free` commands awaiting a batch to carry them.
    pub(super) pending_commands: VecDeque<BatchCommand>,
}

impl Inflight {
    pub(super) fn new() -> Self {
        Self {
            pending_submissions: VecDeque::new(),
            num_pending_transfers: 0,
            batch_id: 0,
            next_arrival_seq: 1,
            pending_calls: HashMap::new(),
            pending_completions: HashMap::new(),
            pending_finishes: HashMap::new(),
            pending_batches: HashMap::new(),
            pending_commands: VecDeque::new(),
        }
    }

    /// Registers the call identities and command receipts owned by one
    /// scheduled batch.
    ///
    /// `Start` admissions are left out of the receipts because the executor
    /// numbers command results without them.
    pub(super) fn register_pending_batch(&mut self, batch: &ExecutionBatch, started: Instant) {
        self.pending_batches.insert(
            batch.id,
            PendingBatch {
                started,
                calls: batch
                    .requests
                    .iter()
                    .map(|(call, _)| (call.request_key, call.call_id))
                    .collect(),
                commands: batch
                    .commands
                    .iter()
                    .filter(|command| !matches!(command, BatchCommand::Start { .. }))
                    .cloned()
                    .collect(),
                worker_exec_us: 0,
                prefill: batch
                    .requests
                    .iter()
                    .any(|(call, _)| batch_kind(call.code) == BatchKind::Prefill),
            },
        );
    }

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

    /// Returns whether a request other than `id` holds the denoiser lane,
    /// that is, has a denoising call in flight.
    pub(super) fn denoiser_lane_held_by_other(&self, id: RequestId) -> bool {
        self.pending_calls
            .iter()
            .filter(|(holder, _)| **holder != id)
            .flat_map(|(_, calls)| calls)
            .any(|call| call.call.code == CallKind::Media(MediaCall::Denoising))
    }

    /// Stateful calls retain request order. Pure media branches complete
    /// independently once their actual input producers have resolved.
    ///
    /// Removes and returns the queued completions that may apply now, ordered
    /// by `completion_priority` and then arrival. A call is ready when it is at
    /// the front of its request's queue, or when it is independent (an
    /// `InflightInput::Media` call that does not advance state) and none of its
    /// tensor inputs is produced by a call still queued. Applying a released
    /// completion can make the next one ready, so the caller repeats until
    /// nothing is returned. Completions of a request with no queued call stay
    /// queued.
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

        // Selection read these completions from the maps it now drains, so
        // each one is still present.
        let mut completions = Vec::with_capacity(ready.len());
        for (_, _, id, call_id) in ready {
            let Some(pending) = self.pending_completions.get_mut(&id) else {
                continue;
            };
            if let Some(completion) = pending.remove(&call_id) {
                completions.push(completion);
            }
            if pending.is_empty() {
                self.pending_completions.remove(&id);
            }
        }
        completions
    }

    /// Removes a selected call and releases its transfer reservation.
    ///
    /// The call must obey the ordering of `take_ready_completions`: it is at
    /// the front of its request's queue or is independent. Returns `None`,
    /// changing nothing, for a call id with batch zero, an unknown request or
    /// call, or a call that is not yet eligible. The batch's
    /// `PendingBatch::calls` entry is not touched here; result validation
    /// removed it when the result arrived.
    pub(super) fn pop_pending_call(
        &mut self,
        request_key: RequestKey,
        call_id: CallId,
    ) -> Option<InflightCall> {
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
        let inflight = queue.remove(index)?;
        if queue.is_empty() {
            self.pending_calls.remove(&id);
        }
        if inflight.call.bounds.max_transfer_bytes > 0 {
            self.release_transfer();
        }
        Some(inflight)
    }

    /// Removes one call a worker failure retired, regardless of its queue
    /// position, and releases its transfer reservation.
    ///
    /// Returns `None` when the batch is unknown or no longer owns the call, or
    /// when the request's queue does not hold it; the worker-failure path
    /// treats that as engine-fatal. Once the call has left the batch's
    /// `calls`, a missing queue entry is reported without restoring it.
    pub(super) fn retire_call(
        &mut self,
        batch_id: u64,
        request: RequestKey,
        call_id: CallId,
    ) -> Option<InflightCall> {
        if !self
            .pending_batches
            .get_mut(&batch_id)?
            .calls
            .remove(&(request, call_id))
        {
            return None;
        }
        let queue = self.pending_calls.get_mut(&request.request_id)?;
        let position = queue.iter().position(|inflight| {
            inflight.call.request_key == request && inflight.call.call_id == call_id
        })?;
        let inflight = queue.remove(position)?;
        if queue.is_empty() {
            self.pending_calls.remove(&request.request_id);
        }
        if inflight.call.bounds.max_transfer_bytes > 0 {
            self.release_transfer();
        }
        Some(inflight)
    }

    /// Releases the transfer reservation a leaving transfer call holds.
    ///
    /// Every registered transfer call holds one reservation, so a release at
    /// zero reports an accounting defect instead of wrapping the count.
    fn release_transfer(&mut self) {
        match self.num_pending_transfers.checked_sub(1) {
            Some(remaining) => self.num_pending_transfers = remaining,
            None => tracing::error!("a transfer call released a reservation it did not hold"),
        }
    }

    /// Drops every submitted call after an unrecoverable execution failure.
    ///
    /// Clears the per-request call queues, queued completions, registered
    /// batches, and transfer reservations. Returns the ids of requests that
    /// owned a submitted call and the recorded commands of every dropped
    /// batch. `pending_submissions`, `pending_finishes`, and `pending_commands`
    /// are left for the caller.
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
