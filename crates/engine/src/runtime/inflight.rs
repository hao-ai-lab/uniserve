//! EngineLoop-owned execution window and request-local completion ordering.

use super::*;

pub(super) enum InflightApply {
    Generation(RuntimeApply),
    Media(MediaCursor),
}

pub(super) struct InflightOp {
    pub(super) operation: Operation,
    pub(super) apply: InflightApply,
    pub(super) started: Instant,
}

impl InflightOp {
    pub(super) fn generation_apply(&self) -> &RuntimeApply {
        match &self.apply {
            InflightApply::Generation(apply) => apply,
            InflightApply::Media(_) => unreachable!("media operation entered generation planning"),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct SubmittedRunAccounting {
    pub(super) domain: uniserve_worker_ipc::Domain,
    pub(super) mixed: bool,
    pub(super) submission_group: u32,
    pub(super) operation_count: usize,
}

pub(super) struct PendingCompletion {
    pub(super) record: ModelOutput,
    pub(super) products: Arc<[ProductPayload]>,
    pub(super) arrival_seq: u64,
}

pub(super) struct PendingFinish {
    pub(super) reason: FinishReason,
    pub(super) stop_reason: Option<uniserve_core::StopReason>,
}

pub(super) struct InflightWindow {
    pub(super) transfer_capacity: usize,
    pub(super) inflight_transfers: usize,
    batch_id: u64,
    next_arrival_seq: u64,
    pub(super) operations: HashMap<RequestId, VecDeque<InflightOp>>,
    pub(super) completions: HashMap<RequestId, BTreeMap<u64, PendingCompletion>>,
    pub(super) finishes: HashMap<RequestId, PendingFinish>,
    pub(super) batch_started: HashMap<u64, Instant>,
    pub(super) prefill_steps: HashSet<u64>,
    pub(super) batch_operations: HashMap<u64, HashSet<(RequestKey, OpId)>>,
    pub(super) batch_group_worker_exec_us: HashMap<u64, HashMap<u32, u64>>,
    pub(super) command_batches: HashMap<u64, Vec<BatchCommand>>,
}

impl InflightWindow {
    pub(super) fn new(transfer_capacity: usize) -> Self {
        Self {
            transfer_capacity,
            inflight_transfers: 0,
            batch_id: 0,
            next_arrival_seq: 1,
            operations: HashMap::new(),
            completions: HashMap::new(),
            finishes: HashMap::new(),
            batch_started: HashMap::new(),
            prefill_steps: HashSet::new(),
            batch_operations: HashMap::new(),
            batch_group_worker_exec_us: HashMap::new(),
            command_batches: HashMap::new(),
        }
    }

    pub(super) fn next_batch_id(&mut self) -> u64 {
        self.batch_id = self.batch_id.saturating_add(1);
        self.batch_id
    }

    pub(super) fn next_arrival(&mut self) -> u64 {
        let arrival = self.next_arrival_seq;
        self.next_arrival_seq = self.next_arrival_seq.saturating_add(1);
        arrival
    }

    pub(super) fn contains(&self, id: RequestId) -> bool {
        self.operations
            .get(&id)
            .is_some_and(|queue| !queue.is_empty())
    }

    pub(super) fn len(&self, id: RequestId) -> usize {
        self.operations.get(&id).map_or(0, VecDeque::len)
    }

    pub(super) fn any_denoise(&self) -> bool {
        self.operations
            .values()
            .flatten()
            .any(|op| op.operation.kind == RunKind::DiffusionStep)
    }

    pub(super) fn take_ready(&mut self) -> Vec<PendingCompletion> {
        let mut ready = self
            .completions
            .iter()
            .filter_map(|(id, pending)| {
                let inflight = self.operations.get(id)?.front()?;
                let op_id = inflight.operation.op_id.0;
                let completion = pending.get(&op_id)?;
                Some((
                    completion_priority(inflight.operation.kind),
                    completion.arrival_seq,
                    *id,
                    op_id,
                ))
            })
            .collect::<Vec<_>>();
        ready.sort_unstable_by_key(|(priority, arrival_seq, ..)| (*priority, *arrival_seq));
        let mut completions = Vec::with_capacity(ready.len());
        for (_, _, id, op_id) in ready {
            let Some(pending) = self.completions.get_mut(&id) else {
                continue;
            };
            if let Some(completion) = pending.remove(&op_id) {
                completions.push(completion);
            }
            if pending.is_empty() {
                self.completions.remove(&id);
            }
        }
        completions
    }

    pub(super) fn pop(&mut self, request_key: RequestKey, op_id: u64) -> Option<InflightOp> {
        let id = request_key.request_id;
        let queue = self.operations.get_mut(&id)?;
        if op_id == 0
            || queue.front().is_none_or(|inflight| {
                inflight.operation.request_key != request_key || inflight.operation.op_id.0 != op_id
            })
        {
            return None;
        }
        let inflight = queue.pop_front().expect("front checked above");
        let empty = queue.is_empty();
        if inflight.operation.bounds().max_transfer_bytes > 0 {
            self.inflight_transfers = self
                .inflight_transfers
                .checked_sub(1)
                .expect("completed transfer operation owns one reservation");
        }
        if empty {
            self.operations.remove(&id);
        }
        Some(inflight)
    }

    pub(super) fn clear_failed(&mut self) -> (Vec<RequestId>, Vec<BatchCommand>) {
        let ids = self.operations.keys().copied().collect();
        let commands = self
            .command_batches
            .drain()
            .flat_map(|(_, commands)| commands)
            .collect();
        self.operations.clear();
        self.inflight_transfers = 0;
        self.completions.clear();
        self.batch_started.clear();
        self.prefill_steps.clear();
        self.batch_operations.clear();
        self.batch_group_worker_exec_us.clear();
        (ids, commands)
    }
}
