//! Submission of explicitly targeted calls to WorkerGroup instances.
//!
//! Each destination has independent bounded capacity. Ready work may bypass
//! unrelated blocked requests while preserving a request's command order.

use crate::executor::WorkerResult;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::Duration;

use super::{BatchSubmitError, WorkerGroup};
use crate::executor::{
    BatchResult, CommandOutcome, ExecutionBatch, Executor, ExecutorInfo, ExecutorSubmitError,
    RequestPlacement, TransferConfig, WorkerFailure, WorkerId, logical_result, physical_batch,
};
use anyhow::Context;
use uniserve_core::CommandWaker;
use uniserve_worker_ipc::{
    BatchCommand, BufferAllocation, BufferId, Call, KvTransfer, NewRequest, RequestKey,
    TensorPublication,
};

/// One call's expected worker and result family until its batch retires.
struct PendingCall {
    worker: usize,
    /// Whether the batch carrying this computation has been dispatched.
    submitted: bool,
    kind: uniserve_worker_ipc::CallKind,
    completed: bool,
}

/// Dispatch and receipts for one scheduler batch.
struct PendingBatch {
    /// Workers this batch is dispatched to and has not yet returned from.
    submitted_workers: HashSet<usize>,
    expected_workers: u64,
    calls: HashMap<(RequestKey, uniserve_worker_ipc::CallId), PendingCall>,
    commands: Vec<BatchCommand>,
    command_outcomes: HashMap<u32, CommandOutcome>,
}

impl PendingBatch {
    /// A failed release keeps ownership even when another endpoint retires it.
    fn set_command_outcome(&mut self, requests: &HashSet<RequestKey>, outcome: CommandOutcome) {
        for (index, command) in self.commands.iter().enumerate() {
            if requests.contains(&command.request_key()) {
                let current = self.command_outcomes.entry(index as u32).or_insert(outcome);
                if outcome == CommandOutcome::Failed {
                    *current = outcome;
                }
            }
        }
    }
}

#[derive(Clone)]
struct WorkerSubmission {
    batch: ExecutionBatch,
    dependencies: Vec<BufferId>,
}

impl WorkerSubmission {
    /// Builds the one submission a worker receives for this batch.
    ///
    /// A batch carries one computation to one component, so a worker receives
    /// it once, under one identity, and answers it once. Product inputs the
    /// worker does not yet hold travel as dependencies and hold the submission
    /// in its queue until they resolve.
    fn build(
        batch_id: u64,
        calls: Vec<(Call, RequestPlacement)>,
        commands: Vec<BatchCommand>,
        inputs: Vec<TensorPublication>,
        kv_inputs: Vec<KvTransfer>,
        dependencies: Vec<BufferId>,
    ) -> Self {
        let consumed = calls
            .iter()
            .flat_map(|(call, _)| call.input_buffers())
            .collect::<HashSet<_>>();
        let input_transfers = inputs
            .into_iter()
            .filter(|payload| consumed.contains(&payload.product.buffer_id()))
            .collect();
        let dependencies = dependencies
            .into_iter()
            .filter(|product| consumed.contains(product))
            .collect();
        let mut batch = ExecutionBatch::new(batch_id, calls, commands, input_transfers);
        batch.kv_inputs = kv_inputs
            .into_iter()
            .filter(|publication| consumed.contains(&publication.source))
            .collect();
        Self {
            batch,
            dependencies,
        }
    }
}

#[derive(Clone)]
struct BufferRoute {
    worker_index: usize,
    component: String,
}

/// Dispatch targeted calls and track their independently completed results.
pub struct WorkerExecutor {
    routing: HashMap<WorkerId, usize>,
    workers: Vec<(WorkerId, WorkerGroup)>,
    executor_info: ExecutorInfo,
    depth: usize,
    command_wake: crate::handle::WakeSignal,
    progress_fds: Vec<i32>,
    pending: BTreeMap<u64, PendingBatch>,
    ready: VecDeque<BatchResult>,
    admissions: HashMap<RequestKey, NewRequest>,
    admitted_workers: HashSet<(usize, RequestKey)>,
    buffer_routes: HashMap<BufferId, BufferRoute>,
    buffer_allocations: HashMap<BufferId, BufferAllocation>,
    buffer_workers: HashMap<BufferId, HashSet<usize>>,
    transfer_products: HashMap<BufferId, TensorPublication>,
    kv_transfers: HashMap<BufferId, KvTransfer>,
    worker_submissions: Vec<VecDeque<WorkerSubmission>>,
    worker_collective_seqs: Vec<u64>,
    command_wake_pending: bool,
    pending_failure: Option<WorkerFailure>,
    closed: bool,
}

impl WorkerExecutor {
    pub fn try_new(
        workers: Vec<(WorkerId, WorkerGroup)>,
        transfer: TransferConfig,
    ) -> anyhow::Result<Self> {
        let command_wake = crate::handle::WakeSignal::new()?;
        let progress_fds = std::iter::once(command_wake.descriptor())
            .chain(
                workers
                    .iter()
                    .flat_map(|(_, worker)| worker.progress_fds().iter().copied()),
            )
            .collect();
        anyhow::ensure!(
            !workers.is_empty(),
            "WorkerExecutor needs at least one worker"
        );
        anyhow::ensure!(
            workers.len() <= u64::BITS as usize,
            "WorkerExecutor supports at most {} workers",
            u64::BITS
        );

        let mut routing = HashMap::new();
        for (index, (id, executor)) in workers.iter().enumerate() {
            anyhow::ensure!(
                executor.transfer_config() == &transfer,
                "WorkerGroup {id} was initialized with different transfer bindings"
            );
            anyhow::ensure!(
                executor.info().endpoint.worker_id == id.0,
                "worker binding {id} disagrees with its loaded instance identity"
            );
            anyhow::ensure!(
                routing.insert(id.clone(), index).is_none(),
                "worker identity {id} is repeated"
            );
            executor
                .info()
                .validate()
                .with_context(|| format!("worker {id} reported invalid capabilities"))?;
        }
        let executor_info = ExecutorInfo::from_workers(
            workers
                .iter()
                .map(|worker| (worker.0.clone(), worker.1.info().clone()))
                .collect(),
        )?;
        let mut physical_edges = HashSet::new();
        for edge in &transfer.edges {
            anyhow::ensure!(
                routing.contains_key(&edge.source_worker)
                    && routing.contains_key(&edge.destination_worker),
                "transfer edge names an unbound WorkerGroup"
            );
            let source = &workers[routing[&edge.source_worker]].1;
            let destination = &workers[routing[&edge.destination_worker]].1;
            for (rank, worker) in [
                (edge.source_rank, source),
                (edge.destination_rank, destination),
            ] {
                anyhow::ensure!(
                    rank.is_none_or(|rank| worker.rank_info(rank as usize).is_some()),
                    "transfer edge names an unbound rank"
                );
            }
            for source_rank in 0..source.info().world_size {
                if edge.source_rank.is_some_and(|rank| rank != source_rank) {
                    continue;
                }
                let source = source
                    .rank_info(source_rank as usize)
                    .expect("validated source rank");
                for destination_rank in 0..destination.info().world_size {
                    if edge
                        .destination_rank
                        .is_some_and(|rank| rank != destination_rank)
                    {
                        continue;
                    }
                    let destination = destination
                        .rank_info(destination_rank as usize)
                        .expect("validated destination rank");
                    anyhow::ensure!(
                        physical_edges
                            .insert((source.endpoint.clone(), destination.endpoint.clone())),
                        "physical transfer edge has conflicting bindings"
                    );
                    // An edge carries device products on one mechanism and
                    // host products on another; each decides for itself how
                    // far it reaches.
                    for mechanism in edge.mechanisms() {
                        let backend = mechanism.as_str();
                        anyhow::ensure!(
                            source.transfer_backends.iter().any(|name| name == backend)
                                && destination
                                    .transfer_backends
                                    .iter()
                                    .any(|name| name == backend),
                            "physical edge requires an uninitialized transfer backend {backend}"
                        );
                        // Each mechanism decides for itself how far it reaches.
                        // An edge it cannot serve would fail on its first
                        // publication, so it is refused here by name instead.
                        let (source_host, destination_host) =
                            (&source.endpoint.node, &destination.endpoint.node);
                        let crosses_hosts = source_host != destination_host;
                        match mechanism {
                            crate::executor::TransferBackend::Local => anyhow::ensure!(
                                source.endpoint.address_space == destination.endpoint.address_space,
                                "local transfer requires a shared address space"
                            ),
                            crate::executor::TransferBackend::CudaVmm => {
                                anyhow::ensure!(
                                    source.device.starts_with("cuda:")
                                        && destination.device.starts_with("cuda:"),
                                    "CUDA VMM requires CUDA devices on both endpoints"
                                );
                                // A device handle crosses hosts only where the
                                // producing device exports a fabric handle and the
                                // consuming device can import one.
                                if crosses_hosts {
                                    let unreachable = if source.fabric_handles {
                                        destination_host
                                    } else {
                                        source_host
                                    };
                                    anyhow::ensure!(
                                        source.fabric_handles && destination.fabric_handles,
                                        "transfer edge from worker {} on host {source_host} to \
                                     worker {} on host {destination_host} crosses hosts, and \
                                     the device on host {unreachable} exports a process \
                                     descriptor the other host cannot import",
                                        edge.source_worker.0,
                                        edge.destination_worker.0,
                                    );
                                }
                            }
                            // Shared memory names a segment in one host's namespace.
                            crate::executor::TransferBackend::Shm => anyhow::ensure!(
                                !crosses_hosts,
                                "transfer edge from worker {} on host {source_host} to worker {} \
                             on host {destination_host} crosses hosts over shared memory, \
                             which names a segment in one host's namespace",
                                edge.source_worker.0,
                                edge.destination_worker.0,
                            ),
                            // A product on the rank channel reaches wherever the
                            // channel does, which is every host of the instance.
                            crate::executor::TransferBackend::Channel => {}
                        }
                    }
                }
            }
        }
        executor_info.runtime_info()?;
        let depth = workers
            .iter()
            .map(|worker| worker.1.info().queue_depth.max(1) as usize)
            .sum();
        let worker_count = workers.len();
        Ok(Self {
            routing,
            workers,
            executor_info,
            depth,
            command_wake,
            progress_fds,
            pending: BTreeMap::new(),
            ready: VecDeque::new(),
            admissions: HashMap::new(),
            admitted_workers: HashSet::new(),
            buffer_routes: HashMap::new(),
            buffer_allocations: HashMap::new(),
            buffer_workers: HashMap::new(),
            transfer_products: HashMap::new(),
            kv_transfers: HashMap::new(),
            worker_submissions: (0..worker_count).map(|_| VecDeque::new()).collect(),
            worker_collective_seqs: vec![0; worker_count],
            command_wake_pending: false,
            pending_failure: None,
            closed: false,
        })
    }

    /// Returns the mask bit assigned to a command worker.
    fn worker_bit(index: usize) -> u64 {
        1u64 << index
    }

    /// Refreshes physical discovery after replacement; other instance counters remain intact.
    fn refresh_worker(&mut self, index: usize) {
        self.executor_info.workers[index].1 = self.workers[index].1.info().clone();
        self.refresh_progress_fds();
    }

    fn refresh_progress_fds(&mut self) {
        self.progress_fds = std::iter::once(self.command_wake.descriptor())
            .chain(
                self.workers
                    .iter()
                    .flat_map(|(_, worker)| worker.progress_fds().iter().copied()),
            )
            .collect();
    }

    /// Reconciles abandoned work and its dependents while retaining live physical owners.
    fn reconcile_worker_failure(&mut self, error: anyhow::Error) -> anyhow::Result<WorkerFailure> {
        let mut loss = error.downcast::<WorkerFailure>()?;
        let index = *self
            .routing
            .get(&loss.worker_id)
            .context("failed WorkerGroup is not bound")?;
        let lost = !loss.endpoints.is_empty();
        if lost {
            self.refresh_worker(index);
        }
        let failed_run = (!lost)
            .then(|| loss.execution.as_ref().and_then(|error| error.batch_id))
            .flatten()
            .filter(|batch_id| {
                self.pending
                    .get(batch_id)
                    .is_some_and(|pending_batch| pending_batch.submitted_workers.contains(&index))
            });
        let mut retired = loss.retired.iter().copied().collect::<HashSet<_>>();
        let failed_calls = retired
            .iter()
            .map(|(_, request, call)| (*request, *call))
            .collect::<HashSet<_>>();
        let mut requests = loss.requests.iter().copied().collect::<HashSet<_>>();
        let mut buffers = loss.buffers.iter().cloned().collect::<HashSet<_>>();
        if lost {
            for (buffer, payload) in &mut self.transfer_products {
                let handle = &mut payload.value;
                let tensors = handle.tensors_mut();
                for tensor in tensors.iter_mut() {
                    tensor
                        .locations
                        .retain(|location| !loss.endpoints.contains(&location.source));
                }
                if tensors.iter().any(|tensor| !tensor.has_complete_coverage()) {
                    buffers.insert(*buffer);
                }
            }
        }
        if lost {
            for (buffer, publication) in &mut self.kv_transfers {
                for tensor in &mut publication.tensors {
                    tensor
                        .locations
                        .retain(|location| !loss.endpoints.contains(&location.source));
                }
                if publication
                    .tensors
                    .iter()
                    .any(|tensor| !tensor.has_complete_coverage())
                {
                    buffers.insert(*buffer);
                }
            }
        }
        for (buffer, route) in &self.buffer_routes {
            if (lost
                && route.worker_index == index
                && !self.transfer_products.contains_key(buffer)
                && !self.kv_transfers.contains_key(buffer))
                || failed_calls.contains(&(buffer.owner, buffer.producer_call_id))
            {
                buffers.insert(*buffer);
            }
        }
        for pending_batch in self.pending.values() {
            requests.extend(
                pending_batch
                    .calls
                    .iter()
                    .filter_map(|((request, _), worker)| {
                        (lost && worker.worker == index).then_some(*request)
                    }),
            );
        }
        // Abandoning an unstarted producer also invalidates its future buffers.
        // Already running work on another instance keeps its real completion path.
        loop {
            let before = (requests.len(), buffers.len());
            for (_, worker) in &self.workers {
                for call in worker.inflight_calls() {
                    if call.input_buffers().any(|buffer| buffers.contains(&buffer)) {
                        requests.insert(call.request_key);
                    }
                }
            }
            for submission in self.worker_submissions.iter().flatten() {
                for (call, _) in &submission.batch.requests {
                    if call.input_buffers().any(|buffer| buffers.contains(&buffer)) {
                        requests.insert(call.request_key);
                    }
                    if requests.contains(&call.request_key) {
                        buffers.extend(call.output_buffers());
                    }
                }
            }
            if before == (requests.len(), buffers.len()) {
                break;
            }
        }
        for (worker_index, queue) in self.worker_submissions.iter_mut().enumerate() {
            let mut active = VecDeque::new();
            for mut submission in queue.drain(..) {
                let batch_id = submission.batch.id;
                for (call, _) in submission.batch.retire_requests(&requests) {
                    retired.insert((batch_id, call.request_key, call.call_id));
                }
                let inputs = submission
                    .batch
                    .requests
                    .iter()
                    .flat_map(|(op, _)| op.input_buffers())
                    .collect::<HashSet<_>>();
                submission
                    .dependencies
                    .retain(|product| inputs.contains(product));
                if lost && worker_index == index {
                    submission.batch.commands.clear();
                }
                if !submission.batch.requests.is_empty() || !submission.batch.commands.is_empty() {
                    active.push_back(submission);
                }
            }
            *queue = active;
        }
        for (batch_id, pending_batch) in &mut self.pending {
            if lost || failed_run == Some(*batch_id) {
                pending_batch.submitted_workers.remove(&index);
            }
            retired.extend(pending_batch.calls.iter().filter_map(|(identity, worker)| {
                ((lost || (worker.submitted && failed_run == Some(*batch_id)))
                    && worker.worker == index
                    && !worker.completed)
                    .then_some((*batch_id, identity.0, identity.1))
            }));
            for worker_index in 0..self.workers.len() {
                let running = pending_batch.submitted_workers.contains(&worker_index);
                let queued = self.worker_submissions[worker_index]
                    .iter()
                    .any(|submission| submission.batch.id == *batch_id);
                if !running && !queued {
                    pending_batch.expected_workers &= !Self::worker_bit(worker_index);
                }
            }
            pending_batch.set_command_outcome(&requests, CommandOutcome::Retired);
        }
        if let Some(batch_id) = failed_run {
            let failed_requests = loss.requests.iter().copied().collect();
            if let Some(pending) = self.pending.get_mut(&batch_id) {
                pending.set_command_outcome(&failed_requests, CommandOutcome::Failed);
            }
        }
        loss.retired.clear();
        for (batch_id, request, call) in retired {
            let pending_batch = self
                .pending
                .get_mut(&batch_id)
                .context("retired call has no pending batch")?;
            anyhow::ensure!(
                pending_batch
                    .calls
                    .get(&(request, call))
                    .is_some_and(|call| !call.completed),
                "completed call cannot be abandoned"
            );
            anyhow::ensure!(
                pending_batch.calls.remove(&(request, call)).is_some(),
                "abandoned call is not pending"
            );
            loss.retired.push((batch_id, request, call));
        }
        if lost {
            self.admitted_workers.retain(|(worker, _)| *worker != index);
            for workers in self.buffer_workers.values_mut() {
                workers.remove(&index);
            }
        }
        let batches = self.pending.keys().copied().collect::<Vec<_>>();
        for batch_id in batches {
            if self
                .pending
                .get(&batch_id)
                .is_some_and(|pending| pending.expected_workers == 0)
            {
                self.publish_result(WorkerResult {
                    batch_id,
                    results: Vec::new(),
                    products: Vec::new(),
                    registration: uniserve_worker_ipc::RegistrationAck { visible: true },
                    worker_exec_us: None,
                    forward_stats: None,
                })?;
            }
        }
        loss.requests = requests.into_iter().collect();
        loss.buffers = buffers.into_iter().collect();
        Ok(loss)
    }

    /// Returns the command workers eligible for cache admission.
    fn cache_admissions(&mut self, admissions: &[NewRequest]) -> anyhow::Result<()> {
        for admission in admissions {
            if let Some(existing) = self.admissions.get(&admission.request_key) {
                anyhow::ensure!(
                    existing == admission,
                    "request {:?} was readmitted with a different descriptor before drop",
                    admission.request_key
                );
            } else {
                self.admissions
                    .insert(admission.request_key, admission.clone());
            }
        }
        Ok(())
    }

    /// Partitions new-request admissions by the workers used by their first calls.
    fn admissions_for(
        &self,
        worker_index: usize,
        calls: &[(Call, RequestPlacement)],
    ) -> anyhow::Result<Vec<NewRequest>> {
        let mut admissions = Vec::new();
        for (call, _) in calls {
            if self
                .admitted_workers
                .contains(&(worker_index, call.request_key))
            {
                continue;
            }
            let admission = self.admissions.get(&call.request_key).ok_or_else(|| {
                anyhow::anyhow!(
                    "worker {worker_index} has no admission for request {:?}",
                    call.request_key
                )
            })?;
            admissions.push(admission.clone());
        }
        Ok(admissions)
    }

    /// Includes every admitted or queued owner of this request's computation and products.
    fn request_workers(&self, request: RequestKey) -> Vec<usize> {
        let mut owners = self
            .admitted_workers
            .iter()
            .filter_map(|(worker, key)| (*key == request).then_some(*worker))
            .chain(
                self.pending
                    .values()
                    .flat_map(|pending_batch| pending_batch.calls.iter())
                    .filter_map(|((key, _), call)| (*key == request).then_some(call.worker)),
            )
            .chain(
                self.buffer_workers
                    .iter()
                    .filter(|(buffer, _)| buffer.owner == request)
                    .flat_map(|(_, workers)| workers.iter().copied()),
            )
            .collect::<Vec<_>>();
        owners.sort_unstable();
        owners.dedup();
        owners
    }

    /// Routes buffer release to its storage owners and request closure to every physical owner.
    fn command_workers(&self, command: &BatchCommand) -> Vec<usize> {
        match command {
            BatchCommand::Start { .. } => Vec::new(),
            BatchCommand::Free { buffer } => {
                let mut owners = self
                    .buffer_workers
                    .get(buffer)
                    .map(|workers| workers.iter().copied().collect::<Vec<_>>())
                    .unwrap_or_default();
                owners.sort_unstable();
                owners
            }
            BatchCommand::Finish { request_key, .. } => self.request_workers(*request_key),
        }
    }

    /// Keep resident references direct when producer and consumer execute in
    /// the same address spaces. Different multi-rank components require published
    /// layouts, including when both components belong to this WorkerGroup.
    fn shares_product_storage(&self, producer: &BufferRoute, consumer_entry: &str) -> bool {
        let entries = &self.workers[producer.worker_index].1.info().components;
        let source = entries
            .iter()
            .find(|entry| entry.name == producer.component);
        let destination = entries.iter().find(|entry| entry.name == consumer_entry);
        if producer.component == consumer_entry {
            // A consumer call on the component that produced the product is aligned
            // with the round that produced it: the same media units in the same
            // order on the same ranks, so each rank consumes the shard it wrote
            // and no rank needs another's.
            return true;
        }
        source
            .zip(destination)
            .is_some_and(|(source, destination)| {
                source.config.ranks.len() == 1 && source.config.ranks == destination.config.ranks
            })
    }

    /// Submits or queues one worker-local batch while preserving collective order.
    fn submit_worker(
        &mut self,
        worker_index: usize,
        submission: &WorkerSubmission,
    ) -> anyhow::Result<bool> {
        let batch = &submission.batch;
        let admissions = self.admissions_for(worker_index, &batch.requests)?;
        let request_keys = admissions
            .iter()
            .map(|admission| admission.request_key)
            .collect::<Vec<_>>();
        let commands = admissions
            .into_iter()
            .map(|request| BatchCommand::Start { request })
            .chain(batch.commands.iter().cloned())
            .collect();
        let mut inputs = batch.input_transfers.clone();
        let mut kv_inputs = batch.kv_inputs.clone();
        for dependency in &submission.dependencies {
            if let Some(publication) = self.kv_transfers.get(dependency) {
                anyhow::ensure!(
                    !kv_inputs.iter().any(|input| input.source == *dependency),
                    "KV input is supplied more than once"
                );
                kv_inputs.push(publication.clone());
                continue;
            }
            anyhow::ensure!(
                !inputs
                    .iter()
                    .any(|payload| payload.product.buffer_id() == *dependency),
                "transferred input payload is supplied more than once"
            );
            let payload = self.transfer_products.get(dependency).ok_or_else(|| {
                anyhow::anyhow!(
                    "ready transferred input {:?} lost its transfer descriptor",
                    dependency
                )
            })?;
            inputs.push(payload.clone());
        }
        let collective_seq = self.worker_collective_seqs[worker_index]
            .checked_add(1)
            .context("collective sequence space exhausted")?;
        let mut wire = physical_batch(
            batch.id,
            collective_seq,
            batch.requests.clone(),
            commands,
            inputs,
            kv_inputs,
        )?;
        for dependency in &submission.dependencies {
            if !self.transfer_products.contains_key(dependency)
                || !wire
                    .calls
                    .iter()
                    .flat_map(|call| call.buffer_inputs())
                    .any(|tensor| tensor.buffer_id() == *dependency)
            {
                continue;
            }
            let params = self
                .buffer_allocations
                .get(dependency)
                .copied()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "persistent dependency {:?} has no scheduler params",
                        dependency
                    )
                })?;
            if !wire
                .buffer_allocations
                .iter()
                .any(|existing| existing.buffer == params.buffer)
            {
                wire.buffer_allocations.push(params);
            }
        }
        wire.validate()?;
        let tensors = wire
            .input_products
            .iter_mut()
            .map(|payload| &mut payload.value)
            .flat_map(|handle| handle.tensors_mut())
            .chain(
                wire.kv_inputs
                    .iter_mut()
                    .flat_map(|publication| &mut publication.tensors),
            );
        for tensor in tensors {
            tensor.locations.retain(|location| {
                self.routing
                    .get(&WorkerId(location.source.worker_id.clone()))
                    .and_then(|index| {
                        self.workers[*index]
                            .1
                            .rank_info(location.source.rank as usize)
                    })
                    .is_some_and(|owner| owner.endpoint == location.source)
            });
        }

        match self.workers[worker_index].1.submit_batch(wire) {
            Ok(()) => {}
            Err(BatchSubmitError::WouldBlock(_)) => return Ok(false),
            Err(BatchSubmitError::Failed(error)) => return Err(error),
        }
        self.worker_collective_seqs[worker_index] = collective_seq;
        let pending = self
            .pending
            .get_mut(&batch.id)
            .context("submitted batch is not pending")?;
        pending.submitted_workers.insert(worker_index);
        for (call, _) in &batch.requests {
            pending
                .calls
                .get_mut(&(call.request_key, call.call_id))
                .context("submitted computation is not pending")?
                .submitted = true;
        }
        self.admitted_workers
            .extend(request_keys.into_iter().map(|key| (worker_index, key)));
        Ok(true)
    }

    /// Dispatches each worker's queued runs in order, as far as they are ready.
    ///
    /// Batch ids are dispatched in order, so a rank sees a strictly increasing
    /// subsequence and cooperative ranks launch their collectives in the same
    /// order. A queue therefore holds at its head: a submission whose product
    /// dependencies are not yet published cannot have their locators bound, and
    /// dispatching the submission behind it would hand the rank a batch out of
    /// order, which a rank that launches strictly in channel order refuses.
    fn dispatch_ready(&mut self) -> anyhow::Result<()> {
        loop {
            let mut progressed = false;
            for worker_index in 0..self.workers.len() {
                let Some(head) = self.worker_submissions[worker_index].front() else {
                    continue;
                };
                let ready = head.dependencies.iter().all(|buffer| {
                    self.transfer_products.contains_key(buffer)
                        || self.kv_transfers.contains_key(buffer)
                });
                if !ready {
                    continue;
                }
                let submission = self.worker_submissions[worker_index]
                    .pop_front()
                    .ok_or_else(|| anyhow::anyhow!("ready worker submission disappeared"))?;
                if !self.submit_worker(worker_index, &submission)? {
                    self.worker_submissions[worker_index].push_front(submission);
                    continue;
                }
                progressed = true;
            }
            if !progressed {
                return Ok(());
            }
        }
    }

    /// Advances worker command execution until no immediate progress remains.
    fn pump(&mut self) -> anyhow::Result<()> {
        self.command_wake_pending |= self.command_wake.drain()?;
        self.dispatch_ready()?;
        for worker_index in 0..self.workers.len() {
            loop {
                let polled = self.workers[worker_index].1.poll_batch(Duration::ZERO);
                if self.workers[worker_index].1.take_readiness_change() {
                    self.refresh_worker(worker_index);
                }
                let report = match polled {
                    Ok(report) => report,
                    Err(error) => return Err(error),
                };
                if self.workers[worker_index].1.take_command_wake() {
                    self.command_wake_pending = true;
                }
                let Some(report) = report else {
                    break;
                };
                self.route_result(worker_index, report)?;
            }
        }
        self.dispatch_ready()
    }

    /// Routes a worker result into its worker aggregate and publishes transferable products.
    fn route_result(&mut self, worker_index: usize, report: WorkerResult) -> anyhow::Result<()> {
        let mut report = report;
        let failed_calls = report
            .results
            .iter()
            .filter(|completion| completion.output.status == uniserve_worker_ipc::CallStatus::Error)
            .map(|completion| (completion.output.request_key, completion.output.call_id))
            .collect::<HashSet<_>>();
        let failed_buffers = self
            .buffer_routes
            .keys()
            .filter(|product| failed_calls.contains(&(product.owner, product.producer_call_id)))
            .cloned()
            .collect();
        let batch_id = report.batch_id;
        let worker_bit = Self::worker_bit(worker_index);
        {
            let pending_batch = self
                .pending
                .get_mut(&batch_id)
                .ok_or_else(|| anyhow::anyhow!("worker returned unknown batch {batch_id}"))?;
            anyhow::ensure!(
                pending_batch.submitted_workers.remove(&worker_index),
                "worker returned a batch it was not executing"
            );
            anyhow::ensure!(
                pending_batch.expected_workers & worker_bit != 0,
                "worker {worker_index} returned a duplicate or unexpected result for batch {batch_id}"
            );
            for completion in &report.results {
                let identity = (completion.output.request_key, completion.output.call_id);
                let call = pending_batch
                    .calls
                    .get_mut(&identity)
                    .context("worker returned an unknown call")?;
                anyhow::ensure!(
                    call.worker == worker_index && call.submitted && !call.completed,
                    "worker {worker_index} returned a duplicate or unexpected call for batch {batch_id}"
                );
                anyhow::ensure!(
                    completion.output.code == call.kind,
                    "worker returned a computation code that disagrees with its call"
                );
                call.completed = true;
            }
            // A batch returns one result, so its every call completes with it.
            anyhow::ensure!(
                pending_batch
                    .calls
                    .values()
                    .all(|call| call.worker != worker_index || !call.submitted || call.completed),
                "worker {worker_index} ended batch {batch_id} before every call completed"
            );
            let queued = self.worker_submissions[worker_index]
                .iter()
                .any(|submission| submission.batch.id == batch_id);
            if !queued {
                pending_batch.expected_workers &= !worker_bit;
            }
        }
        for product in report.products.drain(..) {
            let route = self
                .buffer_routes
                .get(&product.product.buffer_id())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "worker {worker_index} returned an unplanned cross-worker product {:?}",
                        product.product
                    )
                })?;
            anyhow::ensure!(
                route.worker_index == worker_index,
                "worker {worker_index} returned a cross-worker product owned by worker {}",
                route.worker_index
            );
            if let Some(existing) = self.transfer_products.get_mut(&product.product.buffer_id()) {
                existing.merge_locations(&product)?;
            } else {
                self.transfer_products
                    .insert(product.product.buffer_id(), product.clone());
            }
        }
        for publication in report
            .results
            .iter()
            .filter_map(|completion| completion.output.kv_output.as_ref())
        {
            let route = self
                .buffer_routes
                .get(&publication.source)
                .context("worker returned an unplanned KV publication")?;
            anyhow::ensure!(
                route.worker_index == worker_index,
                "KV publication was returned by another owner"
            );
            if let Some(stored) = self.kv_transfers.get_mut(&publication.source) {
                stored.merge_locations(publication)?;
            } else {
                self.kv_transfers
                    .insert(publication.source, publication.clone());
            }
        }
        self.publish_result(report)?;
        if !failed_calls.is_empty() {
            // Keep the actual failed completion and all accepted independent work.
            // Future consumers cannot become ready when their producer has failed.
            return Err(WorkerFailure {
                worker_id: self.workers[worker_index].0.clone(),
                endpoints: Vec::new(),
                requests: failed_calls.iter().map(|(request, _)| *request).collect(),
                retired: Vec::new(),
                buffers: failed_buffers,
                execution: None,
                message: "call failed before its dependent work could complete".into(),
            }
            .into());
        }
        self.dispatch_ready()
    }

    /// Publishes ready calls immediately; only lifecycle receipts wait for all workers.
    fn publish_result(&mut self, report: WorkerResult) -> anyhow::Result<()> {
        let batch_id = report.batch_id;
        let pending = self
            .pending
            .get(&batch_id)
            .context("result has no pending batch")?;
        let done = pending.expected_workers == 0;
        anyhow::ensure!(
            !done || pending.calls.values().all(|call| call.completed),
            "worker execution finished without every planned call"
        );
        let mut result = logical_result(report, done, &pending.commands);
        for receipt in &mut result.command_results {
            receipt.outcome = pending
                .command_outcomes
                .get(&receipt.command_index)
                .copied()
                .unwrap_or(CommandOutcome::Applied);
        }
        if done {
            let pending = self
                .pending
                .remove(&batch_id)
                .expect("completed batch is owned");
            for (index, command) in pending.commands.into_iter().enumerate() {
                if pending.command_outcomes.get(&(index as u32)) == Some(&CommandOutcome::Failed) {
                    continue;
                }
                match command {
                    BatchCommand::Finish {
                        request_key,
                        retained_buffers,
                    } => {
                        self.forget_request(request_key, &retained_buffers.into_iter().collect());
                    }
                    BatchCommand::Free { buffer } => {
                        self.buffer_workers.remove(&buffer);
                        self.buffer_allocations.remove(&buffer);
                        self.buffer_routes.retain(|identity, _| *identity != buffer);
                        self.kv_transfers.remove(&buffer);
                        self.transfer_products
                            .retain(|identity, _| *identity != buffer);
                    }
                    BatchCommand::Start { .. } => {
                        unreachable!("admissions are tracked by worker ownership")
                    }
                }
            }
        }
        if !result.results.is_empty() || done {
            self.ready.push_back(result);
        }
        Ok(())
    }

    /// Retires lineage routing while preserving independently owned product locations.
    fn forget_request(&mut self, request: RequestKey, retained: &HashSet<BufferId>) {
        self.admissions
            .retain(|request_key, _| *request_key != request);
        self.admitted_workers
            .retain(|(_, request_key)| *request_key != request);
        self.buffer_routes
            .retain(|buffer, _| buffer.owner != request || retained.contains(buffer));
        self.transfer_products
            .retain(|buffer, _| buffer.owner != request || retained.contains(buffer));
        self.kv_transfers
            .retain(|buffer, _| buffer.owner != request || retained.contains(buffer));
        self.buffer_allocations
            .retain(|buffer, _| buffer.owner != request || retained.contains(buffer));
        self.buffer_workers
            .retain(|buffer, _| buffer.owner != request || retained.contains(buffer));
    }

    /// Returns the command-channel waker.
    pub fn command_waker(&self) -> CommandWaker {
        self.command_wake.waker()
    }
}

impl Executor for WorkerExecutor {
    fn has_capacity(&self, worker: &WorkerId) -> bool {
        self.routing.get(worker).is_some_and(|&index| {
            self.workers[index].1.available_slots() > self.worker_submissions[index].len()
        })
    }

    fn command_has_capacity(&self, command: &BatchCommand) -> bool {
        self.command_workers(command)
            .into_iter()
            .all(|index| self.has_capacity(&self.workers[index].0))
    }
    fn is_ready(&self, worker: &WorkerId) -> bool {
        self.routing
            .get(worker)
            .is_some_and(|&index| self.workers[index].1.is_ready())
    }

    /// Returns the worker metadata.
    fn info(&self) -> &ExecutorInfo {
        &self.executor_info
    }

    /// Partitions a logical batch across owning workers and registers aggregate completion state.
    fn submit(&mut self, batch: ExecutionBatch) -> Result<(), ExecutorSubmitError> {
        if self.closed {
            return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                "Executor is closed"
            )));
        }
        if self.pending_failure.is_some() {
            return Err(ExecutorSubmitError::WouldBlock(batch));
        }
        if self.pending.len() >= self.depth {
            return Err(ExecutorSubmitError::WouldBlock(batch));
        }
        batch.validate().map_err(ExecutorSubmitError::Failed)?;
        let mut targets = HashSet::new();
        for (_, placement) in &batch.requests {
            let Some(&index) = self.routing.get(&placement.worker) else {
                return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                    "unknown target worker {}",
                    placement.worker
                )));
            };
            targets.insert(index);
        }
        for command in &batch.commands {
            targets.extend(self.command_workers(command));
        }
        if targets
            .iter()
            .any(|&index| !self.has_capacity(&self.workers[index].0))
        {
            return Err(ExecutorSubmitError::WouldBlock(batch));
        }
        let batch_id = batch.id;
        if self.pending.contains_key(&batch_id) {
            return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                "worker router already has batch {batch_id} in flight"
            )));
        }
        let admissions = batch.admissions().cloned().collect::<Vec<_>>();
        self.cache_admissions(&admissions)
            .map_err(ExecutorSubmitError::Failed)?;

        let submit_result = (|| -> anyhow::Result<()> {
            let mut worker_ops = (0..self.workers.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<(Call, RequestPlacement)>>>();
            let mut worker_inputs = (0..self.workers.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<TensorPublication>>>();
            let mut worker_kv_inputs: Vec<Vec<KvTransfer>> =
                (0..self.workers.len()).map(|_| Vec::new()).collect();
            let mut worker_dependencies = (0..self.workers.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<BufferId>>>();
            let mut call_routes = HashMap::with_capacity(batch.requests.len());
            let mut input_routes = HashMap::new();
            for (call, placement) in &batch.requests {
                let variant = call.code;
                let call_worker =
                    self.routing
                        .get(&placement.worker)
                        .copied()
                        .ok_or_else(|| {
                            anyhow::anyhow!("call targets unknown worker {}", placement.worker)
                        })?;
                anyhow::ensure!(
                    self.workers[call_worker]
                        .1
                        .info()
                        .supported_ops
                        .contains(&variant),
                    "target worker does not support call {variant:?}"
                );
                let entries = &self.workers[call_worker].1.info().components;
                anyhow::ensure!(
                    entries.is_empty()
                        || entries.iter().any(|binding| binding.name == call.component),
                    "call targets unloaded component {}",
                    call.component
                );
                for output in call.output_buffers() {
                    let route = BufferRoute {
                        worker_index: call_worker,
                        component: call.component.clone(),
                    };
                    if let Some(existing) = self.buffer_routes.get(&output) {
                        anyhow::ensure!(
                            existing.worker_index == route.worker_index
                                && existing.component == route.component,
                            "product identity was routed to conflicting workers"
                        );
                    } else {
                        self.buffer_routes.insert(output, route);
                    }
                    // Every physical product needs release routing, including
                    // paged latents/KV and request-relay values without an arena params.
                    self.buffer_workers
                        .entry(output)
                        .or_default()
                        .insert(call_worker);
                }
                for output in call.buffer_outputs() {
                    let params = placement
                        .buffers
                        .iter()
                        .find(|params| params.buffer == output.buffer_id())
                        .copied()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "persistent product {:?} has no scheduler params",
                                output
                            )
                        })?;
                    if let Some(existing) = self.buffer_allocations.insert(params.buffer, params) {
                        anyhow::ensure!(
                            existing == params,
                            "buffer identity was assigned conflicting allocations"
                        );
                    }
                }
                call_routes.insert(
                    (call.request_key, call.call_id),
                    PendingCall {
                        worker: call_worker,
                        submitted: false,
                        kind: call.code,
                        completed: false,
                    },
                );
                for input in call.input_buffers() {
                    input_routes
                        .entry(input)
                        .or_insert_with(HashSet::new)
                        .insert(call_worker);
                }
            }
            for payload in &batch.input_transfers {
                let worker_indices =
                    input_routes
                        .get(&payload.product.buffer_id())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "WorkerExecutor cannot route undeclared input product {:?}",
                                payload.product
                            )
                        })?;
                for &worker_index in worker_indices {
                    worker_inputs[worker_index].push(payload.clone());
                }
            }
            for publication in &batch.kv_inputs {
                let workers = input_routes
                    .get(&publication.source)
                    .context("cannot route undeclared KV input")?;
                for &worker_index in workers {
                    worker_kv_inputs[worker_index].push(publication.clone());
                }
            }
            for (call, _) in &batch.requests {
                let consumer_worker = call_routes[&(call.request_key, call.call_id)].worker;
                for input in call.input_buffers() {
                    self.buffer_workers
                        .entry(input)
                        .or_default()
                        .insert(consumer_worker);
                    let Some(producer) = self.buffer_routes.get(&input) else {
                        anyhow::ensure!(
                            worker_inputs[consumer_worker]
                                .iter()
                                .any(|payload| payload.product.buffer_id() == input)
                                || worker_kv_inputs[consumer_worker]
                                    .iter()
                                    .any(|publication| publication.source == input),
                            "input {:?} has no registered producer or supplied publication",
                            input
                        );
                        continue;
                    };
                    if producer.worker_index == consumer_worker
                        && self.shares_product_storage(producer, &call.component)
                    {
                        continue;
                    }
                    anyhow::ensure!(
                        !worker_inputs[consumer_worker]
                            .iter()
                            .any(|existing| existing.product.buffer_id() == input)
                            && !worker_kv_inputs[consumer_worker]
                                .iter()
                                .any(|publication| publication.source == input),
                        "transfer descriptor must be supplied by its producing worker"
                    );
                    if !worker_dependencies[consumer_worker].contains(&input) {
                        worker_dependencies[consumer_worker].push(input);
                    }
                }
            }

            for (call, placement) in batch.requests {
                let worker = call_routes[&(call.request_key, call.call_id)].worker;
                worker_ops[worker].push((call, placement));
            }

            let mut worker_commands = (0..self.workers.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<BatchCommand>>>();
            for command in &batch.commands {
                if matches!(command, BatchCommand::Start { .. }) {
                    continue;
                }
                let targets = self.command_workers(command);
                // A rejected admission can retire before any endpoint has seen it.
                // Free is idempotent once every registered location has acknowledged
                // physical retirement; live locations stay indexed until that point.
                if targets.is_empty()
                    && matches!(
                        command,
                        BatchCommand::Finish { .. } | BatchCommand::Free { .. }
                    )
                {
                    continue;
                }

                anyhow::ensure!(
                    !targets.is_empty(),
                    "worker router has no destination for command {command:?}"
                );
                for worker_index in targets {
                    worker_commands[worker_index].push(command.clone());
                }
            }

            let mut expected_workers = 0;
            for (worker_index, (((ops, commands), input_products), dependencies)) in worker_ops
                .into_iter()
                .zip(worker_commands)
                .zip(worker_inputs)
                .zip(worker_dependencies)
                .enumerate()
            {
                if ops.is_empty() && commands.is_empty() {
                    anyhow::ensure!(
                        dependencies.is_empty(),
                        "product dependency has no consuming worker submission"
                    );
                    continue;
                }
                expected_workers |= Self::worker_bit(worker_index);
                // Keep call-owned resources together while dependencies wait.
                // Rank indices and collective order are derived at physical submission.
                self.worker_submissions[worker_index].push_back(WorkerSubmission::build(
                    batch_id,
                    ops,
                    commands,
                    input_products,
                    std::mem::take(&mut worker_kv_inputs[worker_index]),
                    dependencies,
                ));
            }
            self.pending.insert(
                batch_id,
                PendingBatch {
                    submitted_workers: HashSet::new(),
                    expected_workers,
                    calls: call_routes,
                    commands: batch
                        .commands
                        .iter()
                        .filter(|command| !matches!(command, BatchCommand::Start { .. }))
                        .cloned()
                        .collect(),
                    command_outcomes: HashMap::new(),
                },
            );
            self.dispatch_ready()?;
            if self
                .pending
                .get(&batch_id)
                .is_some_and(|pending| pending.expected_workers == 0)
            {
                self.publish_result(WorkerResult {
                    batch_id,
                    results: Vec::new(),
                    products: Vec::new(),
                    registration: uniserve_worker_ipc::RegistrationAck { visible: true },
                    worker_exec_us: None,
                    forward_stats: None,
                })?;
            }
            for command in &batch.commands {
                match command {
                    BatchCommand::Free { buffer } => {
                        self.transfer_products
                            .retain(|identity, _| *identity != *buffer);
                        self.buffer_routes
                            .retain(|identity, _| *identity != *buffer);
                        self.kv_transfers.remove(buffer);
                    }
                    _ => {}
                }
            }
            Ok(())
        })();
        match submit_result {
            Ok(()) => Ok(()),
            Err(error) if error.is::<WorkerFailure>() => {
                self.pending_failure = Some(
                    self.reconcile_worker_failure(error)
                        .map_err(ExecutorSubmitError::Failed)?,
                );
                Ok(())
            }
            Err(error) => {
                self.pending.remove(&batch_id);
                Err(ExecutorSubmitError::Failed(error))
            }
        }
    }

    /// Drives worker progress and returns the next completed logical batch.
    fn poll(&mut self, timeout: Duration) -> anyhow::Result<Option<BatchResult>> {
        if let Some(loss) = self.pending_failure.take() {
            return Err(loss.into());
        }
        if std::mem::take(&mut self.command_wake_pending) {
            return Ok(None);
        }
        if let Err(error) = self.pump() {
            return Err(self.reconcile_worker_failure(error)?.into());
        }
        if std::mem::take(&mut self.command_wake_pending) {
            return Ok(None);
        }
        if let Some(result) = self.ready.pop_front() {
            return Ok(Some(result));
        }
        if timeout.is_zero() {
            return Ok(None);
        }
        crate::worker::park_descriptors(&self.progress_fds, timeout)?;
        if let Err(error) = self.pump() {
            return Err(self.reconcile_worker_failure(error)?.into());
        }
        if std::mem::take(&mut self.command_wake_pending) {
            return Ok(None);
        }
        Ok(self.ready.pop_front())
    }

    /// Closes the component and releases its resources.
    fn close(&mut self) -> anyhow::Result<()> {
        self.closed = true;
        let mut first_error = None;
        for worker in &mut self.workers {
            if let Err(error) = worker.1.close()
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        self.refresh_progress_fds();
        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(())
        }
    }
}
