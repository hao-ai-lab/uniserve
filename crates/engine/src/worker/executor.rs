//! Submission of explicitly targeted operations to Worker instances.
//!
//! Each destination has independent bounded capacity. Ready work may bypass
//! unrelated blocked requests while preserving a request's command order.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::Duration;

use super::{RunSubmitError, Worker};
use crate::executor::{
    Batch, BatchResult, Executor, ExecutorInfo, ExecutorSubmitError, LogicalResultTracker, Op,
    TransportMap, WorkerFailure, WorkerId, physical_run,
};
use anyhow::Context;
use uniserve_core::CommandWaker;
use uniserve_worker_ipc::{
    BatchCommand, BufferAllocation, BufferId, InlineValue, NewRequest, ProductPayload, ProductRef,
    RequestKey, RunResult, TransferKind,
};

struct PendingStep {
    worker_runs: HashMap<usize, u64>,
    expected_workers: u64,
    operations: HashMap<(RequestKey, uniserve_worker_ipc::OpId), usize>,
    returned_operations: HashSet<(RequestKey, uniserve_worker_ipc::OpId)>,
    finished_requests: Vec<(RequestKey, HashSet<BufferId>)>,
    freed_buffers: Vec<BufferId>,
}

#[derive(Clone)]
struct WorkerSubmission {
    batch: Batch,
    dependencies: Vec<ProductRef>,
}

#[derive(Clone)]
struct ProductRoute {
    worker_index: usize,
    entry: String,
}

/// Dispatch targeted operations and track their independently completed results.
pub struct WorkerExecutor {
    routing: HashMap<WorkerId, usize>,
    workers: Vec<(WorkerId, Worker)>,
    executor_info: ExecutorInfo,
    depth: usize,
    command_wake: crate::handle::WakeSignal,
    progress_fds: Vec<i32>,
    pending: BTreeMap<u64, PendingStep>,
    ready: VecDeque<RunResult>,
    admissions: HashMap<RequestKey, NewRequest>,
    admitted_workers: HashSet<(usize, RequestKey)>,
    operation_routes: HashMap<(RequestKey, uniserve_worker_ipc::OpId), usize>,
    product_routes: HashMap<ProductRef, ProductRoute>,
    buffer_allocations: HashMap<BufferId, BufferAllocation>,
    buffer_workers: HashMap<BufferId, HashSet<usize>>,
    transfer_products: HashMap<ProductRef, ProductPayload>,
    worker_submissions: Vec<VecDeque<WorkerSubmission>>,
    worker_collective_seqs: Vec<u64>,
    logical_results: LogicalResultTracker,
    command_wake_pending: bool,
    pending_failure: Option<WorkerFailure>,
    closed: bool,
}

impl WorkerExecutor {
    /// Bind ready instances with independent queues and their physical wake descriptors.
    pub fn try_new(
        workers: Vec<(WorkerId, Worker)>,
        transfer: TransportMap,
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
                "Worker {id} was initialized with different transfer bindings"
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
                "transfer edge names an unbound Worker"
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
                    let backend = edge.transport.as_str();
                    anyhow::ensure!(
                        source.transfer_backends.iter().any(|name| name == backend)
                            && destination
                                .transfer_backends
                                .iter()
                                .any(|name| name == backend),
                        "physical edge requires an uninitialized transfer backend {backend}"
                    );
                    anyhow::ensure!(
                        source.endpoint.node == destination.endpoint.node,
                        "{backend} requires endpoints on the same node"
                    );
                    match edge.transport {
                        crate::executor::TransferBackend::Local => anyhow::ensure!(
                            source.endpoint.address_space == destination.endpoint.address_space,
                            "local transfer requires a shared address space"
                        ),
                        crate::executor::TransferBackend::CudaIpc => anyhow::ensure!(
                            source.device.starts_with("cuda:")
                                && destination.device.starts_with("cuda:"),
                            "CUDA IPC requires CUDA devices on both endpoints"
                        ),
                        crate::executor::TransferBackend::Shm => {}
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
            operation_routes: HashMap::new(),
            product_routes: HashMap::new(),
            buffer_allocations: HashMap::new(),
            buffer_workers: HashMap::new(),
            transfer_products: HashMap::new(),
            worker_submissions: (0..worker_count).map(|_| VecDeque::new()).collect(),
            worker_collective_seqs: vec![0; worker_count],
            logical_results: LogicalResultTracker::default(),
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
            .context("failed Worker is not bound")?;
        let lost = !loss.endpoints.is_empty();
        if lost {
            self.refresh_worker(index);
        }
        let failed_run = (!lost)
            .then(|| loss.execution.as_ref().and_then(|error| error.run_id))
            .flatten()
            .and_then(|run_id| {
                self.pending.iter().find_map(|(&batch_id, step)| {
                    (step.worker_runs.get(&index) == Some(&run_id)).then_some(batch_id)
                })
            });
        let mut retired = loss.retired.iter().copied().collect::<HashSet<_>>();
        let failed_operations = retired
            .iter()
            .map(|(_, request, operation)| (*request, *operation))
            .collect::<HashSet<_>>();
        let mut requests = loss.requests.iter().copied().collect::<HashSet<_>>();
        let mut products = loss.products.iter().cloned().collect::<HashSet<_>>();
        if lost {
            for (product, payload) in &mut self.transfer_products {
                let InlineValue::Transfer(handle) = &mut payload.value else {
                    continue;
                };
                let tensors = match handle {
                    uniserve_worker_ipc::TransferHandle::Encoder { tensor, .. }
                    | uniserve_worker_ipc::TransferHandle::DeviceProduct { tensor, .. }
                    | uniserve_worker_ipc::TransferHandle::Latent { tensor, .. } => {
                        std::slice::from_mut(tensor)
                    }
                    uniserve_worker_ipc::TransferHandle::Kv { tensors, .. } => {
                        tensors.as_mut_slice()
                    }
                };
                for tensor in tensors.iter_mut() {
                    tensor
                        .locations
                        .retain(|location| !loss.endpoints.contains(&location.source));
                }
                if tensors.iter().any(|tensor| !tensor.has_complete_coverage()) {
                    products.insert(product.clone());
                }
            }
        }
        for (product, route) in &self.product_routes {
            if (lost
                && route.worker_index == index
                && !self.transfer_products.contains_key(product))
                || failed_operations.contains(&(product.request_key, product.producer_op_id))
            {
                products.insert(product.clone());
            }
        }
        for step in self.pending.values() {
            requests.extend(step.operations.iter().filter_map(|((request, _), worker)| {
                (lost && *worker == index).then_some(*request)
            }));
        }
        // Abandoning an unstarted producer also invalidates its future products.
        // Already running work on another instance keeps its real completion path.
        loop {
            let before = (requests.len(), products.len());
            for (_, worker) in &self.workers {
                for operation in worker.inflight_operations() {
                    if operation
                        .inputs()
                        .iter()
                        .chain(operation.predicate())
                        .any(|product| products.contains(product))
                    {
                        requests.insert(operation.request_key);
                    }
                }
            }
            for submission in self.worker_submissions.iter().flatten() {
                for operation in &submission.batch.ops {
                    if operation
                        .payload
                        .inputs
                        .iter()
                        .chain(operation.payload.predicate.iter())
                        .any(|product| products.contains(product))
                    {
                        requests.insert(operation.request);
                    }
                    if requests.contains(&operation.request) {
                        products.extend(operation.payload.outputs.iter().cloned());
                    }
                }
            }
            if before == (requests.len(), products.len()) {
                break;
            }
        }
        for (worker_index, queue) in self.worker_submissions.iter_mut().enumerate() {
            let mut active = VecDeque::new();
            for mut submission in queue.drain(..) {
                let batch_id = submission.batch.id;
                for operation in submission.batch.retire_requests(&requests) {
                    retired.insert((batch_id, operation.request, operation.id));
                }
                let inputs = submission
                    .batch
                    .ops
                    .iter()
                    .flat_map(|op| op.payload.inputs.iter().chain(op.payload.predicate.iter()))
                    .collect::<HashSet<_>>();
                submission
                    .dependencies
                    .retain(|product| inputs.contains(product));
                if lost && worker_index == index {
                    submission.batch.commands.clear();
                }
                if submission.batch.ops.is_empty() && submission.batch.commands.is_empty() {
                    if let Some(step) = self.pending.get_mut(&batch_id) {
                        step.expected_workers &= !Self::worker_bit(worker_index);
                    }
                } else {
                    active.push_back(submission);
                }
            }
            *queue = active;
        }
        for (batch_id, step) in &mut self.pending {
            if lost || failed_run == Some(*batch_id) {
                step.worker_runs.remove(&index);
                step.expected_workers &= !Self::worker_bit(index);
            }
            retired.extend(step.operations.iter().filter_map(|(identity, worker)| {
                ((lost || failed_run == Some(*batch_id))
                    && *worker == index
                    && !step.returned_operations.contains(identity))
                .then_some((*batch_id, identity.0, identity.1))
            }));
            self.logical_results.retire_commands(*batch_id, &requests);
        }
        if let Some(batch_id) = failed_run {
            let failed_requests = loss.requests.iter().copied().collect();
            self.logical_results
                .fail_commands(batch_id, &failed_requests);
        }
        loss.retired.clear();
        for (batch_id, request, operation) in retired {
            let step = self
                .pending
                .get_mut(&batch_id)
                .context("retired operation has no pending step")?;
            anyhow::ensure!(
                !step.returned_operations.contains(&(request, operation)),
                "completed operation cannot be abandoned"
            );
            anyhow::ensure!(
                step.operations.remove(&(request, operation)).is_some(),
                "abandoned operation is not pending"
            );
            self.logical_results.retire(batch_id, request, operation)?;
            loss.retired.push((batch_id, request, operation));
        }
        if lost {
            self.admitted_workers.retain(|(worker, _)| *worker != index);
            self.operation_routes.retain(|_, worker| *worker != index);
            for workers in self.buffer_workers.values_mut() {
                workers.remove(&index);
            }
        }
        let batches = self.pending.keys().copied().collect::<Vec<_>>();
        for batch_id in batches {
            if self.try_complete(batch_id)? {
                self.ready.push_back(RunResult {
                    batch_id,
                    run_id: batch_id,
                    completions: Vec::new(),
                    products: Vec::new(),
                    registration: uniserve_worker_ipc::RegistrationAck { visible: true },
                    worker_exec_us: None,
                    forward_stats: None,
                    done: true,
                });
            }
        }
        loss.requests = requests.into_iter().collect();
        loss.products = products.into_iter().collect();
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

    /// Partitions new-request admissions by the workers used by their first operations.
    fn admissions_for(
        &self,
        worker_index: usize,
        operations: &[Op],
    ) -> anyhow::Result<Vec<NewRequest>> {
        let mut admissions = Vec::new();
        for operation in operations {
            if self
                .admitted_workers
                .contains(&(worker_index, operation.request))
            {
                continue;
            }
            let admission = self.admissions.get(&operation.request).ok_or_else(|| {
                anyhow::anyhow!(
                    "worker {worker_index} has no admission for request {:?}",
                    operation.request
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
                self.operation_routes
                    .iter()
                    .filter_map(|((key, _), worker)| (*key == request).then_some(*worker)),
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

    /// Routes semantic commits to their producer and retirement to every physical owner.
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
            BatchCommand::Commit {
                request_key,
                selected,
                ..
            } => self
                .operation_routes
                .get(&(*request_key, selected.op_id))
                .copied()
                .into_iter()
                .collect(),
            BatchCommand::Finish { request_key, .. } | BatchCommand::Retire { request_key, .. } => {
                self.request_workers(*request_key)
            }
        }
    }

    /// Only the cutoff owner selects semantic state; auxiliary owners retire physical storage.
    fn physical_command(
        &self,
        command: &BatchCommand,
        worker: usize,
    ) -> anyhow::Result<BatchCommand> {
        if let BatchCommand::Finish {
            request_key,
            cutoff,
            retained_buffers,
            ..
        } = command
        {
            if cutoff.op_id.0 != 0 {
                let owner = self
                    .operation_routes
                    .get(&(*request_key, cutoff.op_id))
                    .context("Finish cutoff has no producing Worker")?;
                if *owner != worker {
                    return Ok(BatchCommand::Retire {
                        request_key: *request_key,
                        retained_buffers: retained_buffers.clone(),
                    });
                }
            }
        }
        Ok(command.clone())
    }

    /// Keep resident references direct when producer and consumer execute in
    /// the same address spaces. Different multi-rank entries require published
    /// layouts, including when both entries belong to this Worker.
    fn shares_product_storage(&self, producer: &ProductRoute, consumer_entry: &str) -> bool {
        let entries = &self.workers[producer.worker_index].1.info().components;
        let source = entries.iter().find(|entry| entry.name == producer.entry);
        let destination = entries.iter().find(|entry| entry.name == consumer_entry);
        if producer.entry == consumer_entry {
            return source.is_none_or(|entry| entry.config.distribution.is_none());
        }
        source
            .zip(destination)
            .is_some_and(|(source, destination)| {
                source.config.ranks.len() == 1 && source.config.ranks == destination.config.ranks
            })
    }

    /// Submits or queues one worker-local run while preserving collective order.
    fn submit_worker(
        &mut self,
        worker_index: usize,
        submission: &WorkerSubmission,
    ) -> anyhow::Result<bool> {
        let batch = &submission.batch;
        let admissions = self.admissions_for(worker_index, &batch.ops)?;
        let request_keys = admissions
            .iter()
            .map(|admission| admission.request_key)
            .collect::<Vec<_>>();
        let commands = admissions
            .into_iter()
            .map(|request| BatchCommand::Start { request })
            .chain(batch.commands.iter().cloned())
            .collect();
        let mut inputs = batch.inline.clone();
        for dependency in &submission.dependencies {
            anyhow::ensure!(
                !inputs.iter().any(|payload| payload.product == *dependency),
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
            .context("physical run ID space exhausted")?;
        let mut run = physical_run(
            batch.id,
            collective_seq,
            collective_seq,
            batch.ops.clone(),
            commands,
            inputs,
        )?;
        for dependency in &submission.dependencies {
            if !dependency.uses_persistent_buffer() {
                continue;
            }
            let params = self
                .buffer_allocations
                .get(&dependency.buffer_id())
                .copied()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "persistent dependency {:?} has no scheduler params",
                        dependency
                    )
                })?;
            if !run
                .buffer_allocations
                .iter()
                .any(|existing| existing.buffer == params.buffer)
            {
                run.buffer_allocations.push(params);
            }
        }
        run.validate()?;
        for payload in &mut run.input_products {
            use uniserve_worker_ipc::TransferHandle;
            let InlineValue::Transfer(handle) = &mut payload.value else {
                continue;
            };
            let tensors = match handle {
                TransferHandle::Encoder { tensor, .. }
                | TransferHandle::DeviceProduct { tensor, .. }
                | TransferHandle::Latent { tensor, .. } => std::slice::from_mut(tensor),
                TransferHandle::Kv { tensors, .. } => tensors.as_mut_slice(),
            };
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
        }
        match self.workers[worker_index].1.submit_run(run) {
            Ok(()) => {}
            Err(RunSubmitError::WouldBlock(_)) => return Ok(false),
            Err(RunSubmitError::Failed(error)) => return Err(error),
        }
        self.worker_collective_seqs[worker_index] = collective_seq;
        self.pending
            .get_mut(&batch.id)
            .context("submitted batch is not pending")?
            .worker_runs
            .insert(worker_index, collective_seq);
        self.admitted_workers
            .extend(request_keys.into_iter().map(|key| (worker_index, key)));
        Ok(true)
    }

    /// Dispatches queued worker runs whose collective and capacity constraints are satisfied.
    fn dispatch_ready(&mut self) -> anyhow::Result<()> {
        loop {
            let mut progressed = false;
            for worker_index in 0..self.workers.len() {
                let mut blocked_requests = HashSet::new();
                let position =
                    self.worker_submissions[worker_index]
                        .iter()
                        .position(|submission| {
                            let requests = submission
                                .batch
                                .ops
                                .iter()
                                .map(|op| op.request)
                                .chain(
                                    submission
                                        .batch
                                        .commands
                                        .iter()
                                        .map(|command| command.request_key()),
                                )
                                .collect::<HashSet<_>>();
                            let ready = requests.is_disjoint(&blocked_requests)
                                && submission
                                    .dependencies
                                    .iter()
                                    .all(|product| self.transfer_products.contains_key(product));
                            blocked_requests.extend(requests);
                            ready
                        });
                let Some(position) = position else {
                    continue;
                };
                let submission = self.worker_submissions[worker_index]
                    .remove(position)
                    .ok_or_else(|| anyhow::anyhow!("ready worker submission disappeared"))?;
                if !self.submit_worker(worker_index, &submission)? {
                    self.worker_submissions[worker_index].insert(position, submission);
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
                let polled = self.workers[worker_index].1.poll_run(Duration::ZERO);
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
    fn route_result(&mut self, worker_index: usize, report: RunResult) -> anyhow::Result<()> {
        report.validate()?;
        let mut report = report;
        let failed_operations = report
            .completions
            .iter()
            .filter(|completion| completion.status == uniserve_worker_ipc::OpStatus::Error)
            .map(|completion| (completion.request_key, completion.op_id))
            .collect::<HashSet<_>>();
        let failed_products = self
            .product_routes
            .keys()
            .filter(|product| {
                failed_operations.contains(&(product.request_key, product.producer_op_id))
            })
            .cloned()
            .collect();
        let run_id = report.batch_id;
        anyhow::ensure!(
            self.pending
                .get(&run_id)
                .and_then(|step| step.worker_runs.get(&worker_index))
                == Some(&report.run_id),
            "worker returned a run that does not belong to its logical batch"
        );
        // The engine consumes logical batch results; physical IDs stop here.
        report.run_id = run_id;
        let worker_bit = Self::worker_bit(worker_index);
        {
            let step = self
                .pending
                .get_mut(&run_id)
                .ok_or_else(|| anyhow::anyhow!("worker returned unknown step {run_id}"))?;
            anyhow::ensure!(
                step.expected_workers & worker_bit != 0,
                "worker {worker_index} returned a duplicate or unexpected result for step {run_id}"
            );
            let expected_operations = step
                .operations
                .iter()
                .filter_map(|(identity, route)| (*route == worker_index).then_some(*identity))
                .collect::<HashSet<_>>();
            anyhow::ensure!(
                !report.completions.is_empty() || expected_operations.is_empty() || report.done,
                "worker {worker_index} returned an empty partial report for step {run_id}"
            );
            for completion in &report.completions {
                let identity = (completion.request_key, completion.op_id);
                anyhow::ensure!(
                    expected_operations.contains(&identity),
                    "worker {worker_index} returned an unexpected operation for step {run_id}"
                );
                anyhow::ensure!(
                    step.returned_operations.insert(identity),
                    "worker {worker_index} returned an operation twice for step {run_id}"
                );
            }
            if report.done {
                anyhow::ensure!(
                    expected_operations
                        .iter()
                        .all(|identity| step.returned_operations.contains(identity)),
                    "worker {worker_index} terminated step {run_id} before every operation completed"
                );
                step.expected_workers &= !worker_bit;
            }
        }
        let mut visible_products = Vec::with_capacity(report.products.len());
        for product in report.products.drain(..) {
            let InlineValue::Transfer(handle) = &product.value else {
                visible_products.push(product);
                continue;
            };
            let route = self.product_routes.get(&product.product).ok_or_else(|| {
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
            let kind = handle.kind();
            if let Some(existing) = self.transfer_products.get_mut(&product.product) {
                let InlineValue::Transfer(stored) = &mut existing.value else {
                    anyhow::bail!("transfer product changed its representation kind");
                };
                stored.merge_locations(handle)?;
            } else {
                self.transfer_products
                    .insert(product.product.clone(), product.clone());
            }
            if kind == TransferKind::Kv {
                visible_products.push(ProductPayload {
                    product: product.product,
                    value: InlineValue::Bytes(Vec::new()),
                });
            }
        }
        report.products = visible_products;
        let done = self.try_complete(run_id)?;
        report.done = done;
        if !report.completions.is_empty() || done {
            self.ready.push_back(report);
        }
        if !failed_operations.is_empty() {
            // Keep the actual failed completion and all accepted independent work.
            // Future consumers cannot become ready when their producer has failed.
            return Err(WorkerFailure {
                worker_id: self.workers[worker_index].0.clone(),
                endpoints: Vec::new(),
                requests: failed_operations
                    .iter()
                    .map(|(request, _)| *request)
                    .collect(),
                retired: Vec::new(),
                products: failed_products,
                execution: None,
                message: "operation failed before its dependent work could complete".into(),
            }
            .into());
        }
        self.dispatch_ready()
    }

    /// Finalizes a worker run once every required worker has returned its contribution.
    fn try_complete(&mut self, run_id: u64) -> anyhow::Result<bool> {
        let complete = self
            .pending
            .get(&run_id)
            .is_some_and(|step| step.expected_workers == 0);
        if !complete {
            return Ok(false);
        }
        let step = self
            .pending
            .remove(&run_id)
            .ok_or_else(|| anyhow::anyhow!("pending step {run_id} disappeared"))?;
        anyhow::ensure!(
            step.returned_operations.len() == step.operations.len(),
            "worker execution finished without every planned operation"
        );
        let failed = self.logical_results.failed_commands(run_id);
        for (request_key, retained) in step.finished_requests {
            if !failed.iter().any(|command| {
                matches!(command,
                BatchCommand::Finish { request_key: key, .. }
                    | BatchCommand::Retire { request_key: key, .. } if *key == request_key)
            }) {
                self.forget_request(request_key, &retained);
            }
        }
        for buffer in step.freed_buffers {
            if !failed.iter().any(|command| {
                matches!(command,
                BatchCommand::Free { buffer: target } if *target == buffer)
            }) {
                self.buffer_workers.remove(&buffer);
                self.buffer_allocations.remove(&buffer);
                self.product_routes
                    .retain(|product, _| product.buffer_id() != buffer);
                self.transfer_products
                    .retain(|product, _| product.buffer_id() != buffer);
            }
        }
        Ok(true)
    }

    /// Retires lineage routing while preserving independently owned product locations.
    fn forget_request(&mut self, request: RequestKey, retained: &HashSet<BufferId>) {
        self.admissions
            .retain(|request_key, _| *request_key != request);
        self.admitted_workers
            .retain(|(_, request_key)| *request_key != request);
        self.operation_routes
            .retain(|(request_key, _), _| *request_key != request);
        self.product_routes.retain(|product, _| {
            product.request_key != request || retained.contains(&product.buffer_id())
        });
        self.transfer_products.retain(|product, _| {
            product.request_key != request || retained.contains(&product.buffer_id())
        });
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
    fn submit(&mut self, batch: Batch) -> Result<(), ExecutorSubmitError> {
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
        for op in &batch.ops {
            let Some(&index) = self.routing.get(&op.target.0) else {
                return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                    "unknown target worker {}",
                    op.target.0
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
        self.logical_results
            .register(&batch)
            .map_err(ExecutorSubmitError::Failed)?;
        let admissions = batch.admissions().cloned().collect::<Vec<_>>();
        self.cache_admissions(&admissions)
            .map_err(ExecutorSubmitError::Failed)?;

        let submit_result = (|| -> anyhow::Result<()> {
            let mut worker_ops = (0..self.workers.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<Op>>>();
            let mut worker_inputs = (0..self.workers.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<ProductPayload>>>();
            let mut worker_dependencies = (0..self.workers.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<ProductRef>>>();
            let mut operation_routes = HashMap::with_capacity(batch.ops.len());
            let mut input_routes = HashMap::new();
            for op in &batch.ops {
                let operation = op.clone().into_operation();
                let variant = op.kind();
                let operation_worker =
                    self.routing.get(&op.target.0).copied().ok_or_else(|| {
                        anyhow::anyhow!("operation targets unknown worker {}", op.target.0)
                    })?;
                anyhow::ensure!(
                    self.workers[operation_worker]
                        .1
                        .info()
                        .supported_ops
                        .contains(&variant),
                    "target worker does not support operation {variant:?}"
                );
                let entries = &self.workers[operation_worker].1.info().components;
                anyhow::ensure!(
                    entries.is_empty() || entries.iter().any(|binding| binding.name == op.target.1),
                    "operation targets unloaded entry {}",
                    op.target.1
                );
                if let Some(existing) = self
                    .operation_routes
                    .insert((operation.request_key, operation.op_id), operation_worker)
                {
                    anyhow::ensure!(
                        existing == operation_worker,
                        "operation identity was routed to conflicting workers"
                    );
                }
                for output in operation.outputs() {
                    let route = ProductRoute {
                        worker_index: operation_worker,
                        entry: op.target.1.clone(),
                    };
                    if let Some(existing) = self.product_routes.get(output) {
                        anyhow::ensure!(
                            existing.worker_index == route.worker_index
                                && existing.entry == route.entry,
                            "product identity was routed to conflicting workers"
                        );
                    } else {
                        self.product_routes.insert(output.clone(), route);
                    }
                    // Every physical product needs release routing, including
                    // paged latents/KV and request-relay values without an arena params.
                    self.buffer_workers
                        .entry(output.buffer_id())
                        .or_default()
                        .insert(operation_worker);
                    if output.uses_persistent_buffer() {
                        let params = op
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
                        if let Some(existing) =
                            self.buffer_allocations.insert(params.buffer, params)
                        {
                            anyhow::ensure!(
                                existing == params,
                                "buffer identity was assigned conflicting allocations"
                            );
                        }
                    }
                }
                operation_routes.insert((operation.request_key, operation.op_id), operation_worker);
                for input in operation.inputs().iter().chain(
                    operation
                        .predicate()
                        .into_iter()
                        .filter(|predicate| !operation.inputs().contains(*predicate)),
                ) {
                    input_routes
                        .entry(input.clone())
                        .or_insert_with(HashSet::new)
                        .insert(operation_worker);
                }
                worker_ops[operation_worker].push(op.clone());
            }
            for payload in &batch.inline {
                let worker_indices = input_routes.get(&payload.product).ok_or_else(|| {
                    anyhow::anyhow!(
                        "WorkerExecutor cannot route undeclared input product {:?}",
                        payload.product
                    )
                })?;
                for &worker_index in worker_indices {
                    worker_inputs[worker_index].push(payload.clone());
                }
            }
            for op in &batch.ops {
                let operation = op.clone().into_operation();
                let consumer_worker = operation_routes[&(operation.request_key, operation.op_id)];
                for input in operation.inputs().iter().chain(
                    operation
                        .predicate()
                        .into_iter()
                        .filter(|predicate| !operation.inputs().contains(*predicate)),
                ) {
                    if input.storage_class == uniserve_worker_ipc::StorageClass::HostStaging {
                        continue;
                    }
                    self.buffer_workers
                        .entry(input.buffer_id())
                        .or_default()
                        .insert(consumer_worker);
                    let Some(producer) = self.product_routes.get(input) else {
                        anyhow::ensure!(
                            worker_inputs[consumer_worker]
                                .iter()
                                .any(|payload| payload.product == *input),
                            "input {:?} has no registered producer or supplied publication",
                            input
                        );
                        continue;
                    };
                    if producer.worker_index == consumer_worker
                        && self.shares_product_storage(producer, &op.target.1)
                    {
                        continue;
                    }
                    anyhow::ensure!(
                        !worker_inputs[consumer_worker]
                            .iter()
                            .any(|existing| existing.product == *input),
                        "transfer descriptor must be supplied by its producing worker"
                    );
                    if !worker_dependencies[consumer_worker].contains(input) {
                        worker_dependencies[consumer_worker].push(input.clone());
                    }
                }
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
                        BatchCommand::Finish { .. }
                            | BatchCommand::Retire { .. }
                            | BatchCommand::Free { .. }
                    )
                {
                    continue;
                }

                anyhow::ensure!(
                    !targets.is_empty(),
                    "worker router has no destination for command {command:?}"
                );
                for worker_index in targets {
                    worker_commands[worker_index]
                        .push(self.physical_command(command, worker_index)?);
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
                // Keep operation-owned resources together while dependencies wait.
                // Rank indices and collective order are derived at physical submission.
                let submission = WorkerSubmission {
                    batch: Batch::new(batch_id, ops, commands, input_products),
                    dependencies,
                };
                self.worker_submissions[worker_index].push_back(submission);
            }
            self.pending.insert(
                batch_id,
                PendingStep {
                    worker_runs: HashMap::new(),
                    expected_workers,
                    operations: operation_routes,
                    returned_operations: HashSet::new(),
                    freed_buffers: batch
                        .commands
                        .iter()
                        .filter_map(|command| match command {
                            BatchCommand::Free { buffer } => Some(*buffer),
                            _ => None,
                        })
                        .collect(),
                    finished_requests: batch
                        .commands
                        .iter()
                        .filter_map(|control| match control {
                            BatchCommand::Finish {
                                request_key,
                                retained_buffers,
                                ..
                            }
                            | BatchCommand::Retire {
                                request_key,
                                retained_buffers,
                            } => Some((*request_key, retained_buffers.iter().copied().collect())),
                            _ => None,
                        })
                        .collect(),
                },
            );
            self.dispatch_ready()?;
            if self.try_complete(batch_id)? {
                self.ready.push_back(RunResult {
                    batch_id,
                    run_id: batch_id,
                    completions: Vec::new(),
                    products: Vec::new(),
                    registration: uniserve_worker_ipc::RegistrationAck { visible: true },
                    worker_exec_us: None,
                    forward_stats: None,
                    done: true,
                });
            }
            for command in &batch.commands {
                match command {
                    BatchCommand::Free { buffer } => {
                        self.transfer_products
                            .retain(|product, _| product.buffer_id() != *buffer);
                        self.product_routes
                            .retain(|product, _| product.buffer_id() != *buffer);
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
                self.logical_results.unregister(batch_id);
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
            return self.logical_results.apply(result).map(Some);
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
        self.ready
            .pop_front()
            .map(|result| self.logical_results.apply(result))
            .transpose()
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
