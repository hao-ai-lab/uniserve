//! Submission of explicitly targeted calls to WorkerGroup instances.
//!
//! `WorkerExecutor` is the `Executor` the scheduler drives over a set of
//! `WorkerGroup`s. It splits each logical `ExecutionBatch` into one
//! `WorkerSubmission` per target worker, delivers each request's `Start` with
//! its first call on a worker and every other lifecycle command to the workers
//! that hold what it releases, and joins the workers' reports back into
//! `BatchResult`s: call completions are published as they arrive, command
//! receipts once every worker has finished the batch.
//!
//! Each destination has independent bounded capacity. Every worker has its own
//! FIFO of submissions whose head waits until the products it depends on are
//! published. A blocked queue does not hold back other workers, and each
//! worker receives its batches, and therefore a request's commands, in
//! submission order.
//!
//! The executor also holds the head's copy of every cross-worker product
//! publication until the scheduler frees the buffer or its request finishes
//! without retaining it, and on a `WorkerFailure` computes the calls,
//! requests, and buffers the failure invalidates.

use crate::executor::WorkerResult;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::Duration;

use super::{BatchSubmitError, WorkerGroup};
use crate::executor::{
    BatchResult, CommandOutcome, ExecutionBatch, Executor, ExecutorInfo, ExecutorSubmitError,
    RequestPlacement, TransferConfig, UnroutableInputs, WorkerFailure, WorkerId, logical_result,
};
use anyhow::Context;
use uniserve_core::CommandWaker;
use uniserve_worker_ipc::{
    BatchCommand, BufferAllocation, BufferId, Call, KvTransfer, NewRequest, RequestKey,
    TensorPublication, TensorRef,
};

/// One worker's outstanding submission; returned calls leave this record.
struct PendingWorker {
    /// Whether the `WorkerGroup` accepted the submission and still owes its
    /// report. False while the submission waits in
    /// `WorkerExecutor::worker_submissions`, and cleared by failure
    /// reconciliation when the group will send no report.
    submitted: bool,
    /// Calls whose completion this worker has not yet returned, with the
    /// computation code the completion must carry.
    calls: HashMap<(RequestKey, uniserve_worker_ipc::CallId), uniserve_worker_ipc::CallKind>,
}

/// Dispatch and receipts for one scheduler batch.
struct PendingBatch {
    /// Workers, by index into `WorkerExecutor::workers`, that have not yet
    /// finished their part of the batch. The batch is terminal when empty.
    workers: HashMap<usize, PendingWorker>,
    /// The batch's lifecycle commands without `Start` admissions, so a
    /// position here is a `CommandResult::command_index`.
    commands: Vec<BatchCommand>,
    /// Outcomes other than `Applied`, keyed by position in `commands`.
    command_outcomes: HashMap<u32, CommandOutcome>,
}

impl PendingBatch {
    /// Records `outcome` for every command that targets one of `requests`.
    ///
    /// The first recorded outcome is kept, except that `Failed` replaces any
    /// other: a failed release keeps ownership even when another endpoint
    /// retires it.
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

/// One worker's part of a logical batch, queued until it can be dispatched.
#[derive(Clone)]
struct WorkerSubmission {
    /// The calls routed to this worker, with its commands and the external
    /// publications it reads.
    batch: ExecutionBatch,
    /// Products this worker reads that another worker, or a component of this
    /// worker that does not share the reader's storage, produces. The
    /// submission stays at its queue head until each has a publication in
    /// `transfer_products` or `kv_transfers`.
    dependencies: Vec<BufferId>,
}

impl WorkerSubmission {
    /// Builds the one submission a worker receives for this batch.
    ///
    /// A batch carries one computation to one component, so a worker receives
    /// it once, under one identity, and answers it once. Product inputs the
    /// worker does not yet hold travel as dependencies and hold the submission
    /// in its queue until they resolve. Inputs, KV inputs, and dependencies
    /// that none of `calls` reads are dropped.
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

/// What became of one queued submission offered to its worker.
enum Dispatch {
    /// The group accepted the submission.
    Submitted,
    /// The group had no free slot; nothing was recorded.
    Deferred,
    /// The group refused inputs that no configured edge carries to the rank
    /// reading them; nothing was recorded. The failure names the requests
    /// whose calls read those inputs.
    Refused(Box<WorkerFailure>),
}

/// The worker and component whose call produces a buffer.
#[derive(Clone)]
struct BufferRoute {
    worker_index: usize,
    component: String,
    /// The buffer is one of its producing call's declared `outputs`, which
    /// a component publishes from its output ranks
    /// (`ComponentConfig::publishes_on_every_rank`). Every member rank
    /// holds its own copy of the call's other results: its device scalars,
    /// encoder features, image, latents and KV.
    declared_output: bool,
}

/// Dispatch targeted calls and track their independently completed results.
///
/// Worker indices used throughout are positions in `workers`.
pub struct WorkerExecutor {
    /// Worker identity to its index in `workers`.
    routing: HashMap<WorkerId, usize>,
    workers: Vec<(WorkerId, WorkerGroup)>,
    executor_info: ExecutorInfo,
    /// Sum of the workers' queue depths; `submit` refuses a batch with
    /// `WouldBlock` while this many logical batches are pending.
    depth: usize,
    /// Command-channel wake, polled beside the workers' progress descriptors.
    command_wake: crate::handle::WakeSignal,
    /// Logical batches that have not yet published their terminal result.
    pending: BTreeMap<u64, PendingBatch>,
    /// Results `poll` returns in order.
    ready: VecDeque<BatchResult>,
    /// Admission descriptor of every started request until its `Finish`
    /// completes; a worker receives it as a `Start` with the request's first
    /// call there.
    admissions: HashMap<RequestKey, NewRequest>,
    /// (worker, request) pairs whose `Start` the worker has accepted.
    admitted_workers: HashSet<(usize, RequestKey)>,
    /// Producing worker and component of each output buffer of a submitted
    /// call, until the buffer is freed or its request finishes without
    /// retaining it.
    buffer_routes: HashMap<BufferId, BufferRoute>,
    /// Persistent buffer addresses are local to a physical Worker address space.
    buffer_allocations: HashMap<(usize, BufferId), BufferAllocation>,
    /// Every worker that produces or reads a buffer: the targets of its `Free`.
    buffer_workers: HashMap<BufferId, HashSet<usize>>,
    /// Tensor publications returned by producing workers, merged across the
    /// ranks that publish parts of one product. A `channel` locator carries
    /// the product's bytes, which the head therefore holds until the buffer
    /// is freed or its request finishes without retaining it.
    transfer_products: HashMap<BufferId, TensorPublication>,
    /// KV publications returned by producing workers.
    kv_transfers: HashMap<BufferId, KvTransfer>,
    /// Per-worker FIFO of submissions the `WorkerGroup` has not yet accepted.
    worker_submissions: Vec<VecDeque<WorkerSubmission>>,
    /// Last collective sequence each worker accepted. It must strictly
    /// increase per worker: a multi-rank worker treats a batch with calls
    /// whose sequence does not advance as a fatal invariant violation.
    worker_collective_seqs: Vec<u64>,
    /// A command wake was drained; the next `poll` returns `Ok(None)` so the
    /// scheduler reads its commands before waiting again.
    command_wake_pending: bool,
    /// A failure reconciled inside `submit`. `submit` refuses further batches
    /// with `WouldBlock` until `poll` returns it.
    pending_failure: Option<WorkerFailure>,
    closed: bool,
}

/// Returns, for each worker that decodes video, the encoder replicas that
/// keep every media unit it decodes on the host that decoded it.
///
/// A decoded media unit is a host product that its encoder reads in place
/// from shared storage, which is named in one host's namespace. A round deals
/// its units to each component's ranks in order, `units_per_rank` consecutive
/// positions each, so position `p` is decoded by the decoder's rank
/// `p / per_decoder` and encoded by the encoder's rank `p / per_encoder`. A
/// pairing is admissible when the encoder covers a whole round and those two
/// ranks share a host at every position. Requests are routed only to
/// admissible pairings, and a decoding worker without one is refused at
/// startup, naming the unit and the hosts it would cross.
///
/// Returns an empty map when the routing lacks a video decoding or a video
/// encoding component.
pub(crate) fn video_unit_codecs(
    routing: &std::collections::BTreeMap<uniserve_worker_ipc::MediaCall, String>,
    placements: &[(
        &WorkerId,
        &[crate::WorkerRank],
        &std::collections::BTreeMap<String, crate::ComponentConfig>,
    )],
) -> anyhow::Result<std::collections::BTreeMap<WorkerId, std::collections::BTreeSet<WorkerId>>> {
    let mut pairings = std::collections::BTreeMap::new();
    let (Some(decoder), Some(encoder)) = (
        routing.get(&uniserve_worker_ipc::MediaCall::VideoDecoding),
        routing.get(&uniserve_worker_ipc::MediaCall::VideoEncoding),
    ) else {
        return Ok(pairings);
    };
    let holders = |component: &str| {
        placements
            .iter()
            .filter_map(|(id, ranks, components)| {
                components
                    .get(component)
                    .map(|config| (*id, *ranks, config))
            })
            .collect::<Vec<_>>()
    };
    let encoders = holders(encoder);
    for (decoder_worker, decoder_ranks, decoder_config) in holders(decoder) {
        let per_decoder = decoder_config.units_per_rank.max(1);
        let round_units = decoder_config.ranks.len() * per_decoder;
        let mut admissible = std::collections::BTreeSet::new();
        let mut refusals = Vec::new();
        for (encoder_worker, encoder_ranks, encoder_config) in &encoders {
            let per_encoder = encoder_config.units_per_rank.max(1);
            let coverage = encoder_config.ranks.len() * per_encoder;
            if coverage < round_units {
                refusals.push(format!(
                    "worker {encoder_worker} encodes {coverage} units per round and worker \
                     {decoder_worker} decodes {round_units}"
                ));
                continue;
            }
            // `coverage >= round_units` keeps `position / per_encoder` within
            // the encoder's ranks.
            let crossing = (0..round_units).find_map(|position| {
                let decoding_rank = decoder_config.ranks[position / per_decoder];
                let encoding_rank = encoder_config.ranks[position / per_encoder];
                let decoding_host = decoder_ranks
                    .get(decoding_rank)
                    .map_or("<unplaced>", |rank| rank.node.as_str());
                let encoding_host = encoder_ranks
                    .get(encoding_rank)
                    .map_or("<unplaced>", |rank| rank.node.as_str());
                (decoding_host != encoding_host).then(|| {
                    format!(
                        "media unit {position} is decoded by rank {decoding_rank} of worker \
                         {decoder_worker} on host {decoding_host} and would be encoded by rank \
                         {encoding_rank} of worker {encoder_worker} on host {encoding_host}"
                    )
                })
            });
            match crossing {
                Some(refusal) => refusals.push(refusal),
                None => {
                    admissible.insert((*encoder_worker).clone());
                }
            }
        }
        anyhow::ensure!(
            !admissible.is_empty(),
            "no video codec keeps the media units of worker {decoder_worker} on their host, \
             where their encoder reads them through shared storage: {}",
            refusals.join("; ")
        );
        pairings.insert(decoder_worker.clone(), admissible);
    }
    Ok(pairings)
}

impl WorkerExecutor {
    /// Binds launched worker groups into one executor.
    ///
    /// Fails when the command wake cannot be created, `workers` is empty, a
    /// worker identity repeats or disagrees with its loaded instance, a group
    /// was initialized with transfer bindings other than `transfer`, a worker
    /// reports invalid capabilities, the workers loaded different models or
    /// checkpoints or expose different outputs for a replicated component, the
    /// media routing or video codec pairing is refused, a transfer edge names
    /// an unbound worker or rank, repeats a physical rank pair, or uses a
    /// mechanism an endpoint did not initialize or that cannot serve the edge's
    /// endpoints, or no runtime capacity view can be derived from the workers.
    /// As a side effect every group learns the deployment-wide media routing.
    pub fn try_new(
        mut workers: Vec<(WorkerId, WorkerGroup)>,
        transfer: TransferConfig,
    ) -> anyhow::Result<Self> {
        let command_wake = crate::handle::WakeSignal::new()?;
        anyhow::ensure!(
            !workers.is_empty(),
            "WorkerExecutor needs at least one worker"
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
        let mut executor_info = ExecutorInfo::from_workers(
            workers
                .iter()
                .map(|worker| (worker.0.clone(), worker.1.info().clone()))
                .collect(),
        )?;
        // The video graph may span workers, so each group learns the whole
        // deployment's routing to name the ranks that read its products.
        let media_routing = executor_info.media_routing()?;
        let placements = workers
            .iter()
            .map(|(id, worker)| {
                let (ranks, components) = worker.placement();
                (id, ranks, components)
            })
            .collect::<Vec<_>>();
        executor_info.video_codecs = video_unit_codecs(&media_routing, &placements)?;
        for (_, worker) in &mut workers {
            worker.set_media_routing(media_routing.clone());
        }

        // Every edge expands to the physical (source rank, destination rank)
        // pairs it names; an omitted rank names all ranks of its worker. A
        // pair may be bound by only one edge.
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
                let source = source.rank_info(source_rank as usize).with_context(|| {
                    format!("transfer edge source has no rank {source_rank} in its world")
                })?;
                for destination_rank in 0..destination.info().world_size {
                    if edge
                        .destination_rank
                        .is_some_and(|rank| rank != destination_rank)
                    {
                        continue;
                    }
                    let destination = destination
                        .rank_info(destination_rank as usize)
                        .with_context(|| {
                            format!(
                                "transfer edge destination has no rank {destination_rank} in its world"
                            )
                        })?;
                    anyhow::ensure!(
                        physical_edges
                            .insert((source.endpoint.clone(), destination.endpoint.clone())),
                        "physical transfer edge has conflicting bindings"
                    );
                    // An edge carries device products on one mechanism and
                    // host products on another; both endpoints must have
                    // initialized each one.
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
                            // Shared storage names a segment in one host's namespace.
                            crate::executor::TransferBackend::Shm => anyhow::ensure!(
                                !crosses_hosts,
                                "transfer edge from worker {} on host {source_host} to worker {} \
                             on host {destination_host} crosses hosts over shared storage, \
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
        // The derived runtime view is discarded; the call only refuses
        // workers from which it cannot be derived.
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

    /// Refreshes physical discovery after replacement; other instance counters remain intact.
    fn refresh_worker(&mut self, index: usize) {
        self.executor_info.workers[index].1 = self.workers[index].1.info().clone();
    }

    /// The command wake descriptor followed by every worker's progress
    /// descriptors, for `park_descriptors`.
    fn progress_fds(&self) -> Vec<libc::pollfd> {
        std::iter::once(libc::pollfd {
            fd: self.command_wake.descriptor(),
            events: libc::POLLIN,
            revents: 0,
        })
        .chain(
            self.workers
                .iter()
                .flat_map(|(_, worker)| worker.progress_fds()),
        )
        .collect()
    }

    /// Reconciles abandoned work and its dependents while retaining live physical owners.
    ///
    /// A failure is one of two kinds. With endpoints it is a loss: the
    /// `WorkerGroup` terminated its ranks and holds none of its resident
    /// requests, buffers, or publications. Without endpoints the group is
    /// intact and only the named calls, requests or batch failed; the calls
    /// of a named request still queued are retired here.
    ///
    /// Returns the failure extended to everything it invalidates: `requests`
    /// and `buffers` include the dependents found here, and `retired` lists
    /// every pending call that will now never complete, each removed from its
    /// pending batch. Batches left with no pending worker publish an empty
    /// terminal result. An error that is not a `WorkerFailure` is returned
    /// unchanged.
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

        // Without a loss, an execution error names the batch that failed on
        // this worker; only a batch already submitted here can have failed.
        let failed_run = (!lost)
            .then(|| loss.execution.as_ref().and_then(|error| error.batch_id))
            .flatten()
            .filter(|batch_id| {
                self.pending.get(batch_id).is_some_and(|pending_batch| {
                    pending_batch
                        .workers
                        .get(&index)
                        .is_some_and(|worker| worker.submitted)
                })
            });
        let mut retired = loss.retired.iter().copied().collect::<HashSet<_>>();
        let failed_calls = retired
            .iter()
            .map(|(_, request, call)| (*request, *call))
            .collect::<HashSet<_>>();
        let mut requests = loss.requests.iter().copied().collect::<HashSet<_>>();
        let mut buffers = loss.buffers.iter().cloned().collect::<HashSet<_>>();

        // A lost incarnation's locations are gone. A publication that no longer
        // covers its whole value is invalid; one that other ranks or workers
        // still cover in full stays usable.
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

        // Buffers produced on the lost worker that were never published are
        // gone with it, and a failed call's outputs will never be produced.
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
        // On a loss, every request with a call still pending on the worker is
        // affected.
        if lost {
            for pending in self.pending.values() {
                if let Some(worker) = pending.workers.get(&index) {
                    requests.extend(worker.calls.keys().map(|(request, _)| *request));
                }
            }
        }

        // Propagate to a fixed point. A call reading an invalid buffer taints
        // its request. Abandoning an unstarted producer also invalidates its
        // future buffers. Already running work on another instance keeps its
        // real completion path, so an in-flight call only taints its request.
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

        // Remove the tainted requests' calls, and the dependencies no remaining
        // call reads, from every queued submission. A lost worker's queued
        // commands are dropped too, since the state they act on is gone. A
        // submission left empty is discarded.
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
                    .flat_map(|(call, _)| call.input_buffers())
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

        // Retire this worker's outstanding calls in every batch on a loss, or in
        // the failed batch only. The group sends no report for such a batch, so
        // the entry stops awaiting one. Commands of tainted requests are
        // marked `Retired`, and in the failed batch the reported requests'
        // commands are marked `Failed`.
        for (batch_id, pending_batch) in &mut self.pending {
            if let Some(worker) = pending_batch.workers.get_mut(&index)
                && (lost || (worker.submitted && failed_run == Some(*batch_id)))
            {
                retired.extend(
                    worker
                        .calls
                        .keys()
                        .map(|identity| (*batch_id, identity.0, identity.1)),
                );
                worker.submitted = false;
            }
            pending_batch.set_command_outcome(&requests, CommandOutcome::Retired);
        }
        if let Some(batch_id) = failed_run {
            let failed_requests = loss.requests.iter().copied().collect();
            if let Some(pending) = self.pending.get_mut(&batch_id) {
                pending.set_command_outcome(&failed_requests, CommandOutcome::Failed);
            }
        }

        // Remove every retired call from its pending batch. Each must still be
        // pending on some worker, or the bookkeeping has diverged.
        loss.retired.clear();
        for (batch_id, request, call) in retired {
            let pending_batch = self
                .pending
                .get_mut(&batch_id)
                .context("retired call has no pending batch")?;
            let removed = pending_batch
                .workers
                .values_mut()
                .any(|worker| worker.calls.remove(&(request, call)).is_some());
            anyhow::ensure!(removed, "abandoned call is not pending");
            loss.retired.push((batch_id, request, call));
        }

        // A lost worker holds no admissions or allocations: a later call on it
        // sends its request's `Start` again, and no `Free` targets it.
        if lost {
            self.admitted_workers.retain(|(worker, _)| *worker != index);
            self.buffer_allocations
                .retain(|(worker, _), _| *worker != index);
            for workers in self.buffer_workers.values_mut() {
                workers.remove(&index);
            }
        }

        // Drop workers with nothing left to report, then complete the batches
        // that no worker still owes a result.
        for (batch_id, pending) in &mut self.pending {
            pending.workers.retain(|worker_index, worker| {
                worker.submitted
                    || !worker.calls.is_empty()
                    || self.worker_submissions[*worker_index]
                        .iter()
                        .any(|submission| submission.batch.id == *batch_id)
            });
        }
        let batches = self.pending.keys().copied().collect::<Vec<_>>();
        for batch_id in batches {
            if self
                .pending
                .get(&batch_id)
                .is_some_and(|pending| pending.workers.is_empty())
            {
                self.publish_result(WorkerResult {
                    batch_id,
                    done: true,
                    results: Vec::new(),
                    products: Vec::new(),
                    worker_exec_us: None,
                    forward_stats: None,
                })?;
            }
        }
        loss.requests = requests.into_iter().collect();
        loss.buffers = buffers.into_iter().collect();
        Ok(loss)
    }

    /// Records the admission descriptor of each request a batch starts.
    ///
    /// Fails when a request is admitted again, before its `Finish` completes,
    /// with a descriptor that differs from the recorded one; descriptors
    /// recorded earlier in the same call stay recorded.
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

    /// Returns the `Start` descriptors `worker_index` needs before `calls`.
    ///
    /// One descriptor per request in `calls` that the worker has not yet
    /// accepted a `Start` for, carrying the placement's worker-local request
    /// row when it names one and the admission's canonical row otherwise.
    /// Fails when a request has no recorded admission or its calls name
    /// different rows.
    fn admissions_for(
        &self,
        worker_index: usize,
        calls: &[(Call, RequestPlacement)],
    ) -> anyhow::Result<Vec<NewRequest>> {
        let mut admissions = Vec::new();
        let mut rows = HashMap::new();
        for (call, placement) in calls {
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
            let row = placement
                .request_pool_idx
                .unwrap_or(admission.request_pool_idx);
            if let Some(existing) = rows.insert(call.request_key, row) {
                anyhow::ensure!(
                    existing == row,
                    "worker {worker_index} received conflicting request rows for {:?}",
                    call.request_key
                );
                continue;
            }
            let mut admission = admission.clone();
            admission.request_pool_idx = row;
            admissions.push(admission);
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
                    .flat_map(|pending| pending.workers.iter())
                    .filter_map(|(worker_index, worker)| {
                        worker
                            .calls
                            .keys()
                            .any(|(key, _)| *key == request)
                            .then_some(*worker_index)
                    }),
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
    ///
    /// Returns sorted worker indices. `Start` has no target of its own: a
    /// worker receives it with the request's first call (`admissions_for`).
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
    ///
    /// The caller has established that both run on the producer's worker.
    /// Returns true for the producing component itself when each of its ranks
    /// holds its own copy of the buffer, or for two components on the same
    /// single rank; false otherwise, including when either component of a
    /// pair is not loaded on the worker.
    fn shares_product_storage(&self, producer: &BufferRoute, consumer_entry: &str) -> bool {
        let entries = &self.workers[producer.worker_index].1.info().components;
        let source = entries
            .iter()
            .find(|entry| entry.name == producer.component);
        let destination = entries.iter().find(|entry| entry.name == consumer_entry);
        if producer.component == consumer_entry {
            // A consumer call on the component that produced the product runs on
            // the same ranks, and each rank reads the copy it holds when every
            // rank holds one. Every rank holds a call's own results beside its
            // declared outputs. A declared output is written by every rank of a
            // distributed component, whose round deals the same media units in
            // the same order to the same ranks, and of a sequence-parallel one,
            // whose ranks each write their shard; a tensor-parallel or pipelined
            // component publishes it from its output ranks alone, so its other
            // ranks read the product's publication. Without a description of
            // the component, its ranks are taken to hold their own copies.
            return !producer.declared_output
                || source.is_none_or(|source| source.config.publishes_on_every_rank());
        }
        source
            .zip(destination)
            .is_some_and(|(source, destination)| {
                source.config.ranks.len() == 1 && source.config.ranks == destination.config.ranks
            })
    }

    /// Submits or queues one worker-local batch while preserving collective order.
    ///
    /// Prepends the `Start` of every request among its calls that the worker
    /// has not yet admitted, attaches the publications of the submission's
    /// dependencies, which the caller has checked are published, and lowers
    /// the batch to the wire. Returns `Deferred` when the group refuses it
    /// with `WouldBlock` (not ready, or no free slot), and `Refused` when the
    /// group cannot bind inputs to the ranks reading them; in both cases
    /// nothing is recorded and the caller requeues the submission.
    /// `Submitted` records the submission, its collective sequence, and the
    /// admissions it carried. Errors come from resolving admissions, from
    /// building or validating the wire batch, or from the group; a group that
    /// replaced its ranks reports a `WorkerFailure`.
    fn submit_worker(
        &mut self,
        worker_index: usize,
        submission: &WorkerSubmission,
    ) -> anyhow::Result<Dispatch> {
        let batch = &submission.batch;
        let admissions = self.admissions_for(worker_index, &batch.requests)?;
        let request_keys = admissions
            .iter()
            .map(|admission| admission.request_key)
            .collect::<Vec<_>>();
        let commands = admissions
            .into_iter()
            .map(|request| BatchCommand::Start {
                request: Box::new(request),
            })
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

        // The sequence is committed only once the group accepts the batch, so
        // a `WouldBlock` retry carries the same value.
        let collective_seq = self.worker_collective_seqs[worker_index]
            .checked_add(1)
            .context("collective sequence space exhausted")?;
        let mut wire = ExecutionBatch {
            id: batch.id,
            requests: batch.requests.clone(),
            commands,
            input_transfers: inputs,
            kv_inputs,
        }
        .into_protocol(collective_seq)?;

        // A transferred persistent input needs its destination-local span.
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
            // A routed destination may reserve a different address from its
            // producer, and that allocation takes precedence. Without one the
            // producer's allocation is used.
            let params = self
                .buffer_allocations
                .get(&(worker_index, *dependency))
                .or_else(|| {
                    let producer = self.buffer_routes.get(dependency)?;
                    self.buffer_allocations
                        .get(&(producer.worker_index, *dependency))
                })
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

        // Drop every location whose source endpoint is not the current
        // incarnation of a bound rank.
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

        // Routing stays on the head: the group names each call's reading
        // ranks from the workers its request was routed to.
        let readers = batch
            .requests
            .iter()
            .filter_map(|(call, placement)| {
                let readers = placement.readers.clone()?;
                Some(((call.request_key, call.call_id), readers))
            })
            .collect();
        match self.workers[worker_index].1.submit_batch(wire, &readers) {
            Ok(()) => {}
            Err(BatchSubmitError::WouldBlock(_)) => return Ok(Dispatch::Deferred),
            Err(BatchSubmitError::Failed(error)) => {
                // Only an input refusal leaves the group intact; every other
                // failure propagates unchanged.
                let refusal = error.downcast::<UnroutableInputs>()?;
                let failure = self.refused_inputs(worker_index, batch, refusal)?;
                return Ok(Dispatch::Refused(Box::new(failure)));
            }
        }
        self.worker_collective_seqs[worker_index] = collective_seq;
        let pending = self
            .pending
            .get_mut(&batch.id)
            .context("submitted batch is not pending")?;
        let worker = pending
            .workers
            .get_mut(&worker_index)
            .context("submitted worker is not pending")?;
        anyhow::ensure!(!worker.submitted, "worker batch was submitted twice");
        worker.submitted = true;
        self.admitted_workers
            .extend(request_keys.into_iter().map(|key| (worker_index, key)));
        Ok(Dispatch::Submitted)
    }

    /// The failure of the requests whose calls read inputs a worker refused.
    ///
    /// A product with no location on an edge to its consumer is a property of
    /// that product and that edge, such as a device product that fell back to
    /// host bytes crossing an edge that carries only a device mechanism. It
    /// leaves both instances and the batch's other calls intact, so the
    /// failure invalidates no endpoint or buffer and names only the requests
    /// reading the refused products. Reconciliation retires those requests'
    /// calls from the requeued submission, whose remaining calls and
    /// commands dispatch later.
    ///
    /// # Errors
    ///
    /// Fails when no call of `batch` reads a refused product: nothing could
    /// then be retired, and the requeued submission would be refused again.
    fn refused_inputs(
        &self,
        worker_index: usize,
        batch: &ExecutionBatch,
        refusal: UnroutableInputs,
    ) -> anyhow::Result<WorkerFailure> {
        let requests = batch
            .requests
            .iter()
            .filter(|(call, _)| {
                call.input_buffers()
                    .any(|buffer| refusal.buffers.contains(&buffer))
            })
            .map(|(call, _)| call.request_key)
            .collect::<HashSet<_>>();
        anyhow::ensure!(
            !requests.is_empty(),
            "refused input products are read by no call: {refusal}"
        );
        Ok(WorkerFailure {
            worker_id: self.workers[worker_index].0.clone(),
            endpoints: Vec::new(),
            requests: requests.into_iter().collect(),
            retired: Vec::new(),
            buffers: Vec::new(),
            execution: None,
            message: refusal.to_string(),
        })
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
                match self.submit_worker(worker_index, &submission)? {
                    Dispatch::Submitted => progressed = true,
                    Dispatch::Deferred => {
                        self.worker_submissions[worker_index].push_front(submission);
                    }
                    Dispatch::Refused(failure) => {
                        // The refused submission keeps its place, so the
                        // reconciliation of this failure retires the named
                        // requests' calls from it.
                        self.worker_submissions[worker_index].push_front(submission);
                        return Err((*failure).into());
                    }
                }
            }
            if !progressed {
                return Ok(());
            }
        }
    }

    /// Advances worker command execution until no immediate progress remains.
    ///
    /// Latches a command wake, dispatches ready submissions, drains every
    /// worker's available reports without blocking, and dispatches again for
    /// the dependencies those reports published.
    fn pump(&mut self) -> anyhow::Result<()> {
        self.command_wake_pending |= self.command_wake.drain()?;
        self.dispatch_ready()?;
        for worker_index in 0..self.workers.len() {
            loop {
                let polled = self.workers[worker_index].1.poll_batch(Duration::ZERO);
                // Taken before the poll error propagates, so a recovery or
                // close made while polling reaches `executor_info` either way.
                if self.workers[worker_index].1.take_readiness_change() {
                    self.refresh_worker(worker_index);
                }
                let report = polled?;
                let Some(report) = report else {
                    break;
                };
                self.route_result(worker_index, report)?;
            }
        }
        self.dispatch_ready()
    }

    /// Routes a worker result into its worker aggregate and publishes transferable products.
    ///
    /// Validates the report against the worker's pending calls, stores the
    /// product and KV publications it returns, and publishes its completions.
    /// When some calls failed, the report, failed completions included, is
    /// still published and the method then returns a `WorkerFailure` naming
    /// the failed calls' requests and the buffers those calls would have
    /// produced.
    fn route_result(&mut self, worker_index: usize, report: WorkerResult) -> anyhow::Result<()> {
        let mut report = report;
        let failed_calls = report
            .results
            .iter()
            .filter(|completion| completion.output.status == uniserve_worker_ipc::CallStatus::Error)
            .map(|completion| (completion.output.request_key, completion.output.call_id))
            .collect::<HashSet<_>>();
        if !failed_calls.is_empty() {
            // The worker names the code; the request's caller sees only that
            // its dependent work could not complete, so the code is recorded
            // here, where the failing call and worker are known.
            let failed = report
                .results
                .iter()
                .filter(|completion| {
                    completion.output.status == uniserve_worker_ipc::CallStatus::Error
                })
                .map(|completion| {
                    (
                        completion.output.request_key,
                        completion.output.call_id,
                        completion.output.error_code,
                    )
                })
                .collect::<Vec<_>>();
            tracing::warn!(
                worker = %self.workers[worker_index].0,
                batch_id = report.batch_id,
                ?failed,
                "worker returned failed calls"
            );
        }
        let failed_buffers = self
            .buffer_routes
            .keys()
            .filter(|product| failed_calls.contains(&(product.owner, product.producer_call_id)))
            .cloned()
            .collect();

        // Every completion must answer a distinct call still pending on this
        // worker, and a terminal report must answer all of them.
        let batch_id = report.batch_id;
        {
            let pending_batch = self
                .pending
                .get_mut(&batch_id)
                .ok_or_else(|| anyhow::anyhow!("worker returned unknown batch {batch_id}"))?;
            let worker = pending_batch
                .workers
                .get(&worker_index)
                .context("worker returned an unexpected batch")?;
            anyhow::ensure!(
                worker.submitted,
                "worker returned a batch before submission"
            );
            let mut seen = HashSet::new();
            for completion in &report.results {
                let identity = (completion.output.request_key, completion.output.call_id);
                let kind = worker
                    .calls
                    .get(&identity)
                    .context("worker returned an unknown call")?;
                anyhow::ensure!(seen.insert(identity), "worker returned a duplicate call");
                anyhow::ensure!(
                    completion.output.code == *kind,
                    "worker returned a computation code that disagrees with its call"
                );
            }
            anyhow::ensure!(
                !report.done || seen.len() == worker.calls.len(),
                "worker {worker_index} ended batch {batch_id} before every call completed"
            );
        }

        // Store returned publications. Each must come from its buffer's
        // producing worker; a later report for the same buffer adds locations.
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

        // Keep outstanding identities available to recovery until all returned
        // publications have passed their owner and layout checks.
        let pending = self
            .pending
            .get_mut(&batch_id)
            .context("result has no pending batch")?;
        if report.done {
            pending.workers.remove(&worker_index);
        } else {
            // Computation can finish before command-only ranks acknowledge.
            // Keep that worker's ownership until its terminal report arrives.
            let worker = pending
                .workers
                .get_mut(&worker_index)
                .context("result has no pending worker")?;
            for completion in &report.results {
                worker
                    .calls
                    .remove(&(completion.output.request_key, completion.output.call_id));
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
    ///
    /// The batch is terminal once no worker remains in its pending record;
    /// callers remove workers from it as they finish. A terminal result
    /// carries a receipt for every lifecycle command other than `Start`, with
    /// its recorded outcome or `Applied`, and applies the executor's own
    /// retirement bookkeeping for each command that did not fail. A result is
    /// queued for `poll` when it carries completions or is terminal.
    fn publish_result(&mut self, report: WorkerResult) -> anyhow::Result<()> {
        let batch_id = report.batch_id;
        let std::collections::btree_map::Entry::Occupied(entry) = self.pending.entry(batch_id)
        else {
            anyhow::bail!("result has no pending batch");
        };
        let pending = entry.get();
        let done = pending.workers.is_empty();
        let mut result = logical_result(report, done, &pending.commands);
        for receipt in &mut result.command_results {
            receipt.outcome = pending
                .command_outcomes
                .get(&receipt.command_index)
                .copied()
                .unwrap_or(CommandOutcome::Applied);
        }
        if done {
            let pending = entry.remove();
            for (index, command) in pending.commands.into_iter().enumerate() {
                // A failed command keeps physical ownership, so what it would
                // release stays indexed until a later release succeeds.
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
                        self.buffer_allocations
                            .retain(|(_, identity), _| *identity != buffer);
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

    /// Retires request routing while preserving independently owned product locations.
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
            .retain(|(_, buffer), _| buffer.owner != request || retained.contains(buffer));
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
        // Submissions already queued for the worker claim its free group slots.
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
    ///
    /// Returns `WouldBlock` while a reconciled failure awaits `poll`, while
    /// `depth` logical batches are pending, or when a target worker has no
    /// free slot. A `WorkerFailure` raised while dispatching is reconciled and
    /// held for the next `poll`, and the batch counts as submitted: its fate
    /// arrives through that failure and the batch's results. A reconciliation
    /// error is returned as `Failed`. Any other error while partitioning or
    /// dispatching drops the batch's pending record, but state registered
    /// before the error, such as the admissions it carries, the routes,
    /// holders, and allocations of its calls, and submissions already queued,
    /// is not rolled back.
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
            // Each per-worker vector is indexed by worker index.
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

            // Route each call to its worker, check the worker can run it, and
            // register the producer and holders of its outputs and the
            // worker-local buffer allocations its placement names.
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
                        .supported_calls
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
                let declared = call
                    .outputs
                    .iter()
                    .map(TensorRef::buffer_id)
                    .collect::<HashSet<_>>();
                for output in call.output_buffers() {
                    let route = BufferRoute {
                        worker_index: call_worker,
                        component: call.component.clone(),
                        declared_output: declared.contains(&output),
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
                    // paged latents, KV, and request-relay values that have no
                    // buffer allocation.
                    self.buffer_workers
                        .entry(output)
                        .or_default()
                        .insert(call_worker);
                }
                for params in &placement.buffers {
                    if let Some(existing) = self
                        .buffer_allocations
                        .insert((call_worker, params.buffer), *params)
                    {
                        anyhow::ensure!(
                            existing == *params,
                            "buffer identity was assigned conflicting worker-local allocations"
                        );
                    }
                }
                for output in call.buffer_outputs() {
                    anyhow::ensure!(
                        placement
                            .buffers
                            .iter()
                            .any(|params| params.buffer == output.buffer_id()),
                        "persistent product {:?} has no scheduler params",
                        output
                    );
                }
                call_routes.insert((call.request_key, call.call_id), call_worker);
                for input in call.input_buffers() {
                    input_routes
                        .entry(input)
                        .or_insert_with(HashSet::new)
                        .insert(call_worker);
                }
            }

            // External publications go to every worker with a call reading them.
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

            // Resolve every input: supplied externally, read in place from a
            // producer sharing the reader's storage, or a dependency the
            // worker's submission waits on until its producer publishes it.
            for (call, _) in &batch.requests {
                let consumer_worker = call_routes[&(call.request_key, call.call_id)];
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
                let worker = call_routes[&(call.request_key, call.call_id)];
                worker_ops[worker].push((call, placement));
            }

            // `Start` reaches a worker with the request's first call there;
            // other commands go to every worker holding what they release.
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

            // Queue one submission per worker that has calls or commands, and
            // register the batch before dispatching any of them.
            let mut pending_workers = HashMap::new();
            for (worker_index, (((calls, commands), input_products), dependencies)) in worker_ops
                .into_iter()
                .zip(worker_commands)
                .zip(worker_inputs)
                .zip(worker_dependencies)
                .enumerate()
            {
                if calls.is_empty() && commands.is_empty() {
                    anyhow::ensure!(
                        dependencies.is_empty(),
                        "product dependency has no consuming worker submission"
                    );
                    continue;
                }
                pending_workers.insert(
                    worker_index,
                    PendingWorker {
                        submitted: false,
                        calls: calls
                            .iter()
                            .map(|(call, _)| ((call.request_key, call.call_id), call.code))
                            .collect(),
                    },
                );
                // Keep call-owned resources together while dependencies wait.
                // Rank indices and collective order are derived at physical submission.
                self.worker_submissions[worker_index].push_back(WorkerSubmission::build(
                    batch_id,
                    calls,
                    commands,
                    input_products,
                    std::mem::take(&mut worker_kv_inputs[worker_index]),
                    dependencies,
                ));
            }
            self.pending.insert(
                batch_id,
                PendingBatch {
                    workers: pending_workers,
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

            // A batch with no worker part, such as one whose commands have no
            // remaining holder, completes at once.
            if self
                .pending
                .get(&batch_id)
                .is_some_and(|pending| pending.workers.is_empty())
            {
                self.publish_result(WorkerResult {
                    batch_id,
                    done: true,
                    results: Vec::new(),
                    products: Vec::new(),
                    worker_exec_us: None,
                    forward_stats: None,
                })?;
            }

            // A freed buffer loses its producer route and publications now, so
            // a later call that reads it is refused unless its batch supplies
            // a publication. Its holders and allocations stay indexed until the
            // `Free` completes, and remain if the release fails.
            for command in &batch.commands {
                if let BatchCommand::Free { buffer } = command {
                    self.transfer_products
                        .retain(|identity, _| *identity != *buffer);
                    self.buffer_routes
                        .retain(|identity, _| *identity != *buffer);
                    self.kv_transfers.remove(buffer);
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
    ///
    /// A failure held by `submit` is returned first. A drained command wake
    /// makes the call return `Ok(None)` at once, before or after waiting, so
    /// the scheduler can read its commands. A worker failure found while
    /// pumping is reconciled and returned as a `WorkerFailure`; any other
    /// error is returned unchanged.
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
        crate::worker::park_descriptors(&self.progress_fds(), timeout)?;
        if let Err(error) = self.pump() {
            return Err(self.reconcile_worker_failure(error)?.into());
        }
        if std::mem::take(&mut self.command_wake_pending) {
            return Ok(None);
        }
        Ok(self.ready.pop_front())
    }

    /// Closes every worker group and refuses later submissions.
    ///
    /// Every group is closed even when an earlier one fails; the first error
    /// is returned.
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
        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod placement_tests {
    use super::video_unit_codecs;
    use crate::WorkerRank;
    use crate::executor::{ComponentConfig, WorkerId};
    use std::collections::BTreeMap;
    use uniserve_worker_ipc::MediaCall;

    fn rank(node: &str, device: &str) -> WorkerRank {
        WorkerRank {
            node: node.to_owned(),
            device: device.to_owned(),
        }
    }

    fn distributed(ranks: Vec<usize>, units_per_rank: usize) -> ComponentConfig {
        ComponentConfig {
            ranks,
            parallel_config: Default::default(),
            distribution: Some(uniserve_core::ComponentDistribution::TemporalUnits),
            units_per_rank,
        }
    }

    fn routing() -> BTreeMap<MediaCall, String> {
        BTreeMap::from([
            (MediaCall::VideoDecoding, "video_decoder".to_owned()),
            (MediaCall::VideoEncoding, "video_codec".to_owned()),
        ])
    }

    #[test]
    fn a_decoded_unit_names_the_host_worker_ranks_that_read_it() {
        // The decoder's product is read by the video codec on the host
        // worker: the encoder rank dealt this decoder rank's position is
        // named, none of the decoder's own ranks, and the host worker's
        // slots follow the model worker's run. A consumer that is not
        // distributed reads the whole product on every rank.
        use crate::executor::TransferConfig;
        use crate::worker::instance::media_consumer_slots;

        let model_components =
            BTreeMap::from([("video_decoder".to_owned(), distributed(vec![0, 1, 2, 3], 1))]);
        let host_components =
            BTreeMap::from([("video_codec".to_owned(), distributed(vec![0, 1], 2))]);
        let peers = BTreeMap::from([
            ("host".to_owned(), host_components),
            ("model".to_owned(), model_components.clone()),
        ]);
        let mut transfer = TransferConfig::default();
        transfer.worker_ranks.insert("host".to_owned(), 2);
        transfer.worker_ranks.insert("model".to_owned(), 4);
        let slots = media_consumer_slots(
            &[MediaCall::VideoEncoding],
            &routing(),
            &BTreeMap::new(),
            "model",
            &model_components,
            &peers,
            &transfer,
            &[0, 1, 2, 3],
            2,
            model_components.get("video_decoder"),
        );
        // Position 2 of the round, two units per host rank: host rank 1.
        assert_eq!(slots, [transfer.acknowledgment_slot("host", 1)]);

        let routing = BTreeMap::from([
            (MediaCall::AudioDecoding, "audio_decoder".to_owned()),
            (MediaCall::AudioEncoding, "muxer".to_owned()),
        ]);
        let model_components =
            BTreeMap::from([("audio_decoder".to_owned(), distributed(vec![0, 1, 2, 3], 1))]);
        let host_components = BTreeMap::from([(
            "muxer".to_owned(),
            ComponentConfig {
                ranks: vec![0],
                parallel_config: Default::default(),
                distribution: None,
                units_per_rank: 1,
            },
        )]);
        let peers = BTreeMap::from([
            ("host".to_owned(), host_components),
            ("model".to_owned(), model_components.clone()),
        ]);
        let slots = media_consumer_slots(
            &[MediaCall::AudioEncoding],
            &routing,
            &BTreeMap::new(),
            "model",
            &model_components,
            &peers,
            &transfer,
            &[0, 1, 2, 3],
            3,
            model_components.get("audio_decoder"),
        );
        assert_eq!(slots, [transfer.acknowledgment_slot("host", 0)]);
    }

    #[test]
    fn a_shared_producer_names_only_the_replica_its_request_was_routed_to() {
        // A text encoder on its own worker feeds latent preparation on two
        // denoiser replicas. A request routed to flow-1 is read there alone,
        // so its product names flow-1's reading rank and not flow-0's. A
        // component with no route yet (a call placed only when it runs) is
        // read by whichever replica serves it, so each one is named.
        use crate::executor::TransferConfig;
        use crate::worker::instance::media_consumer_slots;

        let text = BTreeMap::from([(
            "text_encoder".to_owned(),
            ComponentConfig::parallel(vec![0], Default::default()),
        )]);
        let denoiser = || {
            BTreeMap::from([(
                "denoiser".to_owned(),
                ComponentConfig::parallel(vec![0], Default::default()),
            )])
        };
        let peers = BTreeMap::from([
            ("flow-0".to_owned(), denoiser()),
            ("flow-1".to_owned(), denoiser()),
            ("text".to_owned(), text.clone()),
        ]);
        let routing = BTreeMap::from([(MediaCall::LatentPreparation, "denoiser".to_owned())]);
        let mut transfer = TransferConfig::default();
        for worker in ["text", "flow-0", "flow-1"] {
            transfer.worker_ranks.insert(worker.to_owned(), 1);
        }
        let slots = |routed: &BTreeMap<String, WorkerId>| {
            media_consumer_slots(
                &[MediaCall::LatentPreparation],
                &routing,
                routed,
                "text",
                &text,
                &peers,
                &transfer,
                &[0],
                0,
                text.get("text_encoder"),
            )
        };

        let routed = BTreeMap::from([("denoiser".to_owned(), WorkerId("flow-1".to_owned()))]);
        assert_eq!(slots(&routed), [transfer.acknowledgment_slot("flow-1", 0)]);

        assert_eq!(
            slots(&BTreeMap::new()),
            [
                transfer.acknowledgment_slot("flow-0", 0),
                transfer.acknowledgment_slot("flow-1", 0),
            ]
        );
    }

    #[test]
    fn a_decoded_unit_names_only_the_encoder_on_its_route() {
        // Two hosts each hold a video codec worker. A request decoded on
        // host 0's model worker is routed to host 0's encoder, so the unit
        // names that encoder's reading rank; host 1's encoder rank at the
        // same position never reads it.
        use crate::executor::TransferConfig;
        use crate::worker::instance::media_consumer_slots;

        let model = BTreeMap::from([("video_decoder".to_owned(), distributed(vec![0, 1], 1))]);
        let encoder = || BTreeMap::from([("video_codec".to_owned(), distributed(vec![0, 1], 1))]);
        let peers = BTreeMap::from([
            ("encoder-0".to_owned(), encoder()),
            ("encoder-1".to_owned(), encoder()),
            ("model-0".to_owned(), model.clone()),
        ]);
        let mut transfer = TransferConfig::default();
        for worker in ["encoder-0", "encoder-1", "model-0"] {
            transfer.worker_ranks.insert(worker.to_owned(), 2);
        }
        let routed = BTreeMap::from([("video_codec".to_owned(), WorkerId("encoder-0".to_owned()))]);

        let slots = media_consumer_slots(
            &[MediaCall::VideoEncoding],
            &routing(),
            &routed,
            "model-0",
            &model,
            &peers,
            &transfer,
            &[0, 1],
            1,
            model.get("video_decoder"),
        );

        assert_eq!(slots, [transfer.acknowledgment_slot("encoder-0", 1)]);
    }

    fn pairing(
        placements: &[(&WorkerId, &[WorkerRank], &BTreeMap<String, ComponentConfig>)],
    ) -> anyhow::Result<BTreeMap<WorkerId, std::collections::BTreeSet<WorkerId>>> {
        video_unit_codecs(&routing(), placements)
    }

    #[test]
    fn units_encoded_on_their_own_host_are_admitted() {
        let model = WorkerId("model".to_owned());
        let host = WorkerId("host".to_owned());
        let model_ranks = vec![
            rank("a", "cuda:0"),
            rank("a", "cuda:1"),
            rank("b", "cuda:0"),
            rank("b", "cuda:1"),
        ];
        let host_ranks = vec![
            rank("a", "cpu"),
            rank("a", "cpu"),
            rank("b", "cpu"),
            rank("b", "cpu"),
        ];
        let model_components =
            BTreeMap::from([("video_decoder".to_owned(), distributed(vec![0, 1, 2, 3], 1))]);
        let host_components =
            BTreeMap::from([("video_codec".to_owned(), distributed(vec![0, 1, 2, 3], 1))]);
        let pairings = pairing(&[
            (&model, &model_ranks, &model_components),
            (&host, &host_ranks, &host_components),
        ])
        .expect("each unit is encoded on its own host");
        assert_eq!(pairings[&model], [host].into());
    }

    #[test]
    fn a_decoder_dealing_several_units_per_rank_keeps_them_on_its_host() {
        // Two units per decoding rank: positions 0 to 3 come from host a's
        // decoders and go to host a's encoders, positions 4 to 7 likewise on
        // host b.
        let model = WorkerId("model".to_owned());
        let host = WorkerId("host".to_owned());
        let model_ranks = vec![
            rank("a", "cuda:0"),
            rank("a", "cuda:1"),
            rank("b", "cuda:0"),
            rank("b", "cuda:1"),
        ];
        let host_ranks = [["a"; 4], ["b"; 4]]
            .concat()
            .into_iter()
            .map(|node| rank(node, "cpu"))
            .collect::<Vec<_>>();
        let model_components =
            BTreeMap::from([("video_decoder".to_owned(), distributed(vec![0, 1, 2, 3], 2))]);
        let host_components =
            BTreeMap::from([("video_codec".to_owned(), distributed((0..8).collect(), 1))]);
        pairing(&[
            (&model, &model_ranks, &model_components),
            (&host, &host_ranks, &host_components),
        ])
        .expect("each host's units are encoded on it");

        // Host a holding three encoders sends its fourth unit to host b.
        let host_ranks = [["a"; 3].as_slice(), ["b"; 5].as_slice()]
            .concat()
            .into_iter()
            .map(|node| rank(node, "cpu"))
            .collect::<Vec<_>>();
        let error = pairing(&[
            (&model, &model_ranks, &model_components),
            (&host, &host_ranks, &host_components),
        ])
        .expect_err("the fourth unit leaves host a");
        assert!(error.to_string().contains("media unit 3"), "{error}");
    }

    #[test]
    fn every_shipped_deployment_encodes_its_units_on_their_host() {
        // Every video deployment file shipped in `configs/fast_h3/` and
        // `configs/minimax_h3/` must pass the placement checks the engine
        // applies at startup, including the pairing of every decoding worker
        // with an encoder on its host.
        for family in ["fast_h3", "minimax_h3"] {
            check_shipped_deployments(family);
        }
    }

    fn check_shipped_deployments(family: &str) {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs")
            .join(family);
        let mut checked = 0;
        for entry in std::fs::read_dir(&root).expect("the deployment directory exists") {
            let path = entry.expect("readable entry").path();
            if path.extension().is_none_or(|extension| extension != "json") {
                continue;
            }
            let workers: Vec<crate::WorkerConfig> =
                serde_json::from_str(&std::fs::read_to_string(&path).expect("readable file"))
                    .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            crate::WorkerConfig::validate_all(&workers)
                .unwrap_or_else(|error| panic!("{}: {error:#}", path.display()));
            let ids = workers
                .iter()
                .map(|worker| worker.id.clone())
                .collect::<Vec<_>>();
            let placements = workers
                .iter()
                .zip(&ids)
                .map(|(worker, id)| (id, worker.ranks.as_slice(), &worker.components))
                .collect::<Vec<_>>();
            let pairings = pairing(&placements)
                .unwrap_or_else(|error| panic!("{}: {error:#}", path.display()));
            let decoders = workers
                .iter()
                .filter(|worker| worker.components.contains_key("video_decoder"))
                .count();
            assert_eq!(pairings.len(), decoders, "{}", path.display());
            checked += 1;
        }
        assert!(checked > 0, "no {family} deployment was checked");
    }

    #[test]
    fn each_replica_decoder_pairs_with_the_encoders_on_its_host() {
        // Single-device flow replicas decode one unit per round, so every
        // round is position zero: an encoder replica is admissible only on
        // the replica's own host.
        let flows = [
            (WorkerId("flow-a".to_owned()), vec![rank("a", "cuda:0")]),
            (WorkerId("flow-b".to_owned()), vec![rank("b", "cuda:0")]),
        ];
        let encoders = [
            (WorkerId("encoder-a".to_owned()), vec![rank("a", "cpu")]),
            (WorkerId("encoder-b".to_owned()), vec![rank("b", "cpu")]),
        ];
        let decoder = BTreeMap::from([("video_decoder".to_owned(), distributed(vec![0], 1))]);
        let encoder = BTreeMap::from([("video_codec".to_owned(), distributed(vec![0], 1))]);
        let placements = flows
            .iter()
            .map(|(id, ranks)| (id, ranks.as_slice(), &decoder))
            .chain(
                encoders
                    .iter()
                    .map(|(id, ranks)| (id, ranks.as_slice(), &encoder)),
            )
            .collect::<Vec<_>>();

        let pairings = pairing(&placements).expect("every host has an encoder");

        assert_eq!(pairings[&flows[0].0], [encoders[0].0.clone()].into());
        assert_eq!(pairings[&flows[1].0], [encoders[1].0.clone()].into());

        // Without an encoder on host b, the replica there is refused by name
        // even though another replica pairs.
        let error = pairing(&placements[..3]).expect_err("host b has no encoder");
        let message = error.to_string();
        assert!(message.contains("worker flow-b"), "{message}");
        assert!(
            message.contains("host b") && message.contains("host a"),
            "{message}"
        );
    }
}
