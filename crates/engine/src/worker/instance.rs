//! Physical worker fan-out, completion agreement, and process recovery.

use crate::executor::{OpResult, WorkerResult};
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use super::{RankProcess, RunSubmitError};
use crate::executor::{WorkerExecError, WorkerFailure};
use anyhow::Context;
use sha2::{Digest as _, Sha256};
use uniserve_worker_ipc::{BatchCommand, RequestKey, ScheduleBatch, WorkerInfo};

use crate::worker::WorkerProcessArgs;

/// Maximum interval without rank progress before an in-flight run is treated as failed.
const NEXT_RESULT_DEADLINE: Duration = Duration::from_secs(300);

impl WorkerProcessArgs {
    /// Launches and connects every rank in one physical worker group.
    fn launch(&self, cancel: Option<Arc<AtomicBool>>) -> anyhow::Result<Vec<RankProcess>> {
        let cancel = cancel.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        crate::WorkerConfig::validate_members(&self.ranks, &self.entries)?;
        anyhow::ensure!(
            self.ranks.iter().all(|rank| rank.node == "localhost"),
            "remote worker process launch is unavailable"
        );
        // Local ranks share a unique filesystem rendezvous for their entire
        // lifetime. Probing and releasing a TCP port cannot reserve it for the
        // Python store that starts after process creation.
        let rendezvous = if self.ranks.len() > 1 {
            Some(Arc::new(
                tempfile::Builder::new()
                    .prefix("uniserve-rendezvous-")
                    .tempdir()?,
            ))
        } else {
            None
        };
        let mut launched = Vec::with_capacity(self.ranks.len());
        for rank in 0..self.ranks.len() {
            let rank_device = &self.ranks[rank].device;
            let process = RankProcess::spawn_rank_deferred(
                self,
                &rank_device,
                rank as u32,
                self.ranks.len() as u32,
                rendezvous.clone(),
                &self.entries,
                cancel.clone(),
            )?;
            launched.push(process);
        }
        Ok(launched)
    }
}

/// Cooperative worker processes with one iceoryx2 service per rank.
pub struct WorkerGroup {
    workers: Vec<RankProcess>,
    buffers: Vec<VecDeque<WorkerResult>>,
    info: WorkerInfo,
    depth: usize,
    last_run_id: Option<u64>,
    progress_fds: Vec<i32>,
    last_progress: Instant,
    pending_batches: BTreeMap<u64, PendingBatch>,
    process_args: WorkerProcessArgs,
    resident_requests: HashSet<RequestKey>,
    command_wake_pending: bool,
    readiness_changed: bool,
    closed: bool,
}

type OperationIdentity = (u64, u64, u64, uniserve_worker_ipc::ComputationId);

/// A physical batch and its participating ranks share one retirement lifetime.
struct PendingBatch {
    run: ScheduleBatch,
    remaining: BTreeSet<OperationIdentity>,
    ranks: BTreeMap<usize, RankResult>,
}

/// A participating rank's received prefix, independent of transport fragmentation.
struct RankResult {
    operations: BTreeMap<OperationIdentity, bool>,
    complete: bool,
    error: Option<WorkerExecError>,
}

/// Returns the request and operation identifiers carried by a worker operation.
fn operation_identity(
    request: uniserve_worker_ipc::RequestKey,
    op: uniserve_worker_ipc::ComputationId,
) -> OperationIdentity {
    (
        request.engine_id,
        request.request_id.0,
        request.request_epoch,
        op,
    )
}

/// Combine rank-local resolved identities into one ordered process-world identity.
fn process_world_configuration_id(workers: &[RankProcess]) -> String {
    if workers.len() == 1 {
        return workers[0].info().configuration_id.clone();
    }

    let mut digest = Sha256::new();
    digest.update(b"uniserve-process-world-configuration-v1\0");
    for worker in workers {
        let info = worker.info();
        digest.update(info.endpoint.rank.to_le_bytes());
        digest.update((info.configuration_id.len() as u64).to_le_bytes());
        digest.update(info.configuration_id.as_bytes());
    }
    let mut identity = String::with_capacity(64);
    for byte in digest.finalize() {
        write!(&mut identity, "{byte:02x}").expect("writing to a String is infallible");
    }
    identity
}

impl WorkerGroup {
    /// Launch every configured rank and expose the instance after capability agreement.
    pub fn spawn(process_args: WorkerProcessArgs) -> anyhow::Result<Self> {
        Self::spawn_all(vec![process_args])?
            .pop()
            .context("WorkerGroup launch produced no instance")
    }

    /// Launch the configured static rank groups and wait for loaded capabilities.
    pub fn spawn_all(arguments: Vec<WorkerProcessArgs>) -> anyhow::Result<Vec<Self>> {
        let groups = arguments
            .iter()
            .map(|args| args.launch(None))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut workers = Vec::with_capacity(groups.len());
        for (mut ranks, args) in groups.into_iter().zip(arguments) {
            for rank in &mut ranks {
                rank.finish_startup()?;
            }
            workers.push(Self::from_ranks(args, ranks)?);
        }
        Ok(workers)
    }

    fn from_ranks(
        process_args: WorkerProcessArgs,
        mut workers: Vec<RankProcess>,
    ) -> anyhow::Result<Self> {
        let progress_fds = workers.iter().map(RankProcess::progress_fd).collect();
        anyhow::ensure!(!workers.is_empty(), "need >= 1 worker");
        let n = workers.len();
        let world_size =
            u32::try_from(n).context("process world size exceeds the IPC representation")?;
        let mut info = workers[0].info().clone();
        let mut canonical = info.clone();
        canonical.configuration_id.clear();
        for (rank, worker) in workers.iter().enumerate() {
            let rank_info = worker.info();
            rank_info
                .validate()
                .with_context(|| format!("physical rank {rank} reported invalid worker info"))?;
            anyhow::ensure!(
                rank_info.endpoint.rank == rank as u32 && rank_info.world_size == world_size,
                "physical rank {rank} reported topology ({}/{}) for launched topology ({rank}/{world_size})",
                rank_info.endpoint.rank,
                rank_info.world_size,
            );
            let mut normalized = rank_info.clone();
            anyhow::ensure!(
                rank_info.endpoint.worker_id == process_args.worker_id,
                "physical rank {rank} reported another WorkerGroup identity"
            );
            normalized.endpoint = canonical.endpoint.clone();
            normalized.device = canonical.device.clone();
            normalized.transfer_backends = canonical.transfer_backends.clone();
            if let (Some(local), Some(reference)) = (&mut normalized.kv_cache, &canonical.kv_cache)
            {
                local.kv_head_offset = reference.kv_head_offset;
                local.layer_offset = reference.layer_offset;
                local.num_layers = reference.num_layers;
                local.bytes_per_token = reference.bytes_per_token;
            }
            // The resolved identity includes rank-local parameter and buffer
            // layouts. Component params may therefore give each physical
            // rank a distinct identity while all scheduling/resource bounds
            // still have to agree.
            normalized.configuration_id.clear();
            anyhow::ensure!(
                normalized == canonical,
                "physical rank {rank} worker info disagree with rank 0"
            );
        }
        if let Some(cache) = &mut info.kv_cache {
            let regions: Vec<_> = workers
                .iter()
                .filter_map(|worker| worker.info().kv_cache.as_ref())
                .collect();
            let layer_bounds: BTreeSet<_> = regions
                .iter()
                .flat_map(|region| [region.layer_offset, region.layer_offset + region.num_layers])
                .chain([0, cache.total_layers])
                .collect();
            let layer_bounds: Vec<_> = layer_bounds.into_iter().collect();
            for layers in layer_bounds.windows(2) {
                let mut heads: Vec<_> = regions
                    .iter()
                    .filter(|region| {
                        region.layer_offset <= layers[0]
                            && region.layer_offset + region.num_layers >= layers[1]
                    })
                    .map(|region| {
                        (
                            region.kv_head_offset,
                            region.kv_head_offset + region.num_kv_heads,
                        )
                    })
                    .collect();
                heads.sort_unstable();
                let mut covered = 0;
                for (start, end) in heads {
                    anyhow::ensure!(
                        start <= covered,
                        "worker KV regions leave a logical head gap"
                    );
                    covered = covered.max(end);
                }
                anyhow::ensure!(
                    covered == cache.total_kv_heads,
                    "worker KV regions do not cover layers {}..{}",
                    layers[0],
                    layers[1],
                );
            }
            // The scheduler reserves pages shared by every stage. Its byte
            // accounting must cover the largest rank-local layer partition.
            cache.bytes_per_token = regions
                .iter()
                .map(|region| region.bytes_per_token)
                .max()
                .unwrap_or(cache.bytes_per_token);
        }
        info.configuration_id = process_world_configuration_id(&workers);
        for worker in &mut workers {
            worker.check_worker("WorkerGroup readiness")?;
            worker.set_startup_cancel(None);
        }
        let depth = info.queue_depth.max(1) as usize;
        let buffers = (0..n).map(|_| VecDeque::new()).collect();
        Ok(Self {
            workers,
            buffers,
            info,
            depth,
            last_run_id: None,
            progress_fds,
            last_progress: Instant::now(),
            pending_batches: BTreeMap::new(),
            process_args,
            resident_requests: HashSet::new(),
            command_wake_pending: false,
            readiness_changed: false,
            closed: false,
        })
    }

    /// Drains immediately available rank results into per-rank agreement buffers.
    fn pump_once(&mut self) -> anyhow::Result<()> {
        for rank in 0..self.workers.len() {
            loop {
                let result = self.workers[rank].poll_run(Duration::ZERO);
                let command_wake = self.workers[rank].take_command_wake();
                self.command_wake_pending |= command_wake;
                match result {
                    Ok(Some(result)) => {
                        let run_id = result.run_id;
                        let progress = self
                            .pending_batches
                            .get_mut(&run_id)
                            .and_then(|batch| batch.ranks.get_mut(&rank))
                            .ok_or_else(|| {
                                anyhow::anyhow!("rank {rank} returned unsubmitted step {run_id}")
                            })?;
                        anyhow::ensure!(
                            !progress.complete,
                            "rank {rank} returned after retirement"
                        );
                        let identities = report_operation_ids(&result);
                        anyhow::ensure!(
                            result.products.iter().all(|product| identities.contains(
                                &operation_identity(
                                    product.product.request_key,
                                    product.product.producer_op_id,
                                )
                            )),
                            "rank {rank} published a product without its operation completion"
                        );
                        for identity in identities {
                            let completed = progress.operations.get_mut(&identity).ok_or_else(|| {
                                anyhow::anyhow!("rank {rank} returned an unplanned operation for run {run_id}")
                            })?;
                            anyhow::ensure!(
                                !*completed,
                                "rank {rank} returned an operation more than once for run {run_id}"
                            );
                            *completed = true;
                        }
                        if result.done {
                            anyhow::ensure!(
                                progress.operations.values().all(|completed| *completed),
                                "rank {rank} retired run {run_id} without all completions"
                            );
                            progress.complete = true;
                        }
                        self.buffers[rank].push_back(result);
                    }
                    Ok(None) => break,
                    Err(error) => self.record_rank_error(rank, &error)?,
                }
                if command_wake {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Records the rank error.
    fn record_rank_error(&mut self, rank: usize, error: &anyhow::Error) -> anyhow::Result<()> {
        let execution = error
            .downcast_ref::<WorkerExecError>()
            .ok_or_else(|| anyhow::anyhow!("rank {rank} failed: {error:#}"))?;
        let run_id = execution.run_id.ok_or_else(|| {
            anyhow::anyhow!("rank {rank} returned an execution error without a step identity")
        })?;
        let progress = self
            .pending_batches
            .get_mut(&run_id)
            .and_then(|batch| batch.ranks.get_mut(&rank))
            .ok_or_else(|| {
                anyhow::anyhow!("rank {rank} returned an execution error for unknown run {run_id}")
            })?;
        if let Some(existing) = &progress.error {
            anyhow::ensure!(
                *existing == *execution,
                "rank {rank} returned conflicting errors for run {run_id}"
            );
        } else {
            progress.error = Some(execution.clone());
        }
        Ok(())
    }

    /// Replaces the complete rank group after worker loss and invalidates affected requests.
    fn recover_workers(&mut self, cause: &anyhow::Error) -> anyhow::Result<()> {
        let endpoints = self
            .workers
            .iter()
            .map(|worker| worker.info().endpoint.clone())
            .collect();
        let requests = self.resident_requests.iter().copied().collect();
        let expected = self
            .workers
            .iter()
            .map(|worker| worker.info().clone())
            .collect::<Vec<_>>();
        for worker in &mut self.workers {
            worker.terminate();
        }
        self.workers.clear();
        self.progress_fds.clear();
        self.clear_execution();
        let recovery = (|| -> anyhow::Result<()> {
            let mut workers = self.process_args.launch(None)?;
            for worker in &mut workers {
                worker.finish_startup()?;
            }
            anyhow::ensure!(
                workers.len() == expected.len(),
                "replacement rank count changed"
            );
            for (rank, (before, after)) in expected.iter().zip(&workers).enumerate() {
                validate_replacement_info(before, after.info(), rank)?;
            }
            anyhow::ensure!(
                process_world_configuration_id(&workers) == self.info.configuration_id,
                "replacement process-world configuration changed"
            );
            for worker in &mut workers {
                worker.check_worker("WorkerGroup readiness")?;
                worker.set_startup_cancel(None);
            }
            let descriptors = workers.iter().map(RankProcess::progress_fd).collect();
            self.install_replacement(workers, descriptors);
            Ok(())
        })();
        self.readiness_changed = true;
        let recovery_status = match recovery {
            Ok(()) => "rank group recovered".to_owned(),
            Err(error) => {
                self.closed = true;
                format!("rank recovery failed: {error:#}")
            }
        };
        Err(WorkerFailure {
            worker_id: crate::WorkerId(self.process_args.worker_id.clone()),
            endpoints,
            requests,
            retired: Vec::new(),
            buffers: Vec::new(),
            execution: cause.downcast_ref::<WorkerExecError>().cloned(),
            message: format!(
                "WorkerGroup {} lost its resident allocations; {recovery_status}: {cause:#}",
                self.process_args.worker_id
            ),
        }
        .into())
    }

    /// Installs a capability-compatible replacement rank group and resets rank-local state.
    fn install_replacement(&mut self, workers: Vec<RankProcess>, progress_fds: Vec<i32>) {
        self.info.endpoint = workers[0].info().endpoint.clone();
        self.workers = workers;
        self.progress_fds = progress_fds;
        self.last_progress = Instant::now();
        let ranks = self.process_args.ranks.len();
        self.buffers = (0..ranks).map(|_| VecDeque::new()).collect();
    }

    /// Clears execution bookkeeping after physical ownership has retired.
    fn clear_execution(&mut self) {
        self.pending_batches.clear();
        self.buffers.iter_mut().for_each(VecDeque::clear);
        self.resident_requests.clear();
    }

    /// Joins mutually agreeing rank reports into one logical physical result.
    fn try_join(&mut self) -> anyhow::Result<Option<WorkerResult>> {
        let Some((run_id, output_rank, operation_ids)) = self.joinable_report_key() else {
            // Preserve successful partial results already agreed by every rank
            // before retiring the unresolved remainder of a failed run.
            self.join_rank_errors()?;
            return Ok(None);
        };

        let pending = self
            .pending_batches
            .get_mut(&run_id)
            .ok_or_else(|| anyhow::anyhow!("joined run {run_id} has no pending batch"))?;
        let participants = pending
            .ranks
            .iter()
            .filter_map(|(&rank, run)| {
                (operation_ids.is_empty()
                    || operation_ids
                        .iter()
                        .all(|id| run.operations.contains_key(id)))
                .then_some(rank)
            })
            .collect::<Vec<_>>();
        let batch = &pending.run;
        let mut out =
            take_rank_operations(&mut self.buffers[output_rank], batch, &operation_ids, true)?;
        validate_and_order_rank_report(batch, &mut out, output_rank)?;
        for rank in participants.into_iter().filter(|rank| *rank != output_rank) {
            let mut report =
                take_rank_operations(&mut self.buffers[rank], batch, &operation_ids, false)?;
            validate_and_order_rank_report(batch, &mut report, rank)?;
            merge_rank_report(batch, &mut out, &report, rank)?;
        }
        let remaining = &mut pending.remaining;
        for identity in &operation_ids {
            anyhow::ensure!(
                remaining.remove(identity),
                "joined step {run_id} repeated an operation"
            );
        }
        // Physical retirement is independent of how a rank groups operation
        // fragments. Release commands still receive their separate terminal ack.
        let needs_retirement_ack = batch.commands.iter().any(|command| {
            matches!(
                command,
                BatchCommand::Free { .. } | BatchCommand::Finish { .. }
            )
        });
        out.done = remaining.is_empty()
            && pending.ranks.values().all(|run| run.complete)
            && (!needs_retirement_ack || operation_ids.is_empty());
        if out.done {
            // Report each rank's accumulated execution time once, after all of
            // its fragments arrive. Ranks execute concurrently, so the run's
            // duration is the maximum of these sums, independent of framing.
            out.worker_exec_us = self
                .buffers
                .iter()
                .filter_map(|buffer| {
                    buffer
                        .iter()
                        .filter(|report| report.run_id == run_id)
                        .filter_map(|report| report.worker_exec_us)
                        .reduce(u64::saturating_add)
                })
                .max();
            for command in &pending.run.commands {
                if let BatchCommand::Finish { request_key, .. } = command {
                    self.resident_requests.remove(request_key);
                }
            }
            self.finish_step(run_id);
        }
        Ok(Some(out))
    }

    /// Returns the file descriptors that signal worker progress.
    pub(crate) fn progress_fds(&self) -> &[i32] {
        &self.progress_fds
    }

    /// Resolves a run once every rank reports either success or a compatible error.
    fn join_rank_errors(&mut self) -> anyhow::Result<()> {
        let steps = self
            .pending_batches
            .iter()
            .filter_map(|(&run_id, pending)| {
                pending
                    .ranks
                    .values()
                    .any(|rank| rank.error.is_some())
                    .then_some(run_id)
            })
            .collect::<Vec<_>>();
        for run_id in steps {
            let pending = &self.pending_batches[&run_id];
            if pending
                .ranks
                .values()
                .any(|rank| rank.error.is_none() && !rank.complete)
            {
                continue;
            }
            let errors = pending
                .ranks
                .iter()
                .filter_map(|(&rank, result)| result.error.as_ref().map(|error| (rank, error)))
                .collect::<Vec<_>>();
            if pending.ranks.values().any(|run| {
                run.error.is_none()
                    && run
                        .operations
                        .keys()
                        .any(|id| pending.remaining.contains(id))
            }) {
                let details = errors
                    .iter()
                    .map(|(rank, error)| format!("rank {rank}: {error}"))
                    .collect::<Vec<_>>()
                    .join("; ");
                self.finish_step(run_id);
                anyhow::bail!(
                    "worker ranks disagreed between success and failure for step {run_id}: {details}"
                );
            }
            let canonical = errors[0].1.clone();
            for (rank, error) in errors.iter().copied().skip(1) {
                anyhow::ensure!(
                    *error == canonical,
                    "rank {rank} returned a different execution error for step {run_id}"
                );
            }
            let batch = &self.pending_batches[&run_id].run;
            if canonical.fatal
                || matches!(
                    canonical.code.as_deref(),
                    Some("SchedulerBug" | "InvariantViolation")
                )
                || batch
                    .commands
                    .iter()
                    .any(|command| matches!(command, BatchCommand::Free { .. }))
            {
                // A failed physical release cannot return an allocation safely.
                // Replacing this instance proves retirement without retrying
                // the same unsuccessful release indefinitely.
                return Err(canonical.into());
            }
            let pending = &self.pending_batches[&run_id].remaining;
            let retired = batch
                .operations
                .iter()
                .filter(|op| pending.contains(&operation_identity(op.request_key, op.op_id)))
                .map(|op| (batch.batch_id, op.request_key, op.op_id))
                .collect::<Vec<_>>();
            let requests = retired
                .iter()
                .map(|(_, request, _)| *request)
                .chain(batch.commands.iter().map(BatchCommand::request_key))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            let failure = WorkerFailure {
                worker_id: crate::WorkerId(self.process_args.worker_id.clone()),
                endpoints: Vec::new(),
                requests,
                retired,
                buffers: Vec::new(),
                message: canonical.to_string(),
                execution: Some(canonical),
            };
            self.finish_step(run_id);
            return Err(failure.into());
        }
        Ok(())
    }

    /// Retires all rank-local tracking for one agreed terminal run.
    fn finish_step(&mut self, run_id: u64) {
        self.pending_batches.remove(&run_id);
        for buffer in &mut self.buffers {
            buffer.retain(|report| report.run_id != run_id);
        }
    }

    /// Join one entry's completed work independently of rank and transport framing.
    fn joinable_report_key(&self) -> Option<(u64, usize, Vec<OperationIdentity>)> {
        for (&run_id, pending_batch) in &self.pending_batches {
            let pending = &pending_batch.remaining;
            if pending.is_empty() {
                if pending_batch.ranks.values().all(|run| run.complete) {
                    return Some((
                        run_id,
                        *pending_batch.ranks.first_key_value()?.0,
                        Vec::new(),
                    ));
                }
                continue;
            }
            let batch = &self.pending_batches[&run_id].run;
            for operation in &batch.operations {
                let identity = operation_identity(operation.request_key, operation.op_id);
                if !pending.contains(&identity) {
                    continue;
                }
                let owner = self.process_args.entries[&operation.entry].ranks[0];
                let members = self.operation_members(run_id, identity);
                for report in self.buffers[owner]
                    .iter()
                    .filter(|report| report.run_id == run_id)
                {
                    let shared = report_operation_ids(report)
                        .into_iter()
                        .filter(|id| {
                            pending.contains(id)
                                && self.operation_members(run_id, *id) == members
                                && batch.operations.iter().any(|candidate| {
                                    operation_identity(candidate.request_key, candidate.op_id)
                                        == *id
                                        && candidate.entry == operation.entry
                                })
                                && members.iter().all(|rank| {
                                    self.buffers[*rank].iter().any(|fragment| {
                                        fragment.run_id == run_id
                                            && fragment.results.iter().any(|completion| {
                                                operation_identity(
                                                    completion.output.request_key,
                                                    completion.output.op_id,
                                                ) == *id
                                            })
                                    })
                                })
                        })
                        .collect::<Vec<_>>();
                    if !shared.is_empty() {
                        return Some((run_id, owner, shared));
                    }
                }
            }
        }
        None
    }

    fn operation_members(&self, run_id: u64, identity: OperationIdentity) -> Vec<usize> {
        self.pending_batches[&run_id]
            .ranks
            .iter()
            .filter_map(|(&rank, run)| run.operations.contains_key(&identity).then_some(rank))
            .collect()
    }
}

/// Restrict a physical invocation to each entry's actual members. Request and
/// storage commands retain group-wide visibility, including on otherwise idle
/// ranks; they do not create synthetic computation completions.
fn split_rank_runs(
    batch: &ScheduleBatch,
    entries: &BTreeMap<String, crate::ComponentConfig>,
    rank_count: usize,
) -> anyhow::Result<Vec<(usize, ScheduleBatch)>> {
    let members = batch
        .operations
        .iter()
        .map(|operation| {
            let entry = entries
                .get(&operation.entry)
                .with_context(|| format!("unknown computation entry {}", operation.entry))?;
            let count = if entry.distribution.is_some() {
                let range = batch
                    .decode_ranges
                    .iter()
                    .find(|range| {
                        range.request_key == operation.request_key && range.op_id == operation.op_id
                    })
                    .context("temporally distributed entry requires a decode range")?;
                (range.max_units as usize)
                    .div_ceil(entry.units_per_rank)
                    .min(entry.ranks.len())
            } else {
                entry.ranks.len()
            };
            Ok(&entry.ranks[..count])
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut runs = Vec::new();
    for rank in 0..rank_count {
        let indices = members
            .iter()
            .enumerate()
            .filter_map(|(index, members)| members.contains(&rank).then_some(index))
            .collect::<Vec<_>>();
        if indices.is_empty() && batch.commands.is_empty() {
            continue;
        }
        let mut run = batch.clone();
        run.commands = batch.commands.clone();
        run.operations = indices
            .iter()
            .map(|index| batch.operations[*index].clone())
            .collect();
        run.forward = batch.forward.select(&indices);
        let slots = run
            .forward
            .request_pool_indices
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        run.block_tables
            .retain(|table| slots.contains(&table.request_pool_idx));
        run.new_cache_pages
            .retain(|pages| slots.contains(&pages.request_pool_idx));
        let identities = run
            .operations
            .iter()
            .map(|operation| (operation.request_key, operation.op_id))
            .collect::<HashSet<_>>();
        run.latent_params
            .retain(|params| identities.contains(&(params.request_key, params.op_id)));
        run.decode_ranges
            .retain(|range| identities.contains(&(range.request_key, range.op_id)));
        let inputs = run
            .operations
            .iter()
            .flat_map(|operation| {
                operation
                    .tensor_inputs()
                    .chain(operation.predicate.as_ref())
            })
            .collect::<HashSet<_>>();
        run.input_products
            .retain(|payload| inputs.contains(&payload.product));
        run.kv_inputs.retain(|publication| {
            run.operations
                .iter()
                .any(|operation| operation.kv_input == Some(publication.source))
        });
        let buffers = inputs
            .iter()
            .copied()
            .chain(
                run.operations
                    .iter()
                    .flat_map(|operation| operation.tensor_outputs()),
            )
            .map(|product| product.buffer_id())
            .collect::<HashSet<_>>();
        run.buffer_allocations
            .retain(|allocation| buffers.contains(&allocation.buffer));
        if run.operations.is_empty() && run.commands.is_empty() {
            continue;
        }
        run.validate()?;
        runs.push((rank, run));
    }
    Ok(runs)
}

/// Consume selected operations while retaining unmatched fragments and terminal
/// acknowledgements. An original fragment's aggregate statistics are emitted
/// once, when its final operation is consumed; they are never divided or copied.
fn take_rank_operations(
    buffer: &mut VecDeque<WorkerResult>,
    batch: &ScheduleBatch,
    identities: &[OperationIdentity],
    retain_forward_stats: bool,
) -> anyhow::Result<WorkerResult> {
    let mut output = WorkerResult {
        batch_id: batch.batch_id,
        run_id: batch.run_id,
        results: Vec::with_capacity(identities.len()),
        products: Vec::new(),
        registration: uniserve_worker_ipc::RegistrationAck { visible: true },
        worker_exec_us: None,
        forward_stats: None,
        done: identities.is_empty(),
    };
    for report in buffer
        .iter_mut()
        .filter(|report| report.run_id == batch.run_id)
    {
        let selected = report.results.iter().any(|completion| {
            identities.contains(&operation_identity(
                completion.output.request_key,
                completion.output.op_id,
            ))
        });
        if !selected && !(identities.is_empty() && report.done) {
            continue;
        }
        output.registration.visible &= report.registration.visible;
        output
            .results
            .extend(report.results.extract_if(.., |completion| {
                identities.contains(&operation_identity(
                    completion.output.request_key,
                    completion.output.op_id,
                ))
            }));
        output
            .products
            .extend(report.products.extract_if(.., |product| {
                identities.contains(&operation_identity(
                    product.product.request_key,
                    product.product.producer_op_id,
                ))
            }));
        if report.results.is_empty() {
            anyhow::ensure!(
                report.products.is_empty(),
                "rank fragment retained an unowned product"
            );
            if retain_forward_stats && report.forward_stats.is_some() {
                anyhow::ensure!(
                    output.forward_stats.is_none(),
                    "canonical operation selection spans statistics fragments"
                );
                output.forward_stats = report.forward_stats.take();
            }
        }
    }
    buffer.retain(|report| {
        report.run_id != batch.run_id
            || !report.results.is_empty()
            || (report.done && !identities.is_empty())
            || report.worker_exec_us.is_some()
    });
    anyhow::ensure!(
        output.results.len() == identities.len(),
        "rank fragment omitted selected operations"
    );
    Ok(output)
}

/// Returns the operation identifiers carried by a rank report.
fn report_operation_ids(report: &WorkerResult) -> Vec<OperationIdentity> {
    let mut ids = report
        .results
        .iter()
        .map(|output| operation_identity(output.output.request_key, output.output.op_id))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

/// Join cooperative completion reports into the designated output owner's report.
/// Every participant agrees on semantic completion; only the output owner
/// publishes host products.
fn merge_rank_report(
    batch: &ScheduleBatch,
    canonical_report: &mut WorkerResult,
    participant_report: &WorkerResult,
    rank: usize,
) -> anyhow::Result<()> {
    let run_id = batch.run_id;
    if participant_report.run_id != run_id {
        anyhow::bail!(
            "rank {rank} completion report run_id mismatch while joining step {run_id}: got {}",
            participant_report.run_id
        );
    }
    anyhow::ensure!(
        participant_report.done == canonical_report.done,
        "rank {rank} completion flag differs from the output owner for step {run_id}"
    );
    if participant_report.results.len() != canonical_report.results.len() {
        anyhow::bail!(
            "rank {rank} report for step {run_id} has {} completions, expected {}",
            participant_report.results.len(),
            canonical_report.results.len()
        );
    }
    for (completion_index, (canonical, actual)) in canonical_report
        .results
        .iter_mut()
        .zip(&participant_report.results)
        .enumerate()
    {
        anyhow::ensure!(
            batch
                .operations
                .iter()
                .any(
                    |operation| operation.request_key == canonical.output.request_key
                        && operation.op_id == canonical.output.op_id
                ),
            "rank join received an unplanned operation for step {run_id}"
        );
        if let Err(error) = merge_completion_record(canonical, actual) {
            anyhow::bail!(
                "rank {rank} report for step {run_id} completion {completion_index} differs from the output owner: {error:#}"
            );
        }
    }
    for product in &participant_report.products {
        anyhow::ensure!(
            batch
                .operations
                .iter()
                .flat_map(|operation| operation.tensor_outputs())
                .any(|output| output == &product.product),
            "rank {rank} published locations for an undeclared product"
        );
        if let Some(existing) = canonical_report
            .products
            .iter_mut()
            .find(|existing| existing.product == product.product)
        {
            existing.merge_locations(product)?;
        } else {
            canonical_report.products.push(product.clone());
        }
    }
    Ok(())
}

/// Validates a rank report and orders completions to match the submitted operation sequence.
fn validate_and_order_rank_report(
    batch: &ScheduleBatch,
    report: &mut WorkerResult,
    rank: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        report.run_id == batch.run_id,
        "rank {rank} returned step {} for pending step {}",
        report.run_id,
        batch.run_id
    );
    let report_count = report.results.len();
    let returned = report_operation_ids(report)
        .into_iter()
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        report_count > 0 || report.done,
        "rank {rank} returned an empty partial report for step {}",
        batch.run_id
    );
    let mut ordered = Vec::with_capacity(report_count);
    for planned in &batch.operations {
        let identity = operation_identity(planned.request_key, planned.op_id);
        if !returned.contains(&identity) {
            continue;
        }
        let index = report
            .results
            .iter()
            .position(|completion| {
                completion.output.request_key == planned.request_key
                    && completion.output.op_id == planned.op_id
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "rank {rank} omitted an operation for step {} collective {}",
                    batch.run_id,
                    batch.collective_seq
                )
            })?;
        let completion = report.results.swap_remove(index);
        anyhow::ensure!(
            completion.output.status != uniserve_worker_ipc::OpStatus::Predicated
                || report.products.iter().all(|publication| {
                    publication.product.request_key != completion.output.request_key
                        || publication.product.producer_op_id != completion.output.op_id
                }),
            "predicated operation published a tensor"
        );
        if planned.code
            == uniserve_worker_ipc::Computation::Transfer(
                uniserve_worker_ipc::TransferMode::KvPublish,
            )
            && completion.output.status == uniserve_worker_ipc::OpStatus::Ok
        {
            let publication = completion
                .output
                .kv_output
                .as_ref()
                .context("successful KV publication has no transfer descriptor")?;
            anyhow::ensure!(
                Some(publication.source) == planned.kv_output,
                "KV publication differs from its declared output"
            );
            let bytes = publication.tensors.iter().try_fold(0_u64, |sum, tensor| {
                Ok::<_, uniserve_worker_ipc::ValidationError>(
                    sum.saturating_add(tensor.validate()?),
                )
            })?;
            anyhow::ensure!(
                bytes <= planned.bounds.max_transfer_bytes,
                "KV publication exceeds its transfer-byte bound"
            );
        }
        ordered.push(completion);
    }
    anyhow::ensure!(
        ordered.len() == report_count,
        "rank {rank} returned an unplanned operation for step {}",
        batch.run_id
    );
    report.results = ordered;
    Ok(())
}

/// Merge one participant into the output owner's result. Accepted progress and
/// allocation generations agree; KV publications contribute immutable rank locations.
fn merge_completion_record(
    canonical: &mut OpResult,
    rank_completion: &OpResult,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        canonical.output.request_key == rank_completion.output.request_key
            && canonical.output.op_id == rank_completion.output.op_id,
        "completion identity diverged"
    );
    anyhow::ensure!(
        canonical.output.committed_tokens.as_slice()
            == rank_completion.output.committed_tokens.as_slice()
            && canonical.output.status == rank_completion.output.status
            && canonical.output.position == rank_completion.output.position
            && canonical.output.kv_visible_len == rank_completion.output.kv_visible_len
            && canonical.output.kv_computed_len == rank_completion.output.kv_computed_len
            && canonical.output.num_completed_steps == rank_completion.output.num_completed_steps
            && canonical.output.finish_flags == rank_completion.output.finish_flags,
        "completion result fields diverged"
    );
    anyhow::ensure!(
        rank_completion.media.as_ref().is_ok_and(Option::is_none)
            && rank_completion.output.sampled_logprob.is_none()
            && rank_completion.output.top_logprobs.is_empty()
            && rank_completion.output.prompt_logprobs.is_empty(),
        "non-designated rank returned public output"
    );
    anyhow::ensure!(
        canonical.output.product_generations == rank_completion.output.product_generations,
        "designated-rank product generations diverged"
    );
    match (
        &mut canonical.output.kv_output,
        &rank_completion.output.kv_output,
    ) {
        (Some(stored), Some(publication)) => stored.merge_locations(publication)?,
        (None, None) => {}
        _ => anyhow::bail!("rank KV publication presence diverged"),
    }
    Ok(())
}

/// Validates that a replacement rank preserves the established worker contract.
fn validate_replacement_info(
    expected: &WorkerInfo,
    actual: &WorkerInfo,
    rank: usize,
) -> anyhow::Result<()> {
    actual
        .validate()
        .with_context(|| format!("replacement rank {rank} reported invalid worker info"))?;
    let mut normalized_expected = expected.clone();
    anyhow::ensure!(
        actual.endpoint.worker_id == expected.endpoint.worker_id
            && actual.endpoint.rank == rank as u32
            && actual.endpoint.incarnation != expected.endpoint.incarnation,
        "replacement rank {rank} did not establish a new endpoint incarnation"
    );
    normalized_expected.endpoint = actual.endpoint.clone();
    normalized_expected.configuration_id.clear();
    let mut normalized_actual = actual.clone();
    normalized_actual.configuration_id.clear();
    anyhow::ensure!(
        normalized_expected == normalized_actual,
        "replacement rank {rank} worker info changed"
    );
    Ok(())
}

impl WorkerGroup {
    /// Reports whether every rank of this instance is ready to execute.
    pub fn is_ready(&self) -> bool {
        !self.closed && !self.workers.is_empty()
    }

    /// Physical submission slots not yet owned by accepted runs.
    pub(crate) fn available_slots(&self) -> usize {
        if self.is_ready() {
            self.depth.saturating_sub(self.pending_batches.len())
        } else {
            0
        }
    }

    pub(crate) fn take_readiness_change(&mut self) -> bool {
        std::mem::take(&mut self.readiness_changed)
    }

    /// Returns metadata for the physical worker.
    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Return the same bindings used to initialize this instance's transports.
    pub(crate) fn transfer_config(&self) -> &crate::executor::TransferConfig {
        &self.process_args.transfer
    }

    /// Exposes accepted operations whose required ranks have not returned their result.
    pub(crate) fn inflight_operations(
        &self,
    ) -> impl Iterator<Item = &uniserve_worker_ipc::ScheduledRequest> {
        self.pending_batches.values().flat_map(|pending| {
            pending.run.operations.iter().filter(|operation| {
                pending
                    .remaining
                    .contains(&operation_identity(operation.request_key, operation.op_id))
            })
        })
    }

    /// Returns the loaded physical endpoints used for transfer binding.
    pub(crate) fn rank_info(&self, rank: usize) -> Option<&WorkerInfo> {
        self.workers.get(rank).map(RankProcess::info)
    }

    /// Submit each operation to its entry members and lifetime commands to the rank group.
    pub fn submit_run(&mut self, batch: ScheduleBatch) -> Result<(), RunSubmitError> {
        if self.closed {
            return Err(RunSubmitError::Failed(anyhow::anyhow!(
                "WorkerGroup is closed"
            )));
        }
        if self.last_run_id.is_some_and(|last| batch.run_id <= last) {
            return Err(RunSubmitError::Failed(anyhow::anyhow!(
                "physical run IDs must increase in submission order"
            )));
        }
        if !self.is_ready() || self.pending_batches.len() >= self.depth {
            return Err(RunSubmitError::WouldBlock(batch));
        }
        batch
            .validate()
            .map_err(anyhow::Error::from)
            .map_err(RunSubmitError::Failed)?;
        let rank_batches = split_rank_runs(&batch, &self.process_args.entries, self.workers.len())
            .and_then(|runs| {
                runs.into_iter()
                    .map(|(rank, mut run)| {
                        self.process_args.transfer.bind_inputs(
                            &mut run.input_products,
                            &mut run.kv_inputs,
                            &self.workers[rank].info().endpoint,
                        )?;
                        Ok((rank, run))
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })
            .map_err(RunSubmitError::Failed)?;
        let run_id = batch.run_id;
        let requests = batch
            .operations()
            .map(|operation| operation.request_key)
            .collect::<Vec<_>>();
        self.last_run_id = Some(run_id);
        self.resident_requests.extend(requests);
        self.resident_requests
            .extend(batch.commands.iter().filter_map(|command| match command {
                BatchCommand::Start { request } => Some(request.request_key),
                _ => None,
            }));
        let ranks = rank_batches
            .iter()
            .map(|(rank, run)| {
                (
                    *rank,
                    RankResult {
                        operations: run
                            .operations
                            .iter()
                            .map(|operation| {
                                (
                                    operation_identity(operation.request_key, operation.op_id),
                                    false,
                                )
                            })
                            .collect(),
                        complete: false,
                        error: None,
                    },
                )
            })
            .collect();
        self.pending_batches.insert(
            run_id,
            PendingBatch {
                remaining: batch
                    .operations
                    .iter()
                    .map(|operation| operation_identity(operation.request_key, operation.op_id))
                    .collect(),
                run: batch,
                ranks,
            },
        );
        self.last_progress = Instant::now();
        for (rank, run) in rank_batches {
            if let Err(error) = self.workers[rank].submit_run(run) {
                let error = match error {
                    RunSubmitError::WouldBlock(_) => {
                        anyhow::anyhow!(
                            "physical rank {rank} rejected an admitted cooperative batch as full"
                        )
                    }
                    RunSubmitError::Failed(error) => {
                        error.context(format!("submit to rank {rank} failed"))
                    }
                };
                let recovered = self.recover_workers(&error).unwrap_err();
                return Err(RunSubmitError::Failed(recovered));
            }
        }
        Ok(())
    }

    /// Waits for rank progress and returns the next fully agreed physical result.
    pub fn poll_run(&mut self, timeout: Duration) -> anyhow::Result<Option<WorkerResult>> {
        // A repeated poll acknowledges the preceding wake even for direct
        // callers that do not use the logical Executor's command ingress.
        self.command_wake_pending = false;
        if self.closed {
            return Ok(None);
        }
        match self.poll_progress(timeout) {
            Ok(report) => Ok(report),
            Err(error) if error.is::<WorkerFailure>() => Err(error),
            Err(error) => {
                self.recover_workers(&error)?;
                unreachable!("WorkerGroup replacement returns its invalidated ownership")
            }
        }
    }

    fn poll_progress(&mut self, timeout: Duration) -> anyhow::Result<Option<WorkerResult>> {
        if self.command_wake_pending {
            return Ok(None);
        }
        // Idle ranks still report command ingress and process death. Forward
        // those wakes before the logical Executor parks on the shared descriptors.
        if self.pending_batches.len() == 0 {
            self.pump_once()?;
            if self.command_wake_pending {
                return Ok(None);
            }
            if timeout.is_zero() {
                return Ok(None);
            }
            crate::worker::park_descriptors(&self.progress_fds, timeout)?;
            self.pump_once()?;
            if self.command_wake_pending {
                return Ok(None);
            }
            return self.try_join();
        }
        self.pump_once()?;
        if self.command_wake_pending {
            return Ok(None);
        }
        if let Some(result) = self.try_join()? {
            self.last_progress = Instant::now();
            return Ok(Some(result));
        }
        if timeout.is_zero() {
            return Ok(None);
        }
        let deadline = Instant::now() + timeout;
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            crate::worker::park_descriptors(&self.progress_fds, deadline - now)?;
            self.pump_once()?;
            if self.command_wake_pending {
                return Ok(None);
            }
            if let Some(result) = self.try_join()? {
                self.last_progress = Instant::now();
                return Ok(Some(result));
            }
            if self.pending_batches.len() > 0
                && self.last_progress.elapsed() >= NEXT_RESULT_DEADLINE
            {
                let error = anyhow::anyhow!(
                    "cooperative workers produced no progress within {:?}",
                    NEXT_RESULT_DEADLINE
                );
                return Err(error);
            }
        }
    }

    /// Consumes the pending command-wake notification.
    pub(crate) fn take_command_wake(&mut self) -> bool {
        std::mem::take(&mut self.command_wake_pending)
    }

    fn release_closed_resources(&mut self) {
        // Closed endpoints never trigger serving recovery.
        self.closed = true;
        self.workers.clear();
        self.progress_fds.clear();
        self.clear_execution();
        self.readiness_changed = true;
        self.command_wake_pending = true;
    }

    /// Closes every rank in the physical worker group.
    pub fn close(&mut self) -> anyhow::Result<()> {
        self.closed = true;
        let mut first_error = None;
        for worker in self.workers.iter_mut() {
            if let Err(error) = worker.close()
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        self.release_closed_resources();
        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(())
        }
    }
}
