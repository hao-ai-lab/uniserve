//! Physical worker fan-out, completion agreement, and process recovery.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use super::{RankProcess, RunSubmitError};
use crate::executor::{WorkerExecError, WorkerFailure};
use anyhow::Context;
use sha2::{Digest as _, Sha256};
use uniserve_worker_ipc::{
    BatchCommand, ModelOutput, RequestKey, Run as PhysicalRun, RunResult, WorkerInfo,
};

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
pub struct Worker {
    workers: Vec<RankProcess>,
    buffers: Vec<VecDeque<RunResult>>,
    info: WorkerInfo,
    depth: usize,
    inflight: usize,
    last_run_id: Option<u64>,
    progress_fds: Vec<i32>,
    last_progress: Instant,
    pending_batches: BTreeMap<u64, PhysicalRun>,
    pending_operations: BTreeMap<u64, BTreeSet<OperationIdentity>>,
    rank_errors: Vec<BTreeMap<u64, WorkerExecError>>,
    rank_runs: Vec<BTreeMap<u64, RankRun>>,
    operation_ranks: BTreeMap<OperationIdentity, Vec<usize>>,
    process_args: WorkerProcessArgs,
    resident_requests: HashSet<RequestKey>,
    command_wake_pending: bool,
    readiness_changed: bool,
    closed: bool,
}

type OperationIdentity = (u64, u64, u64, u64);

/// Exactly the work submitted to a rank, independent of result fragmentation.
struct RankRun {
    operations: BTreeSet<OperationIdentity>,
    returned: BTreeSet<OperationIdentity>,
    complete: bool,
}

/// Returns the request and operation identifiers carried by a worker operation.
fn operation_identity(
    request: uniserve_worker_ipc::RequestKey,
    op: uniserve_worker_ipc::OpId,
) -> OperationIdentity {
    (
        request.authority_id,
        request.request_id.0,
        request.epoch,
        op.0,
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

impl Worker {
    /// Launch every configured rank and expose the instance after capability agreement.
    pub fn spawn(process_args: WorkerProcessArgs) -> anyhow::Result<Self> {
        Self::spawn_all(vec![process_args])?
            .pop()
            .context("Worker launch produced no instance")
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
                "physical rank {rank} reported another Worker identity"
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
            worker.check_worker("Worker readiness")?;
            worker.set_startup_cancel(None);
        }
        let depth = info.queue_depth.max(1) as usize;
        let buffers = (0..n).map(|_| VecDeque::new()).collect();
        Ok(Self {
            workers,
            buffers,
            info,
            depth,
            inflight: 0,
            last_run_id: None,
            progress_fds,
            last_progress: Instant::now(),
            pending_batches: BTreeMap::new(),
            pending_operations: BTreeMap::new(),
            rank_errors: (0..n).map(|_| BTreeMap::new()).collect(),
            rank_runs: (0..n).map(|_| BTreeMap::new()).collect(),
            operation_ranks: BTreeMap::new(),
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
                        let progress = self.rank_runs[rank].get_mut(&run_id).ok_or_else(|| {
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
                            anyhow::ensure!(
                                progress.returned.insert(identity),
                                "rank {rank} returned an operation more than once for step {run_id}"
                            );
                        }
                        anyhow::ensure!(
                            progress.returned.is_subset(&progress.operations),
                            "rank {rank} returned an unplanned lane for step {run_id}"
                        );
                        if result.done {
                            anyhow::ensure!(
                                progress.returned == progress.operations,
                                "rank {rank} retired step {run_id} without all completions"
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
        anyhow::ensure!(
            self.rank_runs[rank].contains_key(&run_id),
            "rank {rank} returned an execution error for unknown step {run_id}"
        );
        if let Some(existing) = self.rank_errors[rank].insert(run_id, execution.clone()) {
            anyhow::ensure!(
                existing == *execution,
                "rank {rank} returned conflicting errors for step {run_id}"
            );
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
                worker.check_worker("Worker readiness")?;
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
            products: Vec::new(),
            execution: cause.downcast_ref::<WorkerExecError>().cloned(),
            message: format!(
                "Worker {} lost its resident allocations; {recovery_status}: {cause:#}",
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
        self.rank_errors = (0..ranks).map(|_| BTreeMap::new()).collect();
        self.rank_runs = (0..ranks).map(|_| BTreeMap::new()).collect();
    }

    /// Clears execution bookkeeping after physical ownership has retired.
    fn clear_execution(&mut self) {
        self.pending_batches.clear();
        self.pending_operations.clear();
        self.buffers.iter_mut().for_each(VecDeque::clear);
        self.rank_errors.iter_mut().for_each(BTreeMap::clear);
        self.rank_runs.iter_mut().for_each(BTreeMap::clear);
        self.operation_ranks.clear();
        self.inflight = 0;
        self.resident_requests.clear();
    }

    /// Joins mutually agreeing rank reports into one logical physical result.
    fn try_join(&mut self) -> anyhow::Result<Option<RunResult>> {
        let Some((run_id, output_rank, operation_ids)) = self.joinable_report_key() else {
            // Preserve successful partial results already agreed by every rank
            // before retiring the unresolved remainder of a failed run.
            self.join_rank_errors()?;
            return Ok(None);
        };

        let participants = self
            .rank_runs
            .iter()
            .enumerate()
            .filter_map(|(rank, runs)| {
                let run = runs.get(&run_id)?;
                (operation_ids.is_empty()
                    || operation_ids.iter().all(|id| run.operations.contains(id)))
                .then_some(rank)
            })
            .collect::<Vec<_>>();
        let batch = self
            .pending_batches
            .get(&run_id)
            .ok_or_else(|| anyhow::anyhow!("joined step {run_id} has no pending batch"))?;
        let mut out =
            take_rank_operations(&mut self.buffers[output_rank], batch, &operation_ids, true)?;
        validate_and_order_rank_report(batch, &mut out, output_rank)?;
        for rank in participants.into_iter().filter(|rank| *rank != output_rank) {
            let mut report =
                take_rank_operations(&mut self.buffers[rank], batch, &operation_ids, false)?;
            validate_and_order_rank_report(batch, &mut report, rank)?;
            merge_rank_report(batch, &mut out, &report, rank)?;
        }
        let remaining = self
            .pending_operations
            .get_mut(&run_id)
            .ok_or_else(|| anyhow::anyhow!("joined step {run_id} has no operation ledger"))?;
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
                BatchCommand::Free { .. }
                    | BatchCommand::Finish { .. }
                    | BatchCommand::Retire { .. }
            )
        });
        out.done = remaining.is_empty()
            && self
                .rank_runs
                .iter()
                .filter_map(|runs| runs.get(&run_id))
                .all(|run| run.complete)
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
            for command in &self.pending_batches[&run_id].commands {
                if let BatchCommand::Finish { request_key, .. }
                | BatchCommand::Retire { request_key, .. } = command
                {
                    self.resident_requests.remove(request_key);
                    self.operation_ranks.retain(|identity, _| {
                        (identity.0, identity.1, identity.2)
                            != (
                                request_key.authority_id,
                                request_key.request_id.0,
                                request_key.epoch,
                            )
                    });
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
            .rank_errors
            .iter()
            .flat_map(|errors| errors.keys().copied())
            .collect::<BTreeSet<_>>();
        for run_id in steps {
            let terminal = self
                .rank_runs
                .iter()
                .enumerate()
                .filter_map(|(rank, runs)| {
                    runs.get(&run_id)
                        .map(|run| self.rank_errors[rank].contains_key(&run_id) || run.complete)
                })
                .collect::<Vec<_>>();
            if terminal.iter().any(|value| !value) {
                continue;
            }
            let errors = self
                .rank_errors
                .iter()
                .enumerate()
                .filter_map(|(rank, entries)| entries.get(&run_id).map(|error| (rank, error)))
                .collect::<Vec<_>>();
            if self.rank_runs.iter().enumerate().any(|(rank, runs)| {
                runs.get(&run_id).is_some_and(|run| {
                    run.operations
                        .iter()
                        .any(|id| self.pending_operations[&run_id].contains(id))
                        && !self.rank_errors[rank].contains_key(&run_id)
                })
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
            let batch = &self.pending_batches[&run_id];
            if canonical.fatal
                || matches!(
                    canonical.code.as_deref(),
                    Some("SchedulerBug" | "InvariantViolation")
                )
                || batch.commands.iter().any(|command| {
                    matches!(
                        command,
                        BatchCommand::Free { .. } | BatchCommand::Retire { .. }
                    )
                })
            {
                // A failed physical release cannot return an allocation safely.
                // Replacing this instance proves retirement without retrying
                // the same unsuccessful release indefinitely.
                return Err(canonical.into());
            }
            let pending = &self.pending_operations[&run_id];
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
                products: Vec::new(),
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
        self.pending_operations.remove(&run_id);
        for buffer in &mut self.buffers {
            buffer.retain(|report| report.run_id != run_id);
        }
        for errors in &mut self.rank_errors {
            errors.remove(&run_id);
        }
        for runs in &mut self.rank_runs {
            runs.remove(&run_id);
        }
        self.inflight = self.inflight.saturating_sub(1);
    }

    /// Join one entry's completed work independently of rank and transport framing.
    fn joinable_report_key(&self) -> Option<(u64, usize, Vec<OperationIdentity>)> {
        for (&run_id, pending) in &self.pending_operations {
            if pending.is_empty() {
                let ranks = self
                    .rank_runs
                    .iter()
                    .enumerate()
                    .filter_map(|(rank, runs)| runs.get(&run_id).map(|run| (rank, run)))
                    .collect::<Vec<_>>();
                if ranks.iter().all(|(_, run)| run.complete) {
                    return Some((run_id, ranks[0].0, Vec::new()));
                }
                continue;
            }
            let batch = &self.pending_batches[&run_id];
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
                                            && fragment.completions.iter().any(|completion| {
                                                operation_identity(
                                                    completion.request_key,
                                                    completion.op_id,
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
        self.rank_runs
            .iter()
            .enumerate()
            .filter_map(|(rank, runs)| {
                runs.get(&run_id)
                    .is_some_and(|run| run.operations.contains(&identity))
                    .then_some(rank)
            })
            .collect()
    }
}

/// Restrict a physical invocation to each entry's actual members. Request and
/// storage commands retain group-wide visibility, including on otherwise idle
/// ranks; they do not create synthetic computation completions.
fn split_rank_runs(
    batch: &PhysicalRun,
    entries: &BTreeMap<String, crate::EntryConfig>,
    rank_count: usize,
    operation_ranks: &BTreeMap<OperationIdentity, Vec<usize>>,
) -> anyhow::Result<Vec<(usize, PhysicalRun)>> {
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
        run.commands = batch
            .commands
            .iter()
            .filter_map(|command| {
                let checkpoint = match command {
                    BatchCommand::Commit { selected, .. } => Some(selected),
                    BatchCommand::Finish { cutoff, .. } => Some(cutoff),
                    _ => None,
                };
                let owner = checkpoint.and_then(|checkpoint| {
                    operation_ranks
                        .get(&operation_identity(command.request_key(), checkpoint.op_id))
                });
                if owner.is_none_or(|ranks| ranks.contains(&rank)) {
                    return Some(command.clone());
                }
                match command {
                    BatchCommand::Finish {
                        request_key,
                        retained_buffers,
                        ..
                    } => Some(BatchCommand::Retire {
                        request_key: *request_key,
                        retained_buffers: retained_buffers.clone(),
                    }),
                    BatchCommand::Commit { .. } => None,
                    _ => unreachable!("only semantic commands have checkpoint owners"),
                }
            })
            .collect();
        run.operations = indices
            .iter()
            .map(|index| batch.operations[*index].clone())
            .collect();
        run.forward_rows = batch
            .forward_rows
            .iter()
            .filter_map(|row| {
                let index = indices
                    .iter()
                    .position(|index| *index == row.operation_index as usize)?;
                let mut row = row.clone();
                row.operation_index = index as u32;
                Some(row)
            })
            .collect();
        let slots = run
            .forward_rows
            .iter()
            .map(|row| row.request_pool_index)
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
            .flat_map(|operation| operation.inputs().iter().chain(operation.predicate()))
            .collect::<HashSet<_>>();
        run.input_products
            .retain(|payload| inputs.contains(&payload.product));
        let buffers = inputs
            .iter()
            .copied()
            .chain(
                run.operations
                    .iter()
                    .flat_map(|operation| operation.outputs()),
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
    buffer: &mut VecDeque<RunResult>,
    batch: &PhysicalRun,
    identities: &[OperationIdentity],
    retain_forward_stats: bool,
) -> anyhow::Result<RunResult> {
    let mut output = RunResult {
        batch_id: batch.batch_id,
        run_id: batch.run_id,
        completions: Vec::with_capacity(identities.len()),
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
        let selected = report.completions.iter().any(|completion| {
            identities.contains(&operation_identity(
                completion.request_key,
                completion.op_id,
            ))
        });
        if !selected && !(identities.is_empty() && report.done) {
            continue;
        }
        output.registration.visible &= report.registration.visible;
        output
            .completions
            .extend(report.completions.extract_if(.., |completion| {
                identities.contains(&operation_identity(
                    completion.request_key,
                    completion.op_id,
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
        if report.completions.is_empty() {
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
            || !report.completions.is_empty()
            || (report.done && !identities.is_empty())
            || report.worker_exec_us.is_some()
    });
    anyhow::ensure!(
        output.completions.len() == identities.len(),
        "rank fragment omitted selected operations"
    );
    Ok(output)
}

/// Returns the operation identifiers carried by a rank report.
fn report_operation_ids(report: &RunResult) -> Vec<OperationIdentity> {
    let mut ids = report
        .completions
        .iter()
        .map(|output| operation_identity(output.request_key, output.op_id))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

/// Join cooperative completion reports into the designated output owner's report.
/// Every participant agrees on semantic completion; only the output owner
/// publishes host products.
fn merge_rank_report(
    batch: &PhysicalRun,
    canonical_report: &mut RunResult,
    participant_report: &RunResult,
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
    if participant_report.completions.len() != canonical_report.completions.len() {
        anyhow::bail!(
            "rank {rank} report for step {run_id} has {} completions, expected {}",
            participant_report.completions.len(),
            canonical_report.completions.len()
        );
    }
    for (completion_index, (canonical, actual)) in canonical_report
        .completions
        .iter_mut()
        .zip(&participant_report.completions)
        .enumerate()
    {
        anyhow::ensure!(
            batch
                .operations
                .iter()
                .any(|operation| operation.request_key == canonical.request_key
                    && operation.op_id == canonical.op_id),
            "rank join received an unplanned operation for step {run_id}"
        );
        if let Err(error) = merge_completion_record(canonical, actual) {
            anyhow::bail!(
                "rank {rank} report for step {run_id} completion {completion_index} differs from the output owner: {error:#}"
            );
        }
    }
    for product in &participant_report.products {
        let uniserve_worker_ipc::InlineValue::Transfer(locations) = &product.value else {
            anyhow::bail!("rank {rank} published host products despite designated-rank ownership");
        };
        anyhow::ensure!(
            batch
                .operations
                .iter()
                .flat_map(|operation| operation.outputs())
                .any(|output| output == &product.product),
            "rank {rank} published locations for an undeclared product"
        );
        if let Some(existing) = canonical_report
            .products
            .iter_mut()
            .find(|existing| existing.product == product.product)
        {
            let uniserve_worker_ipc::InlineValue::Transfer(stored) = &mut existing.value else {
                anyhow::bail!("rank {rank} changed an inline product into a transfer");
            };
            stored.merge_locations(locations)?;
        } else {
            canonical_report.products.push(product.clone());
        }
    }
    Ok(())
}

/// Validates a rank report and orders completions to match the submitted operation sequence.
fn validate_and_order_rank_report(
    batch: &PhysicalRun,
    report: &mut RunResult,
    rank: usize,
) -> anyhow::Result<()> {
    report.validate()?;
    anyhow::ensure!(
        report.run_id == batch.run_id,
        "rank {rank} returned step {} for pending step {}",
        report.run_id,
        batch.run_id
    );
    let report_count = report.completions.len();
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
            .completions
            .iter()
            .position(|completion| {
                completion.request_key == planned.request_key && completion.op_id == planned.op_id
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "rank {rank} omitted an operation for step {} collective {}",
                    batch.run_id,
                    batch.collective_seq
                )
            })?;
        ordered.push(report.completions.swap_remove(index));
    }
    anyhow::ensure!(
        ordered.len() == report_count,
        "rank {rank} returned an unplanned operation for step {}",
        batch.run_id
    );
    report.completions = ordered;
    Ok(())
}

/// Merges one rank's completion record into rank 0's. Every rank must agree on
/// the semantic result; only `product_generations` are per-rank shards, which
/// concatenate in rank order.
fn merge_completion_record(
    canonical: &mut ModelOutput,
    rank_completion: &ModelOutput,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        canonical.request_key == rank_completion.request_key
            && canonical.op_id == rank_completion.op_id
            && canonical.selected_point == rank_completion.selected_point,
        "completion identity or selected point diverged"
    );
    anyhow::ensure!(
        canonical.committed_tokens() == rank_completion.committed_tokens()
            && canonical.status == rank_completion.status
            && canonical.logical_lengths() == rank_completion.logical_lengths()
            && canonical.token_span() == rank_completion.token_span()
            && canonical.finish_flags() == rank_completion.finish_flags(),
        "completion result fields diverged"
    );
    anyhow::ensure!(
        rank_completion.media_output().is_none(),
        "non-designated rank returned a media artifact"
    );
    anyhow::ensure!(
        canonical.product_generations == rank_completion.product_generations,
        "designated-rank product generations diverged"
    );
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

impl Worker {
    /// Reports whether every rank of this instance is ready to execute.
    pub fn is_ready(&self) -> bool {
        !self.closed && !self.workers.is_empty()
    }

    /// Physical submission slots not yet owned by accepted runs.
    pub(crate) fn available_slots(&self) -> usize {
        if self.is_ready() {
            self.depth.saturating_sub(self.inflight)
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
    pub(crate) fn transfer_config(&self) -> &crate::executor::TransportMap {
        &self.process_args.transfer
    }

    /// Exposes accepted operations whose required ranks have not returned their result.
    pub(crate) fn inflight_operations(
        &self,
    ) -> impl Iterator<Item = &uniserve_worker_ipc::Operation> {
        self.pending_batches
            .iter()
            .flat_map(move |(run_id, batch)| {
                batch.operations.iter().filter(move |operation| {
                    self.pending_operations.get(run_id).is_some_and(|pending| {
                        pending
                            .contains(&operation_identity(operation.request_key, operation.op_id))
                    })
                })
            })
    }

    /// Returns the loaded physical endpoints used for transfer binding.
    pub(crate) fn rank_info(&self, rank: usize) -> Option<&WorkerInfo> {
        self.workers.get(rank).map(RankProcess::info)
    }

    /// Submit each operation to its entry members and lifetime commands to the rank group.
    pub fn submit_run(&mut self, batch: PhysicalRun) -> Result<(), RunSubmitError> {
        if self.closed {
            return Err(RunSubmitError::Failed(anyhow::anyhow!("Worker is closed")));
        }
        if self.last_run_id.is_some_and(|last| batch.run_id <= last) {
            return Err(RunSubmitError::Failed(anyhow::anyhow!(
                "physical run IDs must increase in submission order"
            )));
        }
        if !self.is_ready() || self.inflight >= self.depth {
            return Err(RunSubmitError::WouldBlock(batch));
        }
        batch
            .validate()
            .map_err(anyhow::Error::from)
            .map_err(RunSubmitError::Failed)?;
        let rank_batches = split_rank_runs(
            &batch,
            &self.process_args.entries,
            self.workers.len(),
            &self.operation_ranks,
        )
        .and_then(|runs| {
            runs.into_iter()
                .map(|(rank, mut run)| {
                    self.process_args.transfer.bind_inputs(
                        &mut run.input_products,
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
        self.pending_batches.insert(run_id, batch.clone());
        self.pending_operations.insert(
            run_id,
            batch
                .operations
                .iter()
                .map(|operation| operation_identity(operation.request_key, operation.op_id))
                .collect(),
        );
        self.resident_requests.extend(requests);
        self.resident_requests
            .extend(batch.commands.iter().filter_map(|command| match command {
                BatchCommand::Start { request } => Some(request.request_key),
                _ => None,
            }));
        for (rank, run) in &rank_batches {
            for operation in &run.operations {
                self.operation_ranks
                    .entry(operation_identity(operation.request_key, operation.op_id))
                    .or_default()
                    .push(*rank);
            }
            self.rank_runs[*rank].insert(
                run_id,
                RankRun {
                    operations: run
                        .operations
                        .iter()
                        .map(|operation| operation_identity(operation.request_key, operation.op_id))
                        .collect(),
                    returned: BTreeSet::new(),
                    complete: false,
                },
            );
        }
        self.inflight += 1;
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
    pub fn poll_run(&mut self, timeout: Duration) -> anyhow::Result<Option<RunResult>> {
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
                unreachable!("Worker replacement returns its invalidated ownership")
            }
        }
    }

    fn poll_progress(&mut self, timeout: Duration) -> anyhow::Result<Option<RunResult>> {
        if self.command_wake_pending {
            return Ok(None);
        }
        // Idle ranks still report command ingress and process death. Forward
        // those wakes before the logical Executor parks on the shared descriptors.
        if self.inflight == 0 {
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
            if self.inflight > 0 && self.last_progress.elapsed() >= NEXT_RESULT_DEADLINE {
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
