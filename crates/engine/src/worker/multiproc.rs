//! Physical worker fan-out, completion agreement, and process recovery.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;
use std::net::TcpListener;
use std::time::{Duration, Instant};

use crate::executor::{
    Batch, BatchResult, Executor, ExecutorInfo, ExecutorSubmitError, LogicalResultTracker,
    PhysicalExecutor, PhysicalSubmitError, PoolId, WorkerExecError, WorkerLossError, lower_batch,
};
use anyhow::Context;
use sha2::{Digest as _, Sha256};
use uniserve_core::{CommandWaker, RequestId};
use uniserve_worker_ipc::{ModelOutput, Run as PhysicalRun, RunResult, WorkerInfo};

use crate::worker::WorkerProcessArgs;

/// Maximum interval without rank progress before an in-flight run is treated as failed.
const NEXT_RESULT_DEADLINE: Duration = Duration::from_secs(300);

impl WorkerProcessArgs {
    /// Launches and connects every rank in one physical worker group.
    fn launch(
        &self,
    ) -> anyhow::Result<(Vec<Box<dyn PhysicalExecutor>>, Vec<CommandWaker>, Vec<i32>)> {
        let distributed_init_method = if self.world_size > 1 {
            Some(allocate_distributed_init_method()?)
        } else {
            None
        };
        let components = self.resolved_components()?;
        let mut launched = Vec::with_capacity(self.world_size);
        for rank in 0..self.world_size {
            let rank_device = if let Some(deployment) = &self.deployment {
                format!("cuda:{}", deployment.devices[rank])
            } else {
                device_for_rank(&self.device, rank, self.world_size)
            };
            launched.push(crate::UniprocExecutor::spawn_rank_deferred(
                self,
                &rank_device,
                rank as u32,
                self.world_size as u32,
                distributed_init_method.as_deref(),
                &components,
            )?);
        }
        let mut workers: Vec<Box<dyn PhysicalExecutor>> = Vec::with_capacity(self.world_size);
        let mut wakers = Vec::with_capacity(self.world_size);
        let mut progress_fds = Vec::with_capacity(self.world_size);
        for mut worker in launched {
            worker.finish_startup()?;
            wakers.push(worker.command_waker());
            progress_fds.push(worker.progress_fd());
            workers.push(Box::new(worker));
        }
        Ok((workers, wakers, progress_fds))
    }
}

/// Cooperative worker processes with one iceoryx2 service per rank.
pub struct MultiprocExecutor {
    workers: Vec<Box<dyn PhysicalExecutor>>,
    buffers: Vec<VecDeque<RunResult>>,
    info: WorkerInfo,
    executor_info: ExecutorInfo,
    depth: usize,
    inflight: usize,
    command_wakers: Vec<CommandWaker>,
    progress_fds: Vec<i32>,
    last_progress: Instant,
    pending_batches: BTreeMap<u64, PhysicalRun>,
    pending_operations: BTreeMap<u64, BTreeSet<OperationIdentity>>,
    rank_errors: Vec<BTreeMap<u64, WorkerExecError>>,
    rank_returned_operations: Vec<BTreeMap<u64, BTreeSet<OperationIdentity>>>,
    rank_successes: Vec<BTreeSet<u64>>,
    process_args: Option<WorkerProcessArgs>,
    known_requests: BTreeSet<RequestId>,
    dirty_requests: BTreeSet<RequestId>,
    next_collective_seq: u64,
    logical_results: LogicalResultTracker,
    command_wake_pending: bool,
}

type OperationIdentity = (u64, u64, u64, u64);

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
fn process_world_configuration_id(workers: &[Box<dyn PhysicalExecutor>]) -> String {
    if workers.len() == 1 {
        return workers[0]
            .physical_info()
            .single_pool()
            .configuration_id
            .clone();
    }

    let mut digest = Sha256::new();
    digest.update(b"uniserve-process-world-configuration-v1\0");
    for worker in workers {
        let info = worker.physical_info().single_pool();
        digest.update(info.rank.rank.to_le_bytes());
        digest.update((info.configuration_id.len() as u64).to_le_bytes());
        digest.update(info.configuration_id.as_bytes());
    }
    let mut identity = String::with_capacity(64);
    for byte in digest.finalize() {
        write!(&mut identity, "{byte:02x}").expect("writing to a String is infallible");
    }
    identity
}

impl MultiprocExecutor {
    /// Constructs a cooperative executor from connected physical ranks.
    pub fn new(workers: Vec<Box<dyn PhysicalExecutor>>) -> anyhow::Result<Self> {
        Self::from_workers(workers, Vec::new(), Vec::new(), None)
    }

    /// Validates rank topology and constructs shared cooperative executor state.
    fn from_workers(
        workers: Vec<Box<dyn PhysicalExecutor>>,
        command_wakers: Vec<CommandWaker>,
        progress_fds: Vec<i32>,
        process_args: Option<WorkerProcessArgs>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!workers.is_empty(), "need >= 1 worker");
        let n = workers.len();
        let world_size =
            u32::try_from(n).context("process world size exceeds the IPC representation")?;
        let mut info = workers[0].physical_info().single_pool().clone();
        let mut canonical = info.clone();
        canonical.rank.rank = 0;
        canonical.configuration_id.clear();
        for (rank, worker) in workers.iter().enumerate() {
            let rank_info = worker.physical_info().single_pool();
            rank_info
                .validate()
                .with_context(|| format!("physical rank {rank} reported invalid worker info"))?;
            anyhow::ensure!(
                rank_info.rank.rank == rank as u32 && rank_info.rank.world_size == world_size,
                "physical rank {rank} reported topology ({}/{}) for launched topology ({rank}/{world_size})",
                rank_info.rank.rank,
                rank_info.rank.world_size,
            );
            let mut normalized = rank_info.clone();
            normalized.rank.rank = 0;
            // The resolved identity includes rank-local parameter and buffer
            // layouts. Component placement may therefore give each physical
            // rank a distinct identity while all scheduling/resource bounds
            // still have to agree.
            normalized.configuration_id.clear();
            anyhow::ensure!(
                normalized == canonical,
                "physical rank {rank} worker info disagree with rank 0"
            );
        }
        info.configuration_id = process_world_configuration_id(&workers);
        let depth = info.queue_depth.max(1) as usize;
        let buffers = (0..n).map(|_| VecDeque::new()).collect();
        Ok(Self {
            workers,
            buffers,
            executor_info: ExecutorInfo::single(PoolId("local".to_owned()), info.clone()),
            info,
            depth,
            inflight: 0,
            command_wakers,
            progress_fds,
            last_progress: Instant::now(),
            pending_batches: BTreeMap::new(),
            pending_operations: BTreeMap::new(),
            rank_errors: (0..n).map(|_| BTreeMap::new()).collect(),
            rank_returned_operations: (0..n).map(|_| BTreeMap::new()).collect(),
            rank_successes: (0..n).map(|_| BTreeSet::new()).collect(),
            process_args,
            known_requests: BTreeSet::new(),
            dirty_requests: BTreeSet::new(),
            next_collective_seq: 1,
            logical_results: LogicalResultTracker::default(),
            command_wake_pending: false,
        })
    }

    /// Spawns every physical rank and validates their shared capabilities.
    pub fn spawn(args: WorkerProcessArgs) -> anyhow::Result<Self> {
        anyhow::ensure!(args.world_size > 0, "worker world size must be positive");
        let (workers, wakers, progress_fds) = args.launch()?;
        Self::from_workers(workers, wakers, progress_fds, Some(args))
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
                        let expected = self
                            .pending_batches
                            .get(&run_id)
                            .ok_or_else(|| {
                                anyhow::anyhow!("rank {rank} returned unknown step {run_id}")
                            })?
                            .operations
                            .iter()
                            .map(|operation| {
                                operation_identity(operation.request_key, operation.op_id)
                            })
                            .collect::<BTreeSet<_>>();
                        let returned = self.rank_returned_operations[rank]
                            .entry(run_id)
                            .or_default();
                        for identity in report_operation_ids(&result) {
                            anyhow::ensure!(
                                returned.insert(identity),
                                "rank {rank} returned an operation more than once for step {run_id}"
                            );
                        }
                        anyhow::ensure!(
                            returned.is_subset(&expected),
                            "rank {rank} returned an unplanned lane for step {run_id}"
                        );
                        if *returned == expected {
                            self.rank_successes[rank].insert(run_id);
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
            self.pending_batches.contains_key(&run_id),
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
        let args = self.process_args.clone().ok_or_else(|| {
            anyhow::anyhow!("worker process failed without a restart specification: {cause}")
        })?;
        tracing::warn!(error = %cause, "worker process lost; replacing its complete rank group");
        for worker in &mut self.workers {
            let _ = worker.close_physical();
        }

        let (workers, wakers, progress_fds) =
            args.launch().context("spawning replacement worker ranks")?;
        self.install_replacement(workers, wakers, progress_fds, &args)?;
        self.process_args = Some(args);
        self.discard_requests_after_loss();
        Err(WorkerLossError {
            message: format!(
                "worker process was replaced and affected requests were terminated: {cause:#}"
            ),
        }
        .into())
    }

    /// Installs a capability-compatible replacement rank group and resets rank-local state.
    fn install_replacement(
        &mut self,
        workers: Vec<Box<dyn PhysicalExecutor>>,
        command_wakers: Vec<CommandWaker>,
        progress_fds: Vec<i32>,
        args: &WorkerProcessArgs,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            workers.len() == args.world_size,
            "replacement worker rank count changed"
        );
        for (rank, worker) in workers.iter().enumerate() {
            validate_replacement_info(&self.info, worker.physical_info().single_pool(), rank)?;
        }
        anyhow::ensure!(
            process_world_configuration_id(&workers) == self.info.configuration_id,
            "replacement worker process-world configuration changed"
        );
        self.workers = workers;
        self.command_wakers = command_wakers;
        self.progress_fds = progress_fds;
        self.last_progress = Instant::now();
        self.buffers = (0..args.world_size).map(|_| VecDeque::new()).collect();
        self.rank_errors = (0..args.world_size).map(|_| BTreeMap::new()).collect();
        self.rank_returned_operations = (0..args.world_size).map(|_| BTreeMap::new()).collect();
        self.rank_successes = (0..args.world_size).map(|_| BTreeSet::new()).collect();
        Ok(())
    }

    /// Discards the requests after loss.
    fn discard_requests_after_loss(&mut self) {
        self.pending_batches.clear();
        self.pending_operations.clear();
        self.buffers.iter_mut().for_each(VecDeque::clear);
        self.rank_errors.iter_mut().for_each(BTreeMap::clear);
        self.rank_returned_operations
            .iter_mut()
            .for_each(BTreeMap::clear);
        self.rank_successes.iter_mut().for_each(BTreeSet::clear);
        self.inflight = 0;
        self.known_requests.clear();
        self.dirty_requests.clear();
    }

    /// Discards the inflight.
    fn discard_inflight(&mut self) {
        self.pending_batches.clear();
        self.pending_operations.clear();
        self.buffers.iter_mut().for_each(VecDeque::clear);
        self.rank_errors.iter_mut().for_each(BTreeMap::clear);
        self.rank_returned_operations
            .iter_mut()
            .for_each(BTreeMap::clear);
        self.rank_successes.iter_mut().for_each(BTreeSet::clear);
        self.inflight = 0;
    }

    /// Advances rank I/O until no immediate progress remains.
    fn pump(&mut self) -> anyhow::Result<()> {
        loop {
            match self.pump_once() {
                Ok(()) => return Ok(()),
                Err(error) if error.downcast_ref::<WorkerExecError>().is_some() => {
                    self.discard_inflight();
                    return Err(error);
                }
                Err(error) => {
                    self.recover_workers(&error)?;
                }
            }
        }
    }

    /// Joins mutually agreeing rank reports into one logical physical result.
    fn try_join(&mut self) -> anyhow::Result<Option<RunResult>> {
        self.join_rank_errors()?;
        if self.buffers.iter().any(|buffer| buffer.is_empty()) {
            return Ok(None);
        }
        let Some((run_id, operation_ids)) = self.joinable_report_key() else {
            return Ok(None);
        };

        let mut per_rank = Vec::with_capacity(self.buffers.len());
        for buffer in &mut self.buffers {
            let pos = buffer
                .iter()
                .position(|report| {
                    report.run_id == run_id && report_operation_ids(report) == operation_ids
                })
                .ok_or_else(|| {
                    anyhow::anyhow!("joinable step {run_id} disappeared from rank buffer")
                })?;
            per_rank.push(buffer.remove(pos).ok_or_else(|| {
                anyhow::anyhow!("joinable step {run_id} index disappeared from rank buffer")
            })?);
        }

        let batch = self
            .pending_batches
            .get(&run_id)
            .ok_or_else(|| anyhow::anyhow!("joined step {run_id} has no pending batch"))?;
        for (rank, report) in per_rank.iter_mut().enumerate() {
            validate_and_order_rank_report(batch, report, rank)?;
        }
        let output_rank = self
            .info
            .components
            .iter()
            .find(|component| component.name == "output")
            .map_or(0, |component| component.deployment.ranks[0]);
        let mut out = per_rank.remove(output_rank);
        for (index, report) in per_rank.iter().enumerate() {
            let rank = if index >= output_rank {
                index + 1
            } else {
                index
            };
            merge_rank_report(batch, &mut out, report, rank)?;
        }
        let remaining = self
            .pending_operations
            .get_mut(&run_id)
            .ok_or_else(|| anyhow::anyhow!("joined step {run_id} has no operation ledger"))?;
        for identity in operation_ids {
            anyhow::ensure!(
                remaining.remove(&identity),
                "joined step {run_id} repeated an operation"
            );
        }
        out.done = remaining.is_empty();
        if out.done {
            self.finish_step(run_id);
        }
        Ok(Some(out))
    }

    /// Returns the command-channel waker.
    pub(crate) fn command_waker(&self) -> CommandWaker {
        let wakes = self.command_wakers.clone();
        CommandWaker::new(move || {
            for wake in &wakes {
                wake.wake();
            }
        })
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
            let terminal = (0..self.workers.len())
                .map(|rank| {
                    self.rank_errors[rank].contains_key(&run_id)
                        || self.rank_successes[rank].contains(&run_id)
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
            if errors.len() != self.workers.len() {
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
            self.finish_step(run_id);
            return Err(canonical.into());
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
        for returned in &mut self.rank_returned_operations {
            returned.remove(&run_id);
        }
        for successes in &mut self.rank_successes {
            successes.remove(&run_id);
        }
        self.inflight = self.inflight.saturating_sub(1);
    }

    /// Returns the key used to join a rank report.
    fn joinable_report_key(&self) -> Option<(u64, Vec<OperationIdentity>)> {
        self.buffers[0]
            .iter()
            .filter_map(|report| {
                let run_id = report.run_id;
                let operation_ids = report_operation_ids(report);
                self.buffers
                    .iter()
                    .all(|buffer| {
                        buffer.iter().any(|item| {
                            item.run_id == run_id && report_operation_ids(item) == operation_ids
                        })
                    })
                    .then_some((run_id, operation_ids))
            })
            .min()
    }
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
    anyhow::ensure!(
        participant_report.products.is_empty(),
        "rank {rank} published host products despite designated-rank ownership"
    );
    canonical_report.worker_exec_us = canonical_report
        .worker_exec_us
        .into_iter()
        .chain(participant_report.worker_exec_us)
        .max();
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
        report_count > 0 || batch.operations.is_empty(),
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
    normalized_expected.rank.rank = rank as u32;
    normalized_expected.configuration_id.clear();
    let mut normalized_actual = actual.clone();
    normalized_actual.configuration_id.clear();
    anyhow::ensure!(
        normalized_expected == normalized_actual,
        "replacement rank {rank} worker info changed"
    );
    Ok(())
}

/// Allocates a process-world initialization endpoint.
fn allocate_distributed_init_method() -> anyhow::Result<String> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let addr = listener.local_addr()?;
    Ok(format!("tcp://127.0.0.1:{}", addr.port()))
}

/// Returns the device assigned to a physical worker rank.
fn device_for_rank(device: &str, rank: usize, world_size: usize) -> String {
    let trimmed = device.trim();
    if world_size > 1 && matches!(trimmed, "cuda" | "gpu") {
        format!("cuda:{rank}")
    } else {
        device.to_string()
    }
}

impl PhysicalExecutor for MultiprocExecutor {
    /// Returns metadata for the physical worker.
    fn physical_info(&self) -> &ExecutorInfo {
        &self.executor_info
    }

    /// Broadcasts one validated physical run to every rank as a single ownership transaction.
    fn submit_run(&mut self, batch: PhysicalRun) -> Result<(), PhysicalSubmitError> {
        if self.inflight >= self.depth {
            return Err(PhysicalSubmitError::WouldBlock(batch));
        }
        batch
            .validate()
            .map_err(anyhow::Error::from)
            .map_err(PhysicalSubmitError::Failed)?;
        if self.pending_batches.contains_key(&batch.run_id) {
            return Err(PhysicalSubmitError::Failed(anyhow::anyhow!(
                "step {} is already in flight",
                batch.run_id
            )));
        }
        let run_id = batch.run_id;
        let requests = batch
            .operations()
            .map(|operation| operation.request_key.request_id)
            .collect::<Vec<_>>();
        self.pending_batches.insert(run_id, batch.clone());
        self.pending_operations.insert(
            run_id,
            batch
                .operations
                .iter()
                .map(|operation| operation_identity(operation.request_key, operation.op_id))
                .collect(),
        );
        self.known_requests.extend(requests);
        self.dirty_requests.extend(
            batch
                .operations()
                .map(|operation| operation.request_key.request_id)
                .chain(
                    batch
                        .commands
                        .iter()
                        .map(|control| control.request_key().request_id),
                ),
        );
        for rank in 0..self.workers.len() {
            self.rank_returned_operations[rank].insert(run_id, BTreeSet::new());
        }
        self.inflight += 1;
        self.last_progress = Instant::now();
        for (rank, worker) in self.workers.iter_mut().enumerate() {
            if let Err(error) = worker.submit_run(batch.clone()) {
                let error = match error {
                    PhysicalSubmitError::WouldBlock(_) => {
                        anyhow::anyhow!(
                            "physical rank {rank} rejected an admitted cooperative batch as full"
                        )
                    }
                    PhysicalSubmitError::Failed(error) => {
                        error.context(format!("submit to rank {rank} failed"))
                    }
                };
                let recovered = self.recover_workers(&error).unwrap_err();
                return Err(PhysicalSubmitError::Failed(recovered));
            }
        }
        Ok(())
    }

    /// Waits for rank progress and returns the next fully agreed physical result.
    fn poll_run(&mut self, timeout: Duration) -> anyhow::Result<Option<RunResult>> {
        if self.command_wake_pending {
            return Ok(None);
        }
        // With no submitted run, the only useful wake is a command or worker
        // death. Do not probe the child executors first: probing drains their
        // shared wake descriptors, which can consume a command wake and make
        // the engine sleep for the full liveness interval before reading it.
        if self.inflight == 0 {
            if timeout.is_zero() {
                return Ok(None);
            }
            if self.progress_fds.is_empty() {
                let _ = self.workers[0].poll_run(timeout)?;
            } else {
                crate::worker::park_descriptors(&self.progress_fds, timeout)?;
                self.pump()?;
            }
            if self.command_wake_pending {
                return Ok(None);
            }
            return self.try_join();
        }
        self.pump()?;
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
            if self.progress_fds.is_empty() {
                let idx = (0..self.workers.len())
                    .min_by_key(|rank| self.rank_successes[*rank].len())
                    .unwrap_or(0);
                match self.workers[idx].poll_run(deadline - now) {
                    Ok(Some(result)) => {
                        self.buffers[idx].push_back(result);
                        self.last_progress = Instant::now();
                    }
                    Ok(None) => return Ok(None),
                    Err(error) if error.downcast_ref::<WorkerExecError>().is_some() => {
                        self.record_rank_error(idx, &error)?;
                    }
                    Err(error) => self.recover_workers(&error)?,
                }
            } else {
                crate::worker::park_descriptors(&self.progress_fds, deadline - now)?;
            }
            self.pump()?;
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
                self.recover_workers(&error)?;
            }
        }
    }

    /// Consumes the pending command-wake notification.
    fn take_command_wake(&mut self) -> bool {
        std::mem::take(&mut self.command_wake_pending)
    }

    /// Closes every rank in the physical worker group.
    fn close_physical(&mut self) -> anyhow::Result<()> {
        let mut first_error = None;
        for worker in self.workers.iter_mut() {
            if let Err(error) = worker.close_physical()
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

impl Executor for MultiprocExecutor {
    /// Returns the worker metadata.
    fn info(&self) -> &ExecutorInfo {
        &self.executor_info
    }

    /// Lowers and submits a logical batch while preserving backpressure and result tracking.
    fn submit(&mut self, batch: Batch) -> Result<(), ExecutorSubmitError> {
        if self.inflight >= self.depth {
            return Err(ExecutorSubmitError::WouldBlock(batch));
        }
        let run = lower_batch(&batch, &mut self.next_collective_seq)
            .map_err(ExecutorSubmitError::Failed)?;
        self.logical_results
            .register(&batch)
            .map_err(ExecutorSubmitError::Failed)?;
        match self.submit_run(run) {
            Ok(()) => Ok(()),
            Err(PhysicalSubmitError::WouldBlock(_)) => {
                self.logical_results.unregister(batch.id);
                Err(ExecutorSubmitError::WouldBlock(batch))
            }
            Err(PhysicalSubmitError::Failed(error)) => {
                self.logical_results.unregister(batch.id);
                Err(ExecutorSubmitError::Failed(error))
            }
        }
    }

    /// Polls for the next completed worker operation.
    fn poll(&mut self, timeout: Duration) -> anyhow::Result<Option<BatchResult>> {
        if self.take_command_wake() {
            return Ok(None);
        }
        let report = self.poll_run(timeout)?;
        if report.is_none() {
            self.take_command_wake();
        }
        report
            .map(|report| self.logical_results.apply(report))
            .transpose()
    }

    /// Closes the component and releases its resources.
    fn close(&mut self) -> anyhow::Result<()> {
        self.close_physical()
    }
}
