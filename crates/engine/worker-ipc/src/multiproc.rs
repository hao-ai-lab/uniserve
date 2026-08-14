use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::net::TcpListener;
use std::time::{Duration, Instant};

use anyhow::Context;
use uniserve_core::{CommandWaker, RequestId};
use uniserve_executor::{ControlAck, ControlOp, Executor, WorkerExecError, WorkerLossError};
use uniserve_worker_wire::{
    Batch, CompletionRecord, CompletionReport, SamplingOwnership, WorkerCapabilities,
};

use crate::WorkerLaunchConfig;

/// How long a single rank may go without producing output, while batches are in
/// flight, before `next_result` treats it as dead and bails. Generous so a
/// healthy-but-slow forward is never falsely failed; short enough that a hung
/// rank does not wedge the scheduler loop forever.
const NEXT_RESULT_DEADLINE: Duration = Duration::from_secs(300);
/// Per-iteration bounded wait used while draining a lagging rank.
const NEXT_RESULT_POLL: Duration = Duration::from_millis(1);

#[derive(Clone)]
struct MultiprocSpawnSpec {
    python: String,
    model_dir: String,
    device: String,
    world_size: usize,
    pipeline_depth: usize,
    req_slot_cap: usize,
    resp_slot_cap: usize,
    kv_token_capacity: Option<u64>,
    block_size: u32,
    max_batch_tokens: u32,
    attention_backend: String,
    worker_kind: Option<String>,
    transfer_backend: Option<String>,
    defer_sampling: bool,
    worker_config: WorkerLaunchConfig,
}

impl MultiprocSpawnSpec {
    fn launch(&self) -> anyhow::Result<Vec<Box<dyn Executor>>> {
        let tp_init_method = if self.world_size > 1 {
            Some(allocate_tp_init_method()?)
        } else {
            None
        };
        let mut launched = Vec::with_capacity(self.world_size);
        for rank in 0..self.world_size {
            let rank_device = device_for_rank(&self.device, rank, self.world_size);
            let mut rank_config = self.worker_config.clone();
            if let Some(root) = &self.worker_config.snapshot_dir {
                rank_config.snapshot_dir = Some(
                    std::path::PathBuf::from(root)
                        .join("ranks")
                        .join(rank.to_string())
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            launched.push(crate::UniprocExecutor::spawn_ranked_deferred_with_config(
                &self.python,
                &self.model_dir,
                &rank_device,
                self.pipeline_depth,
                self.req_slot_cap,
                self.resp_slot_cap,
                self.kv_token_capacity,
                self.block_size,
                self.max_batch_tokens,
                &self.attention_backend,
                rank as u32,
                self.world_size as u32,
                tp_init_method.as_deref(),
                self.worker_kind.as_deref(),
                self.transfer_backend.as_deref(),
                self.defer_sampling,
                &rank_config,
            )?);
        }
        let mut workers: Vec<Box<dyn Executor>> = Vec::with_capacity(self.world_size);
        for mut worker in launched {
            worker.finish_startup()?;
            workers.push(Box::new(worker));
        }
        Ok(workers)
    }
}

/// W worker processes, one iceoryx2 request-response service each.
pub struct MultiprocExecutor {
    workers: Vec<Box<dyn Executor>>,
    buffers: Vec<VecDeque<CompletionReport>>,
    caps: WorkerCapabilities,
    depth: usize,
    inflight: usize,
    next_call_id: u64,
    pending_batches: BTreeMap<u64, Batch>,
    pending_partitions: BTreeMap<u64, BTreeSet<u32>>,
    last_collective_seq: u64,
    spawn_spec: Option<MultiprocSpawnSpec>,
    known_sessions: BTreeSet<RequestId>,
    dirty_sessions: BTreeSet<RequestId>,
}

impl MultiprocExecutor {
    pub fn new(workers: Vec<Box<dyn Executor>>) -> anyhow::Result<Self> {
        Self::from_workers(workers, None)
    }

    fn from_workers(
        workers: Vec<Box<dyn Executor>>,
        spawn_spec: Option<MultiprocSpawnSpec>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!workers.is_empty(), "need >= 1 worker");
        let n = workers.len();
        let tp_size = u32::try_from(n).context("TP world size exceeds the wire representation")?;
        let caps = workers[0].caps();
        let mut canonical = caps.clone();
        canonical.rank.tp_rank = 0;
        for (rank, worker) in workers.iter().enumerate() {
            let rank_caps = worker.caps();
            rank_caps
                .validate()
                .with_context(|| format!("TP rank {rank} reported invalid capabilities"))?;
            anyhow::ensure!(
                rank_caps.rank.tp_rank == rank as u32 && rank_caps.rank.tp_size == tp_size,
                "TP rank {rank} reported topology ({}/{}) for launched topology ({rank}/{tp_size})",
                rank_caps.rank.tp_rank,
                rank_caps.rank.tp_size,
            );
            anyhow::ensure!(
                worker.pipeline_depth() == rank_caps.pipeline_depth.max(1) as usize,
                "TP rank {rank} executor depth {} disagrees with capability depth {}",
                worker.pipeline_depth(),
                rank_caps.pipeline_depth.max(1),
            );
            let mut normalized = rank_caps;
            normalized.rank.tp_rank = 0;
            anyhow::ensure!(
                normalized == canonical,
                "TP rank {rank} capabilities disagree with rank 0"
            );
        }
        let depth = caps.pipeline_depth.max(1) as usize;
        let buffers = (0..n).map(|_| VecDeque::new()).collect();
        Ok(Self {
            workers,
            buffers,
            caps,
            depth,
            inflight: 0,
            next_call_id: 1,
            pending_batches: BTreeMap::new(),
            pending_partitions: BTreeMap::new(),
            last_collective_seq: 0,
            spawn_spec,
            known_sessions: BTreeSet::new(),
            dirty_sessions: BTreeSet::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        python: &str,
        model_dir: &str,
        device: &str,
        world_size: usize,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        max_batch_tokens: u32,
        attention_backend: &str,
    ) -> anyhow::Result<Self> {
        Self::spawn_with_config(
            python,
            model_dir,
            device,
            world_size,
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            max_batch_tokens,
            attention_backend,
            &WorkerLaunchConfig::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn_with_config(
        python: &str,
        model_dir: &str,
        device: &str,
        world_size: usize,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        max_batch_tokens: u32,
        attention_backend: &str,
        worker_config: &WorkerLaunchConfig,
    ) -> anyhow::Result<Self> {
        Self::spawn_inner(
            python,
            model_dir,
            device,
            world_size,
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            max_batch_tokens,
            attention_backend,
            None,
            None,
            false,
            worker_config,
        )
    }

    /// Spawn a tensor-parallel pool for a staged worker kind: every rank is told
    /// which pipeline stage it serves. `world_size == 1` yields a one-rank pool
    /// (still a valid `Executor`), so the `StageRouter` composition can treat
    /// every pool uniformly regardless of its tp size.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_staged(
        python: &str,
        model_dir: &str,
        device: &str,
        world_size: usize,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        max_batch_tokens: u32,
        attention_backend: &str,
        worker_kind: &str,
        transfer_backend: Option<&str>,
        defer_sampling: bool,
    ) -> anyhow::Result<Self> {
        Self::spawn_staged_with_config(
            python,
            model_dir,
            device,
            world_size,
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            max_batch_tokens,
            attention_backend,
            worker_kind,
            transfer_backend,
            defer_sampling,
            &WorkerLaunchConfig::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn_staged_with_config(
        python: &str,
        model_dir: &str,
        device: &str,
        world_size: usize,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        max_batch_tokens: u32,
        attention_backend: &str,
        worker_kind: &str,
        transfer_backend: Option<&str>,
        defer_sampling: bool,
        worker_config: &WorkerLaunchConfig,
    ) -> anyhow::Result<Self> {
        Self::spawn_inner(
            python,
            model_dir,
            device,
            world_size,
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            max_batch_tokens,
            attention_backend,
            Some(worker_kind),
            transfer_backend,
            defer_sampling,
            worker_config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_inner(
        python: &str,
        model_dir: &str,
        device: &str,
        world_size: usize,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        max_batch_tokens: u32,
        attention_backend: &str,
        worker_kind: Option<&str>,
        transfer_backend: Option<&str>,
        defer_sampling: bool,
        worker_config: &WorkerLaunchConfig,
    ) -> anyhow::Result<Self> {
        let spec = MultiprocSpawnSpec {
            python: python.to_string(),
            model_dir: model_dir.to_string(),
            device: device.to_string(),
            world_size: world_size.max(1),
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            max_batch_tokens,
            attention_backend: attention_backend.to_string(),
            worker_kind: worker_kind.map(str::to_string),
            transfer_backend: transfer_backend.map(str::to_string),
            defer_sampling,
            worker_config: worker_config.clone(),
        };
        let workers = spec.launch()?;
        Self::from_workers(workers, Some(spec))
    }

    fn pump_once(&mut self) -> anyhow::Result<()> {
        for (i, worker) in self.workers.iter_mut().enumerate() {
            while let Some(result) = worker.poll()? {
                self.buffers[i].push_back(result);
            }
        }
        Ok(())
    }

    fn recover_workers(&mut self, cause: &anyhow::Error) -> anyhow::Result<()> {
        let spec = self.spawn_spec.clone().ok_or_else(|| {
            anyhow::anyhow!("worker process failed without a restart specification: {cause}")
        })?;
        tracing::warn!(error = %cause, "worker process lost; replacing its complete rank group");
        for worker in &mut self.workers {
            worker.shutdown();
        }

        let workers = spec.launch().context("spawning replacement worker ranks")?;
        self.install_replacement(workers, &spec)?;
        self.spawn_spec = Some(spec);
        self.discard_sessions_after_loss();
        Err(WorkerLossError {
            message: "worker process was replaced and affected sessions were terminated"
                .to_string(),
        }
        .into())
    }

    fn install_replacement(
        &mut self,
        workers: Vec<Box<dyn Executor>>,
        spec: &MultiprocSpawnSpec,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            workers.len() == spec.world_size,
            "replacement worker rank count changed"
        );
        for (rank, worker) in workers.iter().enumerate() {
            validate_replacement_caps(&self.caps, &worker.caps(), rank)?;
        }
        self.workers = workers;
        self.buffers = (0..spec.world_size).map(|_| VecDeque::new()).collect();
        Ok(())
    }

    fn discard_sessions_after_loss(&mut self) {
        self.pending_batches.clear();
        self.pending_partitions.clear();
        self.buffers.iter_mut().for_each(VecDeque::clear);
        self.inflight = 0;
        self.known_sessions.clear();
        self.dirty_sessions.clear();
    }

    fn discard_inflight(&mut self) {
        self.pending_batches.clear();
        self.buffers.iter_mut().for_each(VecDeque::clear);
        self.inflight = 0;
    }

    fn apply_control_session_effect(&mut self, operation: &ControlOp, succeeded: bool) {
        if !succeeded {
            return;
        }
        match operation {
            ControlOp::DropSession(session_id) => {
                self.known_sessions.remove(session_id);
                self.dirty_sessions.remove(session_id);
            }
            ControlOp::RestoreSession { snapshot, .. } => {
                let session_id = snapshot.version.request_key.session_id;
                self.known_sessions.insert(session_id);
                self.dirty_sessions.remove(&session_id);
            }
            ControlOp::CopyKv(_) | ControlOp::ReleaseProducts(_) => {
                self.dirty_sessions
                    .extend(self.known_sessions.iter().copied());
            }
            _ => {}
        }
    }

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

    fn try_join(&mut self) -> anyhow::Result<Option<CompletionReport>> {
        if self.buffers.iter().any(|buffer| buffer.is_empty()) {
            return Ok(None);
        }
        let Some((step_id, partition_ids)) = self.joinable_report_key() else {
            return Ok(None);
        };

        let mut per_rank = Vec::with_capacity(self.buffers.len());
        for buffer in &mut self.buffers {
            let pos = buffer
                .iter()
                .position(|report| {
                    report.step_id == step_id && report_partition_ids(report) == partition_ids
                })
                .ok_or_else(|| {
                    anyhow::anyhow!("joinable step {step_id} disappeared from rank buffer")
                })?;
            per_rank.push(buffer.remove(pos).ok_or_else(|| {
                anyhow::anyhow!("joinable step {step_id} index disappeared from rank buffer")
            })?);
        }

        let batch = self
            .pending_batches
            .get(&step_id)
            .ok_or_else(|| anyhow::anyhow!("joined step {step_id} has no pending batch"))?;
        for (rank, report) in per_rank.iter_mut().enumerate() {
            validate_and_order_rank_report(batch, report, rank)?;
        }
        let mut out = per_rank.remove(0);
        for (rank, report) in per_rank.iter().enumerate() {
            merge_rank_report(&self.caps, batch, &mut out, report, rank + 1)?;
        }
        let remaining = self
            .pending_partitions
            .get_mut(&step_id)
            .ok_or_else(|| anyhow::anyhow!("joined step {step_id} has no partition ledger"))?;
        for partition_id in partition_ids {
            anyhow::ensure!(
                remaining.remove(&partition_id),
                "joined step {step_id} repeated partition {partition_id}"
            );
        }
        if remaining.is_empty() {
            self.inflight = self.inflight.saturating_sub(1);
            self.pending_batches.remove(&step_id);
            self.pending_partitions.remove(&step_id);
        }
        Ok(Some(out))
    }

    fn joinable_report_key(&self) -> Option<(u64, Vec<u32>)> {
        self.buffers[0]
            .iter()
            .filter_map(|report| {
                let step_id = report.step_id;
                let partition_ids = report_partition_ids(report);
                self.buffers
                    .iter()
                    .all(|buffer| {
                        buffer.iter().any(|item| {
                            item.step_id == step_id && report_partition_ids(item) == partition_ids
                        })
                    })
                    .then_some((step_id, partition_ids))
            })
            .min()
    }
}

fn report_partition_ids(report: &CompletionReport) -> Vec<u32> {
    let mut ids = report
        .partitions
        .iter()
        .map(|partition| partition.partition_id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

/// Join one step's per-rank completion reports into rank 0's. Tensor-parallel
/// ranks run the identical operation set, so every rank must report the same
/// semantic completion for each operation; only the per-rank product shards
/// differ and are concatenated in rank order.
fn merge_rank_report(
    caps: &WorkerCapabilities,
    batch: &Batch,
    rank0: &mut CompletionReport,
    rankn: &CompletionReport,
    rank: usize,
) -> anyhow::Result<()> {
    let step_id = batch.step_id;
    if rankn.step_id != step_id {
        anyhow::bail!(
            "rank {rank} completion report step_id mismatch while joining step {step_id}: got {}",
            rankn.step_id
        );
    }
    if rankn.partitions.len() != rank0.partitions.len() {
        anyhow::bail!(
            "rank {rank} report for step {step_id} has {} partitions, expected {}",
            rankn.partitions.len(),
            rank0.partitions.len()
        );
    }
    for (partition_index, (canonical_partition, actual_partition)) in rank0
        .partitions
        .iter_mut()
        .zip(&rankn.partitions)
        .enumerate()
    {
        anyhow::ensure!(
            batch
                .partitions
                .iter()
                .any(|partition| partition.partition_id == canonical_partition.partition_id),
            "rank join received unplanned partition {} for step {step_id}",
            canonical_partition.partition_id
        );
        let ownership = caps.sampling_ownership;
        anyhow::ensure!(
            canonical_partition.partition_id == actual_partition.partition_id,
            "rank {rank} report for step {step_id} partition {partition_index} identity differs from rank 0"
        );
        anyhow::ensure!(
            canonical_partition.completions.len() == actual_partition.completions.len(),
            "rank {rank} report for step {step_id} partition {} has {} completions, expected {}",
            canonical_partition.partition_id,
            actual_partition.completions.len(),
            canonical_partition.completions.len()
        );
        for (completion_index, (canonical, actual)) in canonical_partition
            .completions
            .iter_mut()
            .zip(&actual_partition.completions)
            .enumerate()
        {
            if let Err(error) = merge_completion_record(canonical, actual, ownership) {
                anyhow::bail!(
                    "rank {rank} report for step {step_id} partition {} completion {completion_index} differs from rank 0: {error:#}",
                    canonical_partition.partition_id
                );
            }
        }
        match ownership {
            SamplingOwnership::DesignatedRank => anyhow::ensure!(
                actual_partition.products.is_empty(),
                "rank {rank} published host products for designated-rank partition {}",
                canonical_partition.partition_id
            ),
            SamplingOwnership::DeterministicSharded => canonical_partition
                .products
                .extend(actual_partition.products.iter().cloned()),
        }
        canonical_partition.worker_exec_us = canonical_partition
            .worker_exec_us
            .into_iter()
            .chain(actual_partition.worker_exec_us)
            .max();
    }
    Ok(())
}

fn validate_and_order_rank_report(
    batch: &Batch,
    report: &mut CompletionReport,
    rank: usize,
) -> anyhow::Result<()> {
    report.validate()?;
    anyhow::ensure!(
        report.step_id == batch.step_id,
        "rank {rank} returned step {} for pending step {}",
        report.step_id,
        batch.step_id
    );
    let report_count = report.partitions.len();
    let returned = report_partition_ids(report)
        .into_iter()
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        report_count > 0 || batch.partitions.is_empty(),
        "rank {rank} returned an empty partial report for step {}",
        batch.step_id
    );
    let mut ordered = Vec::with_capacity(report_count);
    for planned in &batch.partitions {
        if !returned.contains(&planned.partition_id) {
            continue;
        }
        let index = report
            .partitions
            .iter()
            .position(|partition| partition.partition_id == planned.partition_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "rank {rank} omitted partition {} for step {} collective {}",
                    planned.partition_id,
                    batch.step_id,
                    planned.collective_seq
                )
            })?;
        let actual = report.partitions.swap_remove(index);
        anyhow::ensure!(
            actual.completions.len() == planned.operations.len(),
            "rank {rank} partition {} returned {} completions for {} operations",
            planned.partition_id,
            actual.completions.len(),
            planned.operations.len()
        );
        for (operation, completion) in planned.operations.iter().zip(&actual.completions) {
            anyhow::ensure!(
                operation.request_key == completion.request_key
                    && operation.op_id == completion.op_id,
                "rank {rank} partition {} completion order disagrees with collective {}",
                planned.partition_id,
                planned.collective_seq
            );
        }
        ordered.push(actual);
    }
    anyhow::ensure!(
        ordered.len() == report_count,
        "rank {rank} returned an unplanned partition for step {}",
        batch.step_id
    );
    report.partitions = ordered;
    Ok(())
}

/// Merge one rank's completion record into rank 0's. Every rank must agree on
/// the semantic result; only `product_generations` are per-rank shards, which
/// concatenate in rank order.
fn merge_completion_record(
    canonical: &mut CompletionRecord,
    rank_completion: &CompletionRecord,
    ownership: SamplingOwnership,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        canonical.request_key == rank_completion.request_key
            && canonical.op_id == rank_completion.op_id
            && canonical.selected_point == rank_completion.selected_point
            && canonical.semantic_digest == rank_completion.semantic_digest,
        "completion identity or selected point diverged"
    );
    anyhow::ensure!(
        canonical.committed_tokens == rank_completion.committed_tokens
            && canonical.status == rank_completion.status
            && canonical.logical_lengths == rank_completion.logical_lengths
            && canonical.token_span == rank_completion.token_span
            && canonical.finish_flags == rank_completion.finish_flags,
        "completion result fields diverged"
    );
    match ownership {
        SamplingOwnership::DesignatedRank => anyhow::ensure!(
            canonical.product_generations == rank_completion.product_generations,
            "designated-rank product generations diverged"
        ),
        SamplingOwnership::DeterministicSharded => canonical
            .product_generations
            .extend(rank_completion.product_generations.iter().copied()),
    }
    Ok(())
}

fn validate_replacement_caps(
    expected: &WorkerCapabilities,
    actual: &WorkerCapabilities,
    rank: usize,
) -> anyhow::Result<()> {
    actual
        .validate()
        .with_context(|| format!("replacement rank {rank} reported invalid capabilities"))?;
    let mut normalized_expected = expected.clone();
    normalized_expected.rank.tp_rank = rank as u32;
    anyhow::ensure!(
        normalized_expected == *actual,
        "replacement rank {rank} capabilities changed"
    );
    Ok(())
}

fn allocate_tp_init_method() -> anyhow::Result<String> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let addr = listener.local_addr()?;
    Ok(format!("tcp://127.0.0.1:{}", addr.port()))
}

fn device_for_rank(device: &str, rank: usize, world_size: usize) -> String {
    let trimmed = device.trim();
    if world_size > 1 && matches!(trimmed, "cuda" | "gpu") {
        format!("cuda:{rank}")
    } else {
        device.to_string()
    }
}

impl Executor for MultiprocExecutor {
    fn caps(&self) -> WorkerCapabilities {
        self.caps.clone()
    }

    fn pipeline_depth(&self) -> usize {
        self.depth
    }

    fn in_flight(&self) -> usize {
        self.inflight
    }

    fn submit(&mut self, batch: Batch) -> anyhow::Result<()> {
        batch.validate()?;
        anyhow::ensure!(
            !self.pending_batches.contains_key(&batch.step_id),
            "step {} is already in flight",
            batch.step_id
        );
        let step_id = batch.step_id;
        let mut collective_sequences = batch
            .partitions
            .iter()
            .map(|partition| partition.collective_seq)
            .collect::<Vec<_>>();
        collective_sequences.sort_unstable();
        collective_sequences.dedup();
        anyhow::ensure!(
            collective_sequences
                .windows(2)
                .all(|window| window[0] < window[1])
                && collective_sequences
                    .first()
                    .is_none_or(|sequence| *sequence > self.last_collective_seq),
            "step {step_id} collective order does not advance beyond {}",
            self.last_collective_seq
        );
        let sessions = batch
            .operations()
            .map(|operation| operation.request_key.session_id)
            .collect::<Vec<_>>();
        self.pending_batches.insert(step_id, batch.clone());
        self.pending_partitions.insert(
            step_id,
            batch
                .partitions
                .iter()
                .map(|partition| partition.partition_id)
                .collect(),
        );
        self.known_sessions.extend(sessions);
        self.dirty_sessions.extend(
            batch
                .operations()
                .map(|operation| operation.request_key.session_id)
                .chain(
                    batch
                        .controls
                        .iter()
                        .map(|control| control.request_key().session_id),
                ),
        );
        if let Some(sequence) = collective_sequences.last() {
            self.last_collective_seq = *sequence;
        }
        self.inflight += 1;
        for (rank, worker) in self.workers.iter_mut().enumerate() {
            if let Err(error) = worker
                .submit(batch.clone())
                .with_context(|| format!("submit to rank {rank} failed"))
            {
                return self.recover_workers(&error);
            }
        }
        Ok(())
    }

    fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
        self.pump()?;
        self.try_join()
    }

    fn check_liveness(&mut self) -> anyhow::Result<()> {
        for (rank, worker) in self.workers.iter_mut().enumerate() {
            if let Err(error) = worker
                .check_liveness()
                .with_context(|| format!("rank {rank} liveness check failed"))
            {
                return self.recover_workers(&error);
            }
        }
        Ok(())
    }

    fn event_driven(&self) -> bool {
        // The pool is event-driven only when every rank is, so the scheduler's
        // single park covers them all (a mixed pool falls back to polling).
        !self.workers.is_empty() && self.workers.iter().all(|w| w.event_driven())
    }

    fn command_waker(&self) -> CommandWaker {
        // TP ranks run in lockstep, so rank 0's waker is enough to break the
        // park on a freshly enqueued command (the scheduler then drains all).
        self.workers
            .first()
            .map(|w| w.command_waker())
            .unwrap_or_else(CommandWaker::noop)
    }

    fn park_for_event(&mut self, timeout: std::time::Duration) -> anyhow::Result<()> {
        // Every rank receives the same batch and finishes within a hair of the
        // others, so parking on rank 0's wake surfaces a ready result without a
        // cross-endpoint WaitSet; a spurious wake is harmless because the
        // scheduler re-drains every rank after the park returns.
        match self.workers.first_mut() {
            Some(worker) => match worker.park_for_event(timeout) {
                Ok(()) => Ok(()),
                Err(error) => self.recover_workers(&error),
            },
            None => {
                std::thread::sleep(timeout.min(Duration::from_millis(1)));
                Ok(())
            }
        }
    }

    fn wait_result_timeout(
        &mut self,
        timeout: Duration,
    ) -> anyhow::Result<Option<CompletionReport>> {
        let Some(deadline) = Instant::now().checked_add(timeout) else {
            self.pump()?;
            return self.try_join();
        };
        loop {
            self.pump()?;
            if let Some(result) = self.try_join()? {
                return Ok(Some(result));
            }
            if self.inflight == 0 {
                return Ok(None);
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            std::thread::sleep((deadline - now).min(Duration::from_millis(1)));
        }
    }

    fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
        if self.inflight == 0 {
            anyhow::bail!("next_result called with no in-flight batches");
        }
        // Bound the wait so a single dead/hung rank surfaces as an executor
        // error instead of wedging the scheduler loop forever. The deadline is
        // reset whenever any rank makes progress, so a healthy-but-slow batch is
        // never falsely failed.
        let mut deadline = Instant::now() + NEXT_RESULT_DEADLINE;
        loop {
            self.pump()?;
            if let Some(result) = self.try_join()? {
                return Ok(result);
            }
            // Wait for fresh output on whichever rank is currently behind. Using
            // the per-worker bounded wait (rather than the unbounded
            // next_result) keeps a hung rank from blocking indefinitely.
            let idx = self
                .buffers
                .iter()
                .position(|buffer| buffer.is_empty())
                .unwrap_or(0);
            match self.workers[idx].wait_result_timeout(NEXT_RESULT_POLL) {
                Ok(Some(result)) => {
                    self.buffers[idx].push_back(result);
                    deadline = Instant::now() + NEXT_RESULT_DEADLINE;
                }
                Ok(None) if Instant::now() >= deadline => {
                    let error = anyhow::anyhow!(
                        "rank {idx} produced no result within {:?} while {} batch(es) were in flight",
                        NEXT_RESULT_DEADLINE,
                        self.inflight
                    );
                    self.recover_workers(&error)?;
                    deadline = Instant::now() + NEXT_RESULT_DEADLINE;
                }
                Ok(None) => {}
                Err(error) if error.downcast_ref::<WorkerExecError>().is_some() => {
                    self.discard_inflight();
                    return Err(error);
                }
                Err(error) => {
                    self.recover_workers(&error)?;
                    deadline = Instant::now() + NEXT_RESULT_DEADLINE;
                }
            }
        }
    }

    fn control(&mut self, op: ControlOp) -> anyhow::Result<u64> {
        // NOTE: the returned id is a local fire-and-forget token, NOT a wire
        // call_id. Each per-rank worker.control allocates its own real
        // call_id internally; this counter correlates to none of them. Callers
        // must not use this value to match a later worker ack — use
        // control_wait (which fans out and collects per-rank acks) for any
        // call that needs correlation.
        let call_id = self.next_call_id;
        self.next_call_id += 1;
        loop {
            let mut failure = None;
            for rank in 0..self.workers.len() {
                if let Err(error) = self.workers[rank].control(op.clone()) {
                    failure = Some(error.context(format!("control call failed on rank {rank}")));
                    break;
                }
            }
            match failure {
                Some(error) => self.recover_workers(&error)?,
                None => break,
            }
        }
        self.apply_control_session_effect(&op, true);
        Ok(call_id)
    }

    fn control_wait(
        &mut self,
        op: ControlOp,
        targets: Option<&[u32]>,
    ) -> anyhow::Result<Vec<ControlAck>> {
        loop {
            let mut acks = Vec::new();
            let mut failure = None;
            for rank in 0..self.workers.len() {
                let rank = rank as u32;
                if let Some(targets) = targets
                    && !targets.contains(&rank)
                {
                    continue;
                }
                match self.workers[rank as usize].control_wait(op.clone(), None) {
                    Ok(rank_acks) => {
                        for mut ack in rank_acks {
                            ack.rank = rank;
                            acks.push(ack);
                        }
                    }
                    Err(error) => {
                        failure =
                            Some(error.context(format!("control wait failed on rank {rank}")));
                        break;
                    }
                }
            }
            if let Some(error) = failure {
                self.recover_workers(&error)?;
                continue;
            }
            let succeeded = acks.iter().all(|ack| ack.ok);
            if succeeded && let ControlOp::SnapshotSession(placement) = &op {
                let session_id = placement.request_key.session_id;
                let references = acks
                    .iter()
                    .map(|ack| {
                        ack.snapshot.as_ref().ok_or_else(|| {
                            anyhow::anyhow!(
                                "rank {} acknowledged session {} snapshot without a reference",
                                ack.rank,
                                session_id.0
                            )
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                anyhow::ensure!(
                    !references.is_empty(),
                    "session snapshot control selected no tensor-parallel rank"
                );
                let first = references[0];
                anyhow::ensure!(
                    references.iter().all(|reference| {
                        reference.version.request_key.session_id == session_id
                            && reference.version == first.version
                    }),
                    "tensor-parallel snapshot ranks disagree on session, epoch, or version"
                );
                self.dirty_sessions.remove(&session_id);
            }
            self.apply_control_session_effect(&op, succeeded);
            return Ok(acks);
        }
    }

    fn shutdown(&mut self) {
        for worker in self.workers.iter_mut() {
            worker.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, VecDeque};
    use std::time::Duration;

    use super::{MultiprocExecutor, device_for_rank, report_partition_ids};
    use uniserve_core::RequestId;
    use uniserve_executor::{ControlAck, ControlOp, Executor};
    use uniserve_worker_wire::{
        AttentionRegime, Batch, BatchPartition, Bounds, CompletionRecord, CompletionReport, DType,
        Domain, ExecutionCapability, FinishFlags, LogicalLengths, OpId, OpStatus, Operation,
        PartitionCompletion, PointRange, ProductKind, ProductPayload, ProductRef, RegistrationAck,
        RequestKey, RouteId, SamplingOwnership, ShapeBound, StorageClass, TimingCounters,
        TokenMode, TokenSpan, VersionRef, Work, WorkerCapabilities,
    };

    #[test]
    fn cuda_device_is_ranked_for_multiproc() {
        assert_eq!(device_for_rank("cuda", 1, 4), "cuda:1");
        assert_eq!(device_for_rank("gpu", 2, 4), "cuda:2");
        assert_eq!(device_for_rank("cuda:0", 1, 4), "cuda:0");
        assert_eq!(device_for_rank("cpu", 1, 4), "cpu");
        assert_eq!(device_for_rank("cuda", 0, 1), "cuda");
    }

    #[test]
    fn multiproc_construction_requires_exact_rank_capability_agreement() {
        let exec = MultiprocExecutor::new(vec![fake_worker(0, 2), fake_worker(1, 2)]).unwrap();
        assert_eq!(exec.caps.rank.tp_rank, 0);
        assert_eq!(exec.caps.rank.tp_size, 2);
        assert_eq!(exec.pipeline_depth(), 2);

        let topology_error = MultiprocExecutor::new(vec![fake_worker(0, 2), fake_worker(0, 2)])
            .err()
            .unwrap()
            .to_string();
        assert!(
            topology_error.contains("reported topology"),
            "got: {topology_error}"
        );

        let mut disagreeing_caps = fake_caps(1, 2);
        disagreeing_caps.num_blocks += 1;
        let capability_error = MultiprocExecutor::new(vec![
            fake_worker(0, 2),
            Box::new(FakeExec {
                caps: disagreeing_caps,
                queued: VecDeque::new(),
            }),
        ])
        .err()
        .unwrap()
        .to_string();
        assert!(
            capability_error.contains("capabilities disagree"),
            "got: {capability_error}"
        );

        let mut invalid_protocol = fake_caps(1, 2);
        invalid_protocol.protocol_layout_digest = "0".repeat(64);
        let protocol_error = MultiprocExecutor::new(vec![
            fake_worker(0, 2),
            Box::new(FakeExec {
                caps: invalid_protocol,
                queued: VecDeque::new(),
            }),
        ])
        .err()
        .unwrap()
        .to_string();
        assert!(
            protocol_error.contains("invalid capabilities"),
            "got: {protocol_error}"
        );
    }

    #[test]
    fn multiproc_next_result_bails_with_no_inflight() {
        let mut exec = fake_multiproc(2);
        exec.inflight = 0;
        let err = exec.next_result().unwrap_err().to_string();
        assert!(err.contains("no in-flight batches"), "got: {err}");
    }

    #[test]
    fn multiproc_join_matches_by_step_id_without_consuming_unmatched_front() {
        let mut exec = fake_multiproc(2);
        exec.inflight = 2;
        queue_pending(&mut exec, &[1, 2]);
        exec.buffers[0].push_back(result(2, 20));
        exec.buffers[0].push_back(result(1, 10));
        exec.buffers[1].push_back(result(1, 10));

        let joined = exec.try_join().unwrap().unwrap();

        assert_eq!(joined.step_id, 1);
        assert_eq!(exec.inflight, 1);
        assert_eq!(exec.buffers[0].front().unwrap().step_id, 2);
        assert!(exec.buffers[1].is_empty());
    }

    #[test]
    fn multiproc_join_rejects_rank_token_divergence() {
        let mut exec = fake_multiproc(2);
        exec.inflight = 1;
        queue_pending(&mut exec, &[1]);
        exec.buffers[0].push_back(result(1, 10));
        exec.buffers[1].push_back(result(1, 11));

        let err = exec.try_join().unwrap_err().to_string();

        assert!(err.contains("completion 0 differs"), "got: {err}");
        assert_eq!(exec.inflight, 1);
    }

    #[test]
    fn multiproc_join_requires_identical_selected_point_and_semantic_digest() {
        let mut selected = fake_multiproc(2);
        selected.inflight = 1;
        queue_pending(&mut selected, &[1]);
        selected.buffers[0].push_back(result(1, 10));
        let mut selected_divergence = result(1, 10);
        selected_divergence.partitions[0].completions[0].selected_point = 1;
        selected.buffers[1].push_back(selected_divergence);
        let selected_error = selected.try_join().unwrap_err().to_string();
        assert!(
            selected_error.contains("completion 0 differs"),
            "got: {selected_error}"
        );

        let mut digest = fake_multiproc(2);
        digest.inflight = 1;
        queue_pending(&mut digest, &[1]);
        digest.buffers[0].push_back(result(1, 10));
        let mut digest_divergence = result(1, 10);
        digest_divergence.partitions[0].completions[0].semantic_digest = "1".repeat(64);
        digest.buffers[1].push_back(digest_divergence);
        let digest_error = digest.try_join().unwrap_err().to_string();
        assert!(
            digest_error.contains("completion 0 differs"),
            "got: {digest_error}"
        );
    }

    #[test]
    fn multiproc_join_concatenates_rank_ordered_products() {
        // The tensor-parallel join concatenates each completion's
        // `product_generations` and the report-level `products` in rank order,
        // while the semantic completion is required to be identical per rank.
        let mut exec = fake_multiproc(2);
        exec.caps.sampling_ownership = SamplingOwnership::DeterministicSharded;
        exec.inflight = 1;
        queue_pending(&mut exec, &[1]);
        exec.buffers[0].push_back(result_with_products(1, vec![10, 11], 100));
        exec.buffers[1].push_back(result_with_products(1, vec![20, 21], 200));

        let joined = exec.try_join().unwrap().unwrap();

        assert_eq!(
            joined.partitions[0].completions[0].product_generations,
            vec![10, 11, 20, 21]
        );
        assert_eq!(joined.partitions[0].products.len(), 2);
        assert_eq!(joined.partitions[0].products[0].product.generation, 100);
        assert_eq!(joined.partitions[0].products[1].product.generation, 200);

        // Identity / committed-token divergence across ranks is still rejected.
        let mut exec = fake_multiproc(2);
        exec.caps.sampling_ownership = SamplingOwnership::DeterministicSharded;
        exec.inflight = 1;
        queue_pending(&mut exec, &[1]);
        exec.buffers[0].push_back(result_with_products(1, vec![10], 100));
        let mut diverged = result_with_products(1, vec![20], 200);
        diverged.partitions[0].completions[0].committed_tokens = vec![99];
        exec.buffers[1].push_back(diverged);

        let error = exec.try_join().unwrap_err().to_string();
        assert!(error.contains("completion 0 differs"), "got: {error}");
    }

    #[test]
    fn multiproc_join_releases_each_partition_when_every_rank_reports_it() {
        let mut exec = fake_multiproc(2);
        let batch = two_partition_batch(1);
        exec.pending_partitions.insert(1, BTreeSet::from([1, 2]));
        exec.pending_batches.insert(1, batch);
        exec.inflight = 1;
        exec.buffers[0].push_back(partition_result(1, 1, 7, 1, 10));
        exec.buffers[1].push_back(partition_result(1, 1, 7, 1, 10));

        let first = exec.try_join().unwrap().unwrap();

        assert_eq!(report_partition_ids(&first), vec![1]);
        assert_eq!(exec.inflight, 1);
        assert_eq!(exec.pending_partitions.get(&1), Some(&BTreeSet::from([2])));

        exec.buffers[0].push_back(partition_result(1, 2, 8, 2, 20));
        exec.buffers[1].push_back(partition_result(1, 2, 8, 2, 20));

        let second = exec.try_join().unwrap().unwrap();

        assert_eq!(report_partition_ids(&second), vec![2]);
        assert_eq!(exec.inflight, 0);
        assert!(!exec.pending_batches.contains_key(&1));
        assert!(!exec.pending_partitions.contains_key(&1));
    }

    fn fake_multiproc(n: usize) -> MultiprocExecutor {
        let workers = (0..n).map(|rank| fake_worker(rank, n)).collect();
        MultiprocExecutor::new(workers).unwrap()
    }

    fn fake_caps(rank: usize, world_size: usize) -> WorkerCapabilities {
        let mut caps = WorkerCapabilities::default();
        caps.rank.tp_rank = rank as u32;
        caps.rank.tp_size = world_size as u32;
        caps.pipeline_depth = 2;
        caps
    }

    fn fake_worker(rank: usize, world_size: usize) -> Box<dyn Executor> {
        Box::new(FakeExec {
            caps: fake_caps(rank, world_size),
            queued: VecDeque::new(),
        })
    }

    fn queue_pending(exec: &mut MultiprocExecutor, step_ids: &[u64]) {
        for step_id in step_ids {
            let batch = test_batch(*step_id);
            exec.pending_partitions.insert(
                *step_id,
                batch
                    .partitions
                    .iter()
                    .map(|partition| partition.partition_id)
                    .collect(),
            );
            exec.pending_batches.insert(*step_id, batch);
        }
    }

    fn test_batch(step_id: u64) -> Batch {
        let key = RequestKey::new(1, RequestId(7), 1);
        let operation = Operation::registered(
            key,
            OpId(step_id.max(1)),
            VersionRef::admission_root(key, OpId(1), "0".repeat(64)),
            Work::Token(TokenMode::Decode),
            RouteId(0),
            Domain::Decode,
            Bounds {
                max_points: 1,
                max_tokens: 1,
                ..Bounds::default()
            },
            Vec::new(),
            Vec::new(),
            0,
            None,
            None,
            0,
        );
        Batch::new(
            step_id,
            Vec::new(),
            vec![BatchPartition {
                partition_id: 1,
                submission_group: 1,
                collective_seq: step_id.max(1),
                domain: Domain::Decode,
                route: RouteId(0),
                execution: ExecutionCapability::DomainHomogeneous,
                attention: AttentionRegime::Causal,
                shape_class: 0,
                operations: vec![operation],
                request_pool_indices: vec![1],
                kv_placements: Vec::new(),
                kv_branch_placements: Vec::new(),
                latent_placements: Vec::new(),
            }],
        )
    }

    fn two_partition_batch(step_id: u64) -> Batch {
        let mut batch = test_batch(step_id);
        let key = RequestKey::new(1, RequestId(8), 1);
        let operation = Operation::registered(
            key,
            OpId(2),
            VersionRef::admission_root(key, OpId(1), "0".repeat(64)),
            Work::Token(TokenMode::Decode),
            RouteId(0),
            Domain::Decode,
            Bounds {
                max_points: 1,
                max_tokens: 1,
                ..Bounds::default()
            },
            Vec::new(),
            Vec::new(),
            0,
            None,
            None,
            0,
        );
        batch.partitions.push(BatchPartition {
            partition_id: 2,
            submission_group: 2,
            collective_seq: step_id.max(1) + 1,
            domain: Domain::Decode,
            route: RouteId(0),
            execution: ExecutionCapability::DomainHomogeneous,
            attention: AttentionRegime::Causal,
            shape_class: 0,
            operations: vec![operation],
            request_pool_indices: vec![2],
            kv_placements: Vec::new(),
            kv_branch_placements: Vec::new(),
            latent_placements: Vec::new(),
        });
        batch
    }

    fn completion(
        op: u64,
        committed_tokens: Vec<u32>,
        product_generations: Vec<u32>,
    ) -> CompletionRecord {
        CompletionRecord {
            request_key: RequestKey::new(1, RequestId(7), 1),
            op_id: OpId(op.max(1)),
            completion_slot_generation: 1,
            status: OpStatus::Ok,
            selected_point: 0,
            logical_lengths: LogicalLengths::default(),
            token_span: TokenSpan::default(),
            committed_tokens,
            finish_flags: FinishFlags::default(),
            product_generations,
            semantic_digest: "0".repeat(64),
            error_code: None,
            timing_counters: TimingCounters::default(),
        }
    }

    fn result(step_id: u64, sampled_token_id: u32) -> CompletionReport {
        CompletionReport {
            step_id,
            partitions: vec![PartitionCompletion {
                partition_id: 1,
                completions: vec![completion(step_id, vec![sampled_token_id], Vec::new())],
                products: Vec::new(),
                registration: RegistrationAck::default(),
                worker_exec_us: Some(step_id),
                forward_stats: None,
            }],
        }
    }

    fn partition_result(
        step_id: u64,
        partition_id: u32,
        session_id: u64,
        op_id: u64,
        sampled_token_id: u32,
    ) -> CompletionReport {
        let mut record = completion(op_id, vec![sampled_token_id], Vec::new());
        record.request_key = RequestKey::new(1, RequestId(session_id), 1);
        CompletionReport {
            step_id,
            partitions: vec![PartitionCompletion {
                partition_id,
                completions: vec![record],
                products: Vec::new(),
                registration: RegistrationAck::default(),
                worker_exec_us: Some(step_id),
                forward_stats: None,
            }],
        }
    }

    fn result_with_products(
        step_id: u64,
        product_generations: Vec<u32>,
        payload_generation: u32,
    ) -> CompletionReport {
        CompletionReport {
            step_id,
            partitions: vec![PartitionCompletion {
                partition_id: 1,
                completions: vec![completion(step_id, vec![10], product_generations)],
                products: vec![product_payload(payload_generation)],
                registration: RegistrationAck::default(),
                worker_exec_us: Some(step_id),
                forward_stats: None,
            }],
        }
    }

    fn product_payload(generation: u32) -> ProductPayload {
        ProductPayload {
            product: ProductRef {
                request_key: RequestKey::new(1, RequestId(7), 1),
                producer_op_id: OpId(1),
                output_index: 0,
                generation,
                kind: ProductKind::Kv,
                storage_class: StorageClass::PagedKv,
                dtype: DType::BF16,
                shape_bound: ShapeBound::default(),
                point_range: PointRange::default(),
            },
            bytes: vec![generation as u8],
        }
    }

    struct FakeExec {
        caps: WorkerCapabilities,
        queued: VecDeque<CompletionReport>,
    }

    impl Executor for FakeExec {
        fn caps(&self) -> WorkerCapabilities {
            self.caps.clone()
        }

        fn pipeline_depth(&self) -> usize {
            2
        }

        fn in_flight(&self) -> usize {
            self.queued.len()
        }

        fn submit(&mut self, _batch: Batch) -> anyhow::Result<()> {
            Ok(())
        }

        fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
            Ok(self.queued.pop_front())
        }

        fn wait_result_timeout(
            &mut self,
            _timeout: Duration,
        ) -> anyhow::Result<Option<CompletionReport>> {
            self.poll()
        }

        fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
            self.queued
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("no fake result"))
        }

        fn control(&mut self, _op: ControlOp) -> anyhow::Result<u64> {
            Ok(0)
        }

        fn control_wait(
            &mut self,
            _op: ControlOp,
            _targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            Ok(Vec::new())
        }
    }
}
